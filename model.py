import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import kernel_size


class ChannelAttention(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        k = kernel_size(in_channel)
        self.conv1 = nn.Conv1d(4, 1, kernel_size=k, padding=k // 2)
        self.conv2 = nn.Conv1d(4, 1, kernel_size=k, padding=k // 2)
        self.softmax = nn.Softmax(dim=0)

    def forward(self, t1, t2):
        t1_avg, t1_max = self.avg_pool(t1), self.max_pool(t1)
        t2_avg, t2_max = self.avg_pool(t2), self.max_pool(t2)
        pool = torch.cat([t1_avg, t1_max, t2_avg, t2_max],
                         dim=2).squeeze(-1).transpose(1, 2)
        attn1 = self.conv1(pool)
        attn2 = self.conv2(pool)
        stack = torch.stack([attn1, attn2], dim=0)
        stack = self.softmax(stack).transpose(-1, -2).unsqueeze(-1)
        return stack


class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 1, kernel_size=7, padding=3)
        self.conv2 = nn.Conv2d(4, 1, kernel_size=7, padding=3)
        self.softmax = nn.Softmax(dim=0)

    def forward(self, t1, t2):
        t1_avg, t1_max = torch.mean(t1, dim=1, keepdim=True), torch.max(t1, dim=1, keepdim=True)[0]
        t2_avg, t2_max = torch.mean(t2, dim=1, keepdim=True), torch.max(t2, dim=1, keepdim=True)[0]
        pool = torch.cat([t1_avg, t1_max, t2_avg, t2_max], dim=1)
        attn1 = self.conv1(pool)
        attn2 = self.conv2(pool)
        stack = torch.stack([attn1, attn2], dim=0)
        return self.softmax(stack)


class TFAM(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.ca = ChannelAttention(in_channel)
        self.sa = SpatialAttention()

    def forward(self, t1, t2):
        t1_ = t1.transpose(1, 2).unsqueeze(-1)
        t2_ = t2.transpose(1, 2).unsqueeze(-1)
        w = self.ca(t1_, t2_) + self.sa(t1_, t2_) + 1
        fuse = w[0].squeeze(-1).transpose(1, 2) * t1 + \
               w[1].squeeze(-1).transpose(1, 2) * t2
        return fuse


class SoftAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        alpha = torch.softmax(self.fc(x).squeeze(-1), dim=1)
        return torch.bmm(alpha.unsqueeze(1), x).squeeze(1)


class PSA_DW_DyConv(nn.Module):
    def __init__(self, channel, S=4, k_eca=3):
        super().__init__()
        self.S = S
        assert channel % S == 0
        split_c = channel // S

        self.dw_convs = nn.ModuleList([
            nn.Conv2d(split_c, split_c,
                      kernel_size=3,
                      padding=1 + i, dilation=1 + i,
                      groups=split_c, bias=False)
            for i in range(S)
        ])
        self.pw_conv = nn.Conv2d(channel, channel, kernel_size=1, bias=False)

        self.eca = nn.Conv1d(1, 1, kernel_size=k_eca,
                             padding=(k_eca - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        B, C, H, W = x.size()
        splits = torch.chunk(x, self.S, dim=1)

        y = torch.cat([conv(s) for conv, s in zip(self.dw_convs, splits)], dim=1)
        y = self.pw_conv(y)

        w = F.adaptive_avg_pool2d(y, 1).view(B, 1, C)
        w = self.sigmoid(self.eca(w)).view(B, C, 1, 1)
        return y * w


class DPDF_Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, pred_steps=1, S=4):
        super().__init__()
        self.pred_steps = pred_steps
        self.output_size = output_size

        self.input_size = input_size
        if input_size % S != 0:
            pad_size = S - (input_size % S)
            self.input_size = input_size + pad_size
            self.pad_input = True
            self.pad_size = pad_size
        else:
            self.pad_input = False

        self.psa = PSA_DW_DyConv(self.input_size, S=S)
        self.linear = nn.Linear(self.input_size, hidden_size)

        self.trans = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden_size, nhead=8, batch_first=True), 4)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=3,
                            batch_first=True, bidirectional=False)
        self.tfam = TFAM(hidden_size)

        self.pool = SoftAttention(hidden_size)
        self.fc = nn.Linear(hidden_size, output_size * pred_steps)

    def forward(self, x):
        if self.pad_input:
            pad = torch.zeros(x.size(0), x.size(1), self.pad_size, device=x.device)
            x = torch.cat([x, pad], dim=2)

        x2d = self.psa(x.permute(0, 2, 1).unsqueeze(-1))
        x = self.linear(x2d.squeeze(-1).permute(0, 2, 1))

        t = self.trans(x)
        l, _ = self.lstm(x)

        t = t + x
        l = l + x

        f = self.tfam(t, l)

        feat = self.pool(f)
        out = self.fc(feat)
        out = out.view(-1, self.pred_steps, self.output_size)
        return out
