import torch
import numpy as np
import random
import os
import math


def set_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_run_tag(model, config):
    model_name = str(config.get("model_name", model.__class__.__name__)).lower()
    if model_name.endswith("model"):
        model_name = model_name[:-5]
    data_name = os.path.splitext(os.path.basename(config["train_data_path"]))[0]
    return f"{model_name}_{data_name}"


def kernel_size(in_channel):
    k = int((math.log2(in_channel) + 1) // 2)
    return k + 1 if k % 2 == 0 else k