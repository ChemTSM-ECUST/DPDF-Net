from pathlib import Path

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler


class TEDataset(Dataset):
    def __init__(self, data_path, input_features, target_features, seq_length=12, pred_steps=1,
                 scaler_X=None, scaler_y=None, is_train=True, row_start=0, row_end=None, stride=1):

        self.pred_steps = pred_steps
        data = np.loadtxt(data_path)[row_start:row_end]
        self.row_start = row_start
        self.row_end = row_start + len(data)
        self.stride = stride
        if any(not isinstance(value, int) or value < 1 for value in (seq_length, pred_steps, stride)):
            raise ValueError("seq_length, pred_steps and stride must be positive integers.")
        if len(data) < seq_length + pred_steps:
            raise ValueError("Each data segment must contain at least seq_length + pred_steps rows.")
        input_data = data[:, input_features]
        target_data = data[:, target_features].reshape(-1, len(target_features))

        if is_train:
            self.scaler_X = StandardScaler()
            self.scaler_y = StandardScaler()
            self.input_data_scaled = self.scaler_X.fit_transform(input_data)
            self.target_data_scaled = self.scaler_y.fit_transform(target_data)
        else:
            if scaler_X is None or scaler_y is None:
                raise ValueError("For validation/test data, scalers must be provided.")
            self.scaler_X = scaler_X
            self.scaler_y = scaler_y
            self.input_data_scaled = self.scaler_X.transform(input_data)
            self.target_data_scaled = self.scaler_y.transform(target_data)

        self.seq_length = seq_length

    def __len__(self):
        return (len(self.input_data_scaled) - self.seq_length - self.pred_steps) // self.stride + 1

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self):
            raise IndexError("Window index out of range.")
        idx = idx * self.stride
        x = self.input_data_scaled[idx: idx + self.seq_length]
        y = self.target_data_scaled[idx + self.seq_length: idx + self.seq_length + self.pred_steps]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


def _validate_data_paths(config):
    if Path(config["train_data_path"]).samefile(config["test_data_path"]):
        raise ValueError("Training and test data must be separate source files.")


def create_data_loaders(config):
    _validate_data_paths(config)
    num_rows = len(np.loadtxt(config["train_data_path"]))
    train_ratio = config.get("train_ratio", 0.8)
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1.")
    split_row = int(num_rows * train_ratio)
    train_dataset = TEDataset(
        data_path=config["train_data_path"],
        input_features=config["input_features"],
        target_features=config["target_features"],
        seq_length=config["seq_length"],
        pred_steps=config["pred_steps"],
        is_train=True,
        row_end=split_row,
        stride=config.get("stride", 1)
    )

    val_dataset = TEDataset(
        data_path=config["train_data_path"],
        input_features=config["input_features"],
        target_features=config["target_features"],
        seq_length=config["seq_length"],
        pred_steps=config["pred_steps"],
        scaler_X=train_dataset.scaler_X,
        scaler_y=train_dataset.scaler_y,
        is_train=False,
        row_start=split_row,
        stride=config.get("stride", 1)
    )
    print(
        f"Chronological split: train_ratio={train_ratio}, "
        f"length={config['seq_length']}, stride={config.get('stride', 1)}, "
        f"prediction_steps={config['pred_steps']}"
    )
    print(f"Train rows: 1-{split_row}; windows: {len(train_dataset)}")
    print(f"Validation rows: {split_row + 1}-{num_rows}; windows: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)

    return train_loader, val_loader, train_dataset


def create_test_loader(config, scaler_X, scaler_y):
    _validate_data_paths(config)
    test_dataset = TEDataset(
        data_path=config["test_data_path"],
        input_features=config["input_features"],
        target_features=config["target_features"],
        seq_length=config["seq_length"],
        pred_steps=config["pred_steps"],
        scaler_X=scaler_X,
        scaler_y=scaler_y,
        is_train=False,
        stride=config.get("stride", 1)
    )
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)
    print(f"Test rows: 1-{test_dataset.row_end}; windows: {len(test_dataset)}")

    return test_loader, test_dataset
