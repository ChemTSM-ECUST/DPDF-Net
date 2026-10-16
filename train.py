import torch
import torch.nn as nn
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from model import DPDF_Net
from data_provider import create_data_loaders, create_test_loader
from utils import build_run_tag


def train_model(config):
    train_loader, val_loader, full_dataset = create_data_loaders(config)

    input_size = len(config["input_features"])
    output_size = len(config["target_features"])
    model = DPDF_Net(input_size, config["hidden_size"], output_size, config["pred_steps"])

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)

    early_stopping_patience = 200
    epochs_no_improve = 0
    best_val_loss = float('inf')

    tag = build_run_tag(model, config)
    best_model_path = f"best_model_{tag}.pth"

    print(f"Starting to train model: {tag}")
    print(f"Training data: {len(train_loader)} batches")
    print(f"Validation data: {len(val_loader)} batches")

    for epoch in range(config["num_epochs"]):
        model.train()
        total_train_loss = 0.0
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            pred = model(x_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for x_val, y_val in val_loader:
                pred_val = model(x_val)
                val_loss = criterion(pred_val, y_val)
                total_val_loss += val_loss.item()

        avg_train_loss = total_train_loss / max(len(train_loader), 1)
        avg_val_loss = total_val_loss / max(len(val_loader), 1)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_model_path)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve == early_stopping_patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

        scheduler.step(avg_val_loss)

        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch + 1}/{config['num_epochs']}, "
                  f"Train Loss: {avg_train_loss:.6f}, "
                  f"Val Loss: {avg_val_loss:.6f}")

    model.load_state_dict(torch.load(best_model_path, weights_only=True))
    print(f"Training completed, best validation loss: {best_val_loss:.6f}")

    return model, full_dataset.scaler_X, full_dataset.scaler_y


def evaluate(model, config, scaler_X, scaler_y):
    test_loader, test_dataset = create_test_loader(config, scaler_X, scaler_y)

    model.eval()
    all_preds, all_targets = [], []

    print("Starting model evaluation...")

    with torch.no_grad():
        for x_test, y_test in test_loader:
            pred = model(x_test)
            all_preds.append(pred.detach().cpu().numpy())
            all_targets.append(y_test.detach().cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    preds_inv = np.zeros_like(all_preds)
    targets_inv = np.zeros_like(all_targets)
    output_size = len(config["target_features"])

    if output_size == 1:
        preds_inv_flat = scaler_y.inverse_transform(all_preds.reshape(-1, 1))
        targets_inv_flat = scaler_y.inverse_transform(all_targets.reshape(-1, 1))
        preds_inv = preds_inv_flat.reshape(all_preds.shape)
        targets_inv = targets_inv_flat.reshape(all_targets.shape)
    else:
        for step in range(config["pred_steps"]):
            preds_inv[:, step, :] = scaler_y.inverse_transform(all_preds[:, step, :])
            targets_inv[:, step, :] = scaler_y.inverse_transform(all_targets[:, step, :])

    feature_to_plot_idx = 0
    step_to_plot_idx = 0

    y_true = targets_inv[:, step_to_plot_idx, feature_to_plot_idx]
    y_pred = preds_inv[:, step_to_plot_idx, feature_to_plot_idx]

    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print("=" * 50)
    print("Model evaluation results:")
    print(f"MSE: {mse:.6f}")
    print(f"MAE: {mae:.6f}")
    print(f"R²: {r2:.6f}")
    print("=" * 50)

    return {
        'mse': mse,
        'mae': mae,
        'r2': r2,
        'predictions': preds_inv,
        'targets': targets_inv
    }