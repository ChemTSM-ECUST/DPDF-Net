# DPDF-Net: Dual-Path Deep Fusion Network

This is a deep learning model for time series prediction, particularly suitable for Tennessee Eastman process data.

## Project Structure

```
├── utils.py                # Utility functions (random seed setting, tag building, etc.)
├── model.py                # Model architecture definition
├── data_provider.py        # Data processing and loading
├── train.py                # Training and evaluation functions
├── run.py                  # Main program entry
├── requirements.txt        # Dependency package list
└── README.md               # Project description
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Data Preparation

Make sure you have the Tennessee Eastman dataset:

- Training data: `tennessee-eastman-profBraatz-master/d01.dat`
- Test data: `tennessee-eastman-profBraatz-master/d01_te.dat`

## Run Program

### Basic Run

```bash
python run.py
```

### Custom Configuration

Modify the `config` dictionary in `run.py`:

```python
config = {
    "train_data_path": "tennessee-eastman-profBraatz-master/d01.dat",
    "test_data_path": "tennessee-eastman-profBraatz-master/d01_te.dat",
    "time_step": 12,
    "pred_steps": 1,
    "hidden_size": 64,
    "learning_rate": 0.001,
    "batch_size": 16,
    "num_epochs": 500,
    "patience": 200,
    "target_col": 30
}
```

### Parameter Explanation

**Data Parameters:**
- `train_data_path`: Path to training data file
- `test_data_path`: Path to test data file
- `time_step`: Historical window size (default: 12)
- `pred_steps`: Prediction steps (default: 1)
- `target_col`: Target variable column index

**Model Parameters:**
- `hidden_size`: Hidden layer dimension (default: 64)

**Training Parameters:**
- `learning_rate`: Learning rate (default: 0.001)
- `batch_size`: Batch size (default: 16)
- `num_epochs`: Maximum training epochs (default: 500)
- `patience`: Early stopping patience (default: 200)

## Model Architecture

### PSA_DW_DyConv Module

- Multi-scale dilated depthwise-separable convolutions
- ECA channel attention (1D convolution)
- Dynamic dilation coefficients for different scales

### DPDF_Net Architecture

1. **PSA_DW_DyConv**: Multi-scale feature extraction with channel attention
2. **Dual-branch encoder**:
   - Transformer branch: Long-range temporal dependencies
   - LSTM branch: Sequential patterns
3. **TFAM**: Temporal-spatial feature attention fusion module
4. **Soft attention pooling**: Weighted aggregation of temporal features

## Module Function Description

### utils.py

- `set_random_seed()`: Set random seed to ensure reproducible results
- `build_run_tag()`: Build run tag for model saving
- `kernel_size()`: Calculate appropriate convolution kernel size

### model.py

Contains definitions of all neural network modules, including attention mechanisms, convolution modules, and main network architecture.

### data_provider.py

- `TEDataset`: Tennessee Eastman dataset class
- `create_data_loaders()`: Create training and validation data loaders
- `create_test_loader()`: Create test data loader

### train.py

- `train_model()`: Model training function with early stopping mechanism
- `evaluate()`: Model evaluation function that calculates various metrics

### run.py

Main program entry that integrates all modules and executes the complete training and evaluation workflow.

## Citation

If you use this code in your research, please cite our paper:

```
[Paper citation will be added upon publication]
```
