from utils import set_random_seed
from train import train_model, evaluate


def get_default_config():
    paper_input_features = [
        0, 1, 2, 3, 4, 5, 8, 9, 10, 12, 13, 16, 17, 18, 20, 21
    ]
    paper_target_features = [30]

    config = {
        "model_name": "dpdf",
        "train_data_path": "tennessee-eastman-profBraatz-master/d01.dat",
        "test_data_path": "tennessee-eastman-profBraatz-master/d01_te.dat",
        "input_features": paper_input_features,
        "target_features": paper_target_features,
        "seq_length": 12,
        "pred_steps": 1,
        "hidden_size": 64,
        "num_epochs": 500,
        "batch_size": 16,
        "learning_rate": 0.001,
    }
    return config


def main():
    set_random_seed(42)

    config = get_default_config()

    print("DPDF-Net training started")
    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()

    trained_model, scaler_X, scaler_y = train_model(config)

    results = evaluate(trained_model, config, scaler_X, scaler_y)

    print("Training and evaluation completed!")


if __name__ == '__main__':
    main()