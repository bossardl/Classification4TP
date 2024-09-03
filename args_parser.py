import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Train a model on image data")

    # Arguments
    parser.add_argument('--image_path', type=str, required=True,
                        help='Path to the image directory containing jpg images.')
    parser.add_argument('--label_path', type=str, required=True,
                        help='Path to the CSV file containing image labels.')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size for training (default: 128).')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate for training (default: 0.001).')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs for training (default: 100).')
    parser.add_argument('--epoch_interval', type=int, default=5,
                        help='epoch_interval for training evaluation (default: 5).')
    parser.add_argument('--training_mode', type=str, default='CrossVal',
                    help='Training mode crossVal or only one pass (default: CrossVal).')
    
    return parser.parse_args()
