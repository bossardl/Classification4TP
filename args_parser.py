import argparse
import sys 

def get_args():
    """
    Function to parse the arugments entered in a command line.

    Parameters:
        
    Returns:
        parser.parse_args(): Structure of object with arguments accessible by instantiation
    """
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
    
    parser.add_argument('--model_type', type=str, default='simple',
                        help='Model type in [simple, complex].')
    parser.add_argument('--training_mode', type=str, default='CrossVal',
                        help='Training mode crossVal or only one pass (default: CrossVal).')
    parser.add_argument('--method', type=str, default=None, choices=['undersampling', 'oversampling', 'None'],
                        help='Method to tackle imbalance of the majority class ["undersampling", "oversampling", None] (default: None).')
    
    parser.add_argument('--undersampling', type=float, default=None,
                        help='Percentage of undersampling of the majority class in [0,1] (required if --method is "undersampling").')
    
    parser.add_argument('--oversampling', type=float, default=None,
                        help='Percentage of oversampling of the minority class in [0,1] (required if --method is "oversampling").')
    
    args = parser.parse_args()
    
    # Check for --undersampling value if method is undersampling
    if args.method == 'undersampling':
        if args.undersampling is None or not (0 < args.undersampling <= 1):
            print('Error: --undersampling must be specified and between 0 and 1 when --method is "undersampling".')
            sys.exit(1)
    
    # Check for --oversampling value if method is oversampling
    elif args.method == 'oversampling':
        if args.oversampling is None or not (0 < args.oversampling <= 1):
            print('Error: --oversampling must be specified and between 0 and 1 when --method is "oversampling".')
            sys.exit(1)
    
    
    return args