import argparse
from pathlib import Path
import pandas as pd
from preprocessing import load_and_normalize_images, read_label, get_sample
from networks import build_cnn_model
from trainer import train_and_evaluate_model_classWeight, train_and_evaluate_model_crossVal

def main(
        args,
        image_directory: str,
        label_path: str,
        batch_size: int,
        lr: float,
        epochs: int,
        epoch_interval: int,
        model_type: str,
        training_mode: str,
        method: str,
        undersampling: float,
        oversampling: float
):

    """
    Main function to load data, preprocess it, and train a model based on the specified parameters.
    
    This function handles data loading, normalization, and manages class imbalance using the specified 
    sampling method (undersampling or oversampling). It then loads and trains a model, tracks performance 
    using callbacks, and evaluates the model using the best-performing epochs.

    Parameters:
        image_directory (str): Path to the directory containing the images.
        label_path (str): Path to the directory containing the labels.
        batch_size (int): Number of samples per batch.
        lr (float): Learning rate for the optimizer.
        epochs (int): Number of epochs to train the model.
        epoch_interval (int): Interval for saving metrics and outputs (e.g., images, predicted labels).
        model_type (str): Model architecture to use; options typically include "base model (6M)" or "smaller (800k)".
        training_mode (str): Training mode; options may include "classic evaluation" or "cross-validation".
        method (str): Strategy for handling class imbalance; options include "undersampling" or "oversampling".
        undersampling (float): Factor for undersampling (e.g., 1.0 to match the size of the minority class).
        oversampling (float): Factor for oversampling (e.g., 1.0 to match the size of the majority class).
        
    Returns:
        None: The function runs the entire training pipeline but does not return any value.
    """

        
    import numpy as np
    
    # Load normalized images and target
    X = load_and_normalize_images(Path(image_directory))
    print('\n')
    
    # Read the labels
    y = read_label(label_path)

    # Sampling of few indices for ploting (set flag_display=True)
    get_sample(X, y, n_draw=4, flag_display=False)
    print('\n')

    print('##################################################################')
    print('\n')
    print('\n')
    print('###################  TRAINING & EVALUATION   ##################### ')


    # Build the model
    model = build_cnn_model(image_shape=(64, 64, 3), learning_rate=lr, model_type=model_type)
    print('\n')
    print(model.summary()) # Print the model parameters
    print('\n')

    
    if training_mode == 'CrossVal':
        # Training with cross Validation and class weight to counterbalance 
        # the imbalance of the labels(more 1s than 0s)
        model, _, _ = train_and_evaluate_model_crossVal(X, y, args,n_splits=5)
    elif training_mode == 'Val':
        # Faster training with stratified train set and val set. Fitting with class Weight 
        model, _, _ = train_and_evaluate_model_classWeight(X, y, args)
    print('\n')
    print('\n')
    print('##################################################################')
    


if __name__ == "__main__":
    from args_parser import get_args  
    args = get_args()
    
    main(args, args.image_path, args.label_path, args.batch_size, 
         args.lr, args.epochs, args.epoch_interval, args.model_type, args.training_mode,
          args.method, args.undersampling, args.oversampling)