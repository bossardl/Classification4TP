import argparse
from pathlib import Path
import pandas as pd
from preprocessing import load_and_normalize_images, read_label, get_sample, get_data
from networks import build_cnn_model
from trainer import train_and_evaluate_model_classWeight, train_and_evaluate_model_crossVal

def main(image_directory, label_path, batch_size, learning_rate, epochs, epoch_interval, model_type, training_mode, method, undersampling, oversampling):
    import numpy as np
    
    # Load normalized images and target
    normalized_image_set = load_and_normalize_images(Path(image_directory))
    print('\n')
    
    # Read the labels
    labels = read_label(label_path)
    
    # Sampling of few indices for ploting (set flag_display=True)
    indices = get_sample(normalized_image_set, labels, n_draw=4)
    print('\n')
    X, y = get_data(normalized_image_set,labels, indices)

    if method is None:
        pass

    elif method=='undersampling':
        from imblearn.under_sampling import RandomUnderSampler
        print("Undersampling of the majority class to rebalance the set")

        undersampler = RandomUnderSampler(sampling_strategy=undersampling, random_state=42)
        X_flattened = X.reshape(X.shape[0], -1) 
        X_resampled, y_resampled = undersampler.fit_resample(X_flattened, y)
        #Shuffle the dataset
        X_resampled = X_resampled.reshape(-1, 64, 64, 3)
        order = np.arange(len(y_resampled))
        np.random.shuffle(order)
        X = X_resampled[order]
        y = y_resampled[order]
        print('\n')
        print(f'Updated inforamtion on data: \n ')
        X, y = get_data(X,y, indices)
        print('\n')

    elif method=='oversampling':
        print("Oversampling the minority class to rebalance the set")
        
        pos_labels = np.argwhere(y == 0).flatten()
        neg_labels = np.argwhere(y == 1).flatten()

        pos_features = X[pos_labels]
        neg_features = X[neg_labels]

        ids = np.arange(len(pos_features))
        choices = np.random.choice(ids, int(len(neg_features)*oversampling), replace=True)  # Ensure replace=True for oversampling

        res_pos_features = pos_features[choices]
        res_pos_labels = np.zeros(len(res_pos_features))  # Since pos_labels are all 0, no need to sample

        # Concatenate features and labels
        X_resampled = np.concatenate([res_pos_features, neg_features], axis=0)
        y_resampled = np.concatenate([res_pos_labels, np.ones(len(neg_features))], axis=0)
        
        #Shuffle the dataset
        order = np.arange(len(y_resampled))
        np.random.shuffle(order)
        X = X_resampled[order]
        del X_resampled
        y = y_resampled[order]
        del y_resampled
        print('\n')
        print(f'Updated inforamtion on data: \n ')
        X, y = get_data(X,y, indices)
        print('\n')

    else:
        print("Unknown method selected. Please choose 'undersampling' or 'oversampling'.")
        exit
    
    

    print('##################################################################')
    print('\n')
    print('\n')
    print('###################  TRAINING & EVALUATION   ##################### ')


    # Build the model
    if model_type=='simple':
        model = build_cnn_model(image_shape=(64, 64, 3), learning_rate=args.lr, model_type=model_type)
    print('\n')
    print(model.summary()) # Print the model parameters
    print('\n')
    if training_mode == 'CrossVal':
        # Training with cross Validation and class weight to counterbalance 
        # the imbalance of the labels(more 1s than 0s)
        model, _, _ = train_and_evaluate_model_crossVal(X, y, args, batch_size=batch_size, learning_rate=learning_rate, epochs=epochs, epoch_interval=epoch_interval)
    elif training_mode == 'Val':
        # Faster training with stratified train set and val set. Fitting with class Weight 
        model, _, _ = train_and_evaluate_model_classWeight(X, y, args, batch_size=batch_size, learning_rate=learning_rate, epochs=epochs, epoch_interval=epoch_interval)
    print('\n')
    print('\n')
    print('##################################################################')
    


if __name__ == "__main__":
    from args_parser import get_args  
    args = get_args()
    
    main(args.image_path, args.label_path, args.batch_size, 
         args.lr, args.epochs, args.epoch_interval, args.model_type, args.training_mode,
          args.method, args.undersampling, args.oversampling)