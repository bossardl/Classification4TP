import argparse
from pathlib import Path
import pandas as pd
from preprocessing import load_and_normalize_images, read_label, get_sample, get_data
from networks import build_cnn_model
from trainer import train_and_evaluate_model_classWeight, train_and_evaluate_model_crossVal

def main(image_directory, label_path, batch_size, learning_rate, epochs, epoch_interval, training_mode):
    # Load normalized images and target
    normalized_image_set = load_and_normalize_images(Path(image_directory))
    print('\n')
    labels = read_label(label_path)
    indices = get_sample(normalized_image_set, labels, n_draw=4)
    print('\n')
    X, y = get_data(normalized_image_set,labels, indices)
    X, y = X[:1000], y[:1000] 
    print('##################################################################')
    print('\n')
    print('\n')
    print('###################  TRAINING & EVALUATION   ##################### ')

    model = build_cnn_model(image_shape=(64, 64, 3), learning_rate=args.lr)
    print('\n')
    print(model.summary())
    print('\n')
    if training_mode == 'CrossVal':
        model, _, _ = train_and_evaluate_model_crossVal(X, y, batch_size=batch_size, learning_rate=learning_rate, epochs=epochs, epoch_interval=epoch_interval)
    else:
        model, _, _ = train_and_evaluate_model_classWeight(X, y, batch_size=batch_size, learning_rate=learning_rate, epochs=epochs, epoch_interval=epoch_interval)
    print('\n')
    print('\n')
    print('##################################################################')
    


if __name__ == "__main__":
    from args_parser import get_args  
    args = get_args()
    
    main(args.image_path, args.label_path, args.batch_size, args.lr, args.epochs, args.epoch_interval, args.training_mode)