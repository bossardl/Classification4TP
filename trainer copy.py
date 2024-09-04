import os
import yaml
import numpy as np
import tensorflow as tf
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from networks import build_cnn_model
from evaluation import evaluate_model
from utils import PerformancePlotCallback, SaveBestModelCallback

def train_and_evaluate_model_classWeight(X, y, args, batch_size=32, learning_rate=0.001, epochs=100, epoch_interval=5):
    """
    Implement HTER (Half Total Error Rate)metrics

    Parameters:
        y_true (np.ndarray): A NumPy array of y ground truth labels.
        y_pred (np.ndarray): A NumPy array of y prediction.

        
    Returns:
        float: result of the metric HTER.
    """
        
    image_shape = (64, 64, 3)
    metrics_train :dict = {'HTER':None , 'f1_score':None , 'roc_auc':None}
    metrics_evaluation :dict = {'HTER':None , 'f1_score':None , 'roc_auc':None}

    # Setting up the logs directory
    start_time = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    log_dir = f'run/log_{start_time}_epochs_{epochs}_lr_{learning_rate}_batch_size_{batch_size}_{args.training_mode}_method_{args.method}'
    
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    

    # Load the arguments and parameters for experiment tracking
    args = {
    'image_path': args.image_path,
    'label_path': args.label_path,
    'batch_size': args.batch_size,
    'lr': args.lr,
    'epochs': args.epochs,
    'epoch_interval': args.epoch_interval,
    'training_mode': args.training_mode,
    'method': args.method,
    'undersampling': args.undersampling,
    'oversampling': args.oversampling
    }

    params = {
        'run': f'log_{start_time}'
    }
    
    # Create the YAML files, file writer to store the metrics
    yaml_file_path = os.path.join(log_dir, 'parameters.yaml')
    with open(yaml_file_path, 'w') as file:
        yaml.dump(params, file, default_flow_style=False)
    
    file_writer = tf.summary.create_file_writer(log_dir + "/metrics")
    file_writer_train = tf.summary.create_file_writer(log_dir + "/metrics_train")
    file_writer_val = tf.summary.create_file_writer(log_dir + "/metrics_val")

    # Callbacks
    

    # Set up the Checkpoint and EarlyStopping
    checkpoint_path = os.path.join(log_dir, "cnn_model.keras")
    checkpoint_callback = ModelCheckpoint(checkpoint_path, save_best_only=True) #retrain from checkpoint

    best_model_path = os.path.join(log_dir, 'best_model.keras')
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True) #Stop after no improvement of val_loss
    

    # Train Validation Split of the data to balance bias and variance of the training i.e over- and underfitting
    x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)


    print('\n')
    print('\n')
    print('############################ SUMMARY ##############################')
    print(f'Loaded {len(X)} images in total.')
    print(f'Loaded {len(x_train)} images for training.')
    print(f'Loaded {len(x_val)} images for validation.')
    print(f'Using batch size of {batch_size}, learning rate of {learning_rate}, for {epochs} epochs.')
    print('##################################################################')

    y_train = y_train.flatten() 
    y_val = y_val.flatten() 
    class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
    class_weights_dict = dict(enumerate(class_weights))
    
    model = build_cnn_model(image_shape=image_shape, learning_rate=learning_rate)
    
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    performance_train = PerformancePlotCallback(x_train, y_train, "CNN_model_train", file_writer_train, epoch_interval=epoch_interval, mode='train')
    performance_val = PerformancePlotCallback(x_val, y_val, "CNN_model_val", file_writer_val, epoch_interval=epoch_interval, mode='val')
    save_best_model = SaveBestModelCallback(x_val=x_val, y_val=y_val, file_path=best_model_path)
    
    print('\n')
    print('############################ TRAINING ############################ ')
    print('                             TRAINING                ')
    print('\n')
    history = model.fit(x_train, y_train, 
                        epochs=epochs, 
                        batch_size=batch_size, 
                        validation_data=(x_val, y_val),
                        callbacks=[early_stopping, checkpoint_callback, performance_train, performance_val, save_best_model, tensorboard_callback],
                        class_weight=class_weights_dict,  
                        verbose=1)
    
    print('\n')
    print('                PREDICTION                ')
    print('\n')
    y_pred_prob_train = model.predict(x_train)
    y_pred_prob_val = model.predict(x_val)
    print('\n')
    print('                             EVALUATION                ')
    print('\n')
    HTER_train, HTER_val, f1_train, f1_val, roc_auc_train, roc_auc_val = evaluate_model(y_pred_prob_train, y_train, y_pred_prob_val, y_val, threshold = 0.5)
    metrics_train['HTER'], metrics_train['f1_score'], metrics_train['roc_auc']= HTER_train, f1_train, roc_auc_train
    metrics_evaluation['HTER'], metrics_evaluation['f1_score'], metrics_evaluation['roc_auc']= HTER_val, f1_val, roc_auc_val
    


    print('                             SAVING                ')
    # Saving the metrics for Training and validation
    metrics_file = os.path.join(log_dir, f'performance_metrics.txt')
    with open(metrics_file, 'w') as f:
        f.write(f"HTER Train: {HTER_train:.6f}\n")
        f.write(f"HTER Validation: {HTER_val:.6f}\n")
        f.write(f"F1 Score Train: {f1_train:.6f}\n")
        f.write(f"F1 Score Validation: {f1_val:.6f}\n")
        f.write(f"ROC AUC Train: {roc_auc_train:.6f}\n")
        f.write(f"ROC AUC Validation: {roc_auc_val:.6f}\n")


    # Save the experiment data 
    experiment_data = {
            'arguments': args,
            'parameters': params,
            'performance': {
                'HTER_train': f"{HTER_train:.6f}",
                'HTER_val': f"{HTER_val:.6f}",
                'f1_train': f"{f1_train:.6f}",
                'f1_val': f"{f1_val:.6f}",
                'roc_auc_train': f"{roc_auc_train:.6f}",
                'roc_auc_val': f"{roc_auc_val:.6f}"
            },
        }

    combined_file_path = os.path.join(log_dir, 'experiment_tracking.yaml')
    with open(combined_file_path, 'w') as file:
        yaml.dump(experiment_data, file, default_flow_style=False)
    print(f"Saved performance metrics in {metrics_file}\n")
    
    return model, metrics_train, metrics_evaluation




def train_and_evaluate_model_crossVal(X, y, args, batch_size=32, learning_rate=0.001, epochs=100, n_splits=5, epoch_interval=5):
    from sklearn.model_selection import StratifiedKFold
    from tqdm import tqdm 

    # Initialization global
    image_shape = (64, 64, 3)
    metrics_train :dict = {'HTER':[] , 'f1_score':[] , 'roc_auc':[]}
    metrics_evaluation :dict = {'HTER':[] , 'f1_score':[] , 'roc_auc':[]}
    start_time = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

    log_dir = f'run/log_{start_time}_epochs_{epochs}_lr_{learning_rate}_batch_size_{batch_size}_{args.training_mode}_method_{args.method}'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)


    # Loading arguments of the model
    args = {
    'image_path': args.image_path,
    'label_path': args.label_path,
    'batch_size': args.batch_size,
    'lr': args.lr,
    'epochs': args.epochs,
    'epoch_interval': args.epoch_interval,
    'training_mode': args.training_mode,
    'method': args.method,
    'undersampling': args.undersampling,
    'oversampling': args.oversampling
    }

    params = {
        'run': f'log_{start_time}'
    }
    
    

    # Stratidied split to balance imbalances in each split
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    for fold, (train_index, val_index) in tqdm(enumerate(skf.split(X, y)), total=n_splits):
        # Initialization fold
        print('\n')
        print(f'############################ Fold n°{fold + 1} ############################ ')
        x_train, x_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        new_log_dir = log_dir + '_fold_' + str(fold)
        # Save best models per Fold
        if not os.path.exists(new_log_dir):
            os.makedirs(new_log_dir)
        best_model_path = os.path.join(new_log_dir, 'best_model.keras')
        # file_writer for each fold
        file_writer = tf.summary.create_file_writer(new_log_dir + "/metrics")
        file_writer_train = tf.summary.create_file_writer(new_log_dir + "/metrics_train")
        file_writer_val = tf.summary.create_file_writer(new_log_dir + "/metrics_val")

        # Callbacks
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=new_log_dir, histogram_freq=1)
        performance_train = PerformancePlotCallback(x_train, y_train, "CNN_model_train", file_writer_train, epoch_interval=epoch_interval, mode='train')
        performance_val = PerformancePlotCallback(x_val, y_val, "CNN_model_val", file_writer_val, epoch_interval=epoch_interval, mode='val')
        save_best_model = SaveBestModelCallback(x_val=x_val, y_val=y_val, file_path=best_model_path)
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)


        # Save parameters of the model
        yaml_file_path = os.path.join(new_log_dir, 'parameters.yaml')
        with open(yaml_file_path, 'w') as file:
            yaml.dump(params, file, default_flow_style=False)

        checkpoint_path = os.path.join(new_log_dir, '_fold_' + str(fold) + "_cnn_model.keras")
        checkpoint_callback = ModelCheckpoint(checkpoint_path, save_best_only=True)



        # Compute class weights as number of occurences of each class in the total observations 
        y_train = y_train.flatten() 
        y_val = y_val.flatten() 
        class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
        class_weights_dict = dict(enumerate(class_weights))
        
        model = build_cnn_model(image_shape=image_shape, learning_rate=learning_rate)
    

        history = model.fit(x_train, y_train, 
                            epochs=epochs, 
                            batch_size=batch_size, 
                            validation_data=(x_val, y_val),
                            callbacks=[early_stopping, checkpoint_callback, performance_train, performance_val, save_best_model, tensorboard_callback],
                            class_weight=class_weights_dict,  
                            verbose=1)
        

        print('\n')
        print('                PREDICTION                ')
        print('\n')
        y_pred_prob_train = model.predict(x_train)
        y_pred_prob_val = model.predict(x_val)
        print('\n')
        print('                             EVALUATION                ')
        print('\n')
        HTER_train, HTER_val, f1_train, f1_val, roc_auc_train, roc_auc_val = evaluate_model(y_pred_prob_train, y_train, y_pred_prob_val, y_val, threshold = 0.5)
        
        # Saving the metrics for Training and validation
        metrics_file = os.path.join(new_log_dir, f'performance_metrics_fold_{fold}.txt')
        with open(metrics_file, 'w') as f:
            f.write(f"Fold: {fold}\n")
            f.write(f"HTER Train: {HTER_train:.6f}\n")
            f.write(f"HTER Validation: {HTER_val:.6f}\n")
            f.write(f"F1 Score Train: {f1_train:.6f}\n")
            f.write(f"F1 Score Validation: {f1_val:.6f}\n")
            f.write(f"ROC AUC Train: {roc_auc_train:.6f}\n")
            f.write(f"ROC AUC Validation: {roc_auc_val:.6f}\n")

        # Save the experiment data fo each fold during Cross Validation
        experiment_data = {
                'arguments': args,
                'parameters': params,
                'performance': {
                    'HTER_train': f"{HTER_train:.6f}",
                    'HTER_val': f"{HTER_val:.6f}",
                    'f1_train': f"{f1_train:.6f}",
                    'f1_val': f"{f1_val:.6f}",
                    'roc_auc_train': f"{roc_auc_train:.6f}",
                    'roc_auc_val': f"{roc_auc_val:.6f}"
                },
            }

        combined_file_path = os.path.join(log_dir, 'experiment_tracking_{fold}.yaml')
        with open(combined_file_path, 'w') as file:
            yaml.dump(experiment_data, file, default_flow_style=False)
        print(f"Saved performance metrics for Fold {fold} in {metrics_file}\n")

        # Storing the metrics for Training and validation
        metrics_train['HTER'].append(HTER_train)
        metrics_train['f1_score'].append(f1_train)
        metrics_train['roc_auc'].append(roc_auc_train)

        metrics_evaluation['HTER'].append(HTER_val)
        metrics_evaluation['f1_score'].append(f1_val)
        metrics_evaluation['roc_auc'].append(roc_auc_val)


    average_hter_train = np.mean(metrics_train['HTER'])
    average_hter_val = np.mean(metrics_evaluation['HTER'])
    average_f1_score_train = np.mean(metrics_train['f1_score'])
    average_f1_score_val = np.mean(metrics_evaluation['f1_score'])
    average_roc_train = np.mean(metrics_train['roc_auc'])
    average_roc_val_list = np.mean(metrics_evaluation['roc_auc']) 


    # Save the experiment data for the average of the Cross Validation
    experiment_data = {
                'arguments': args,
                'parameters': params,
                'performance': {
                    'HTER_train': f"{average_hter_train:.6f}",
                    'HTER_val': f"{average_hter_val:.6f}",
                    'f1_train': f"{average_f1_score_train:.6f}",
                    'f1_val': f"{average_f1_score_val:.6f}",
                    'roc_auc_train': f"{average_roc_train:.6f}",
                    'roc_auc_val': f"{average_roc_val_list:.6f}"
                },
            }

    combined_file_path = os.path.join(log_dir, 'experiment_tracking_average.yaml')
    with open(combined_file_path, 'w') as file:
        yaml.dump(experiment_data, file, default_flow_style=False)
    print(f"Saved performance metrics average of Cross Val in {metrics_file}\n")


    # Display the average metrics
    print(f'Average f1 train: {average_f1_score_train:.6f}')
    print(f'Average f1 val: {average_f1_score_val:.6f}')
    print(f'Average ROC train: {average_roc_train:.6f}')
    print(f'Average ROC val: {average_roc_val_list:.6f}')
    print(f'Average HTER train: {average_hter_train:.4f}')
    print(f'Average HTER val: {average_hter_val:.4f}')
    
    return model, metrics_train, metrics_evaluation

