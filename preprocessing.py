import numpy as np
import matplotlib.pyplot as plt
import math
import time
from PIL import Image
import cv2
import os, glob, sys
from tqdm import tqdm
from pathlib import Path
import pandas as pd


def read_label(label_file: Path) -> pd.DataFrame:
    """
    Reads a CSV file containing labels into a DataFrame.
    
    Parameters:
        label_file (Path): Path to the CSV file containing the labels.
        
    Returns:
        pd.DataFrame: A DataFrame with a single column 'label' containing the data from the CSV.
    """
    try:
        df = pd.read_csv(label_file, sep=',', header=None)
        df.columns = ['label']
        print('\n')
        print('\n')
        print('############################ PREPROCESSING ############################ ')
        print('Label du fichier label_train.txt')
        print(df.head())
        print('\n')
        print(f'Nombre de valeurs dans le df len({len(df)})- verification pas d\'outlier')
        print(df.nunique())
        print('\n')
        print(f'Nombre d\'occurences de chaque label')
        print(df.value_counts())
        print('\n')

        y = df.values
        return np.float32(y)
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return None
    

def load_and_normalize_images(image_dir: Path):
    """
    Load and normalize images from directory.

    This function iterates through a directory of JPG images, loads each image,
    normalizes the pixel values to the range [0, 1], and returns them as a NumPy array.

    Parameters:
        image_dir (Path): Path to the image folder containing the jpg images.
        
    Returns:
        np.ndarray: A NumPy array of normalized images.
    """
    images = []

    for filename in tqdm(os.listdir(image_dir)):
        if filename.endswith(".jpg"):
            img_path = os.path.join(image_dir, filename)
            img = Image.open(img_path)  # Load the image using PIL because faster than with plt
            img_array = np.array(img) 
            img_array = img_array.astype('float32') / 255.0  # Normalize the pixel values to [0, 1]

            images.append(img_array)
            
    images = np.array(images)
    
    return images




def get_sample(image_set: np.ndarray, labels: np.ndarray, n_draw: int, flag_display=False):
    
    """
    Display random samples with labels in a grid.

    Parameters:
        image_set (np.ndarray): A NumPy array of images.
        labels (np.ndarray): A NumPy array of labels.
        n_draw (int): Number of samples to draw randomly from image set
        
    Returns:
        
    """
    # Select n_draw random images and corresponding labels from the set
    indices = np.random.choice(len(image_set), size=n_draw, replace=False)
    sample_images = image_set[indices]
    sample_labels = labels[indices]
    
    
    if flag_display == True:
        # Grid parameters
        cols = int(math.ceil(math.sqrt(n_draw)))
        rows = int(math.ceil(n_draw / cols))
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.5))
        
        for i, ax in enumerate(axes.flatten()):
            if i < n_draw:
                ax.imshow(sample_images[i])
                ax.axis('off')
                ax.set_title(f"Image n°{indices[i]} - Label: {sample_labels[i]}")
            else:
                ax.axis('off')  

        plt.tight_layout()
        plt.show()
        
        # Wait for 2 seconds
        time.sleep(2)
        
        # Close the figure
        plt.close(fig)




def train_test_split_and_resampling(args, 
                                    X: np.array, 
                                    y: np.array,
                                    ):
    """
    Train test split and apply resampling method on the train set only to avoid data snooping.

    Parameters:
        args : dict
            Additional keyword arguments for model configuration and training, 
            including callbacks, model architecture, and hyperparameters.
        X (np.ndarray): A NumPy array of images.
        labels (np.ndarray): A NumPy array of labels.

        
    Returns:
    x_train (np.ndarray): Train images after splitting with the chosen resampling strategy
    x_val (np.ndarray): Val images after splitting with the chosen resampling strategy
    y_train (np.ndarray): Train labels after splitting with the chosen resampling strategy
    y_val (np.ndarray): Val labels after splitting with the chosen resampling strategy
        
    """

    from sklearn.model_selection import train_test_split

    if args.method == 'None':
        test_size=0.2
        x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=1, stratify=y)
        print(f'(X,y) split {test_size}. x_train, y_train unchanged. No resampling method selected. ')
        pass

    elif args.method=='undersampling':
        # uniform sub-sampling of majority class for balancing the train set 
        # Will increase the variance but reduce the bias
        # This effect must be mitigated during training
        from imblearn.under_sampling import RandomUnderSampler

        test_size=0.1
        x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=1, stratify=y)
        print('\n')
        print(f'Original information on data: \n ')
        print(f'Shape of the train {x_train.shape} and {x_val.shape} val image set.')
        print(f'Shape of the train {y_train.shape} and {y_val.shape} val label set.')
        # Count occurences of labels in train
        y_train_count, y_val_count = np.where(y_train==1)[0], np.where(y_val==1)[0]
        print(f'Number of 0s in train {len(y_train_count)} and in val {len(y_val_count)}.')
        print(f'Shape of data {x_train[0].shape}.')
        print('\n')
        
        print("Undersampling of the majority class to rebalance the set")
        print(f'(X,y) split {test_size}.')
        print('\n')

        undersampler = RandomUnderSampler(sampling_strategy=args.undersampling, random_state=42)
        X_flattened = x_train.reshape(x_train.shape[0], -1) 
        X_resampled, y_resampled = undersampler.fit_resample(X_flattened, y_train)
        #Shuffle the dataset
        X_resampled = X_resampled.reshape(-1, 64, 64, 3)
        order = np.arange(len(y_resampled))
        np.random.shuffle(order)
        x_train = X_resampled[order].copy()
        del X_resampled
        y_train = y_resampled[order].copy()
        del y_resampled

        print('\n')
        print(f'Updated information on data after split & resampling: \n ')
        print(f'Shape of the train {x_train.shape} and {x_val.shape} val image set.')
        print(f'Shape of the train {y_train.shape} and {y_val.shape} val label set.')
        # Count occurences of labels in train
        y_train_count, y_val_count = np.where(y_train==1)[0], np.where(y_val==1)[0]
        print(f'Number of 0s in train {len(y_train_count)} and in val {len(y_val_count)}.')
        print(f'Shape of data {x_train[0].shape}.')
        print('\n')


    elif args.method=='oversampling':
        # uniform sampling in the train set for augmentation
        # Will increase the bias but reduce the variance
        # This effect must be mitigated during training
        test_size=0.3
        x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=1, stratify=y)
        print('\n')
        print(f'Original information on data: \n ')
        print(f'Shape of the train {x_train.shape} and {x_val.shape} val image set.')
        print(f'Shape of the train {y_train.shape} and {y_val.shape} val label set.')
        # Count occurences of labels in train
        y_train_count, y_val_count = np.where(y_train==0)[0], np.where(y_val==0)[0]
        print(f'Number of 0s in train {len(y_train_count)} and in val {len(y_val_count)}.')
        print(f'Shape of data {x_train[0].shape}.')
        print('\n')
        
        print("Oversampling the minority class to rebalance the set after splitting")
        print(f'(X,y) split {test_size}.')
        print('\n')
        # Select each class
        pos_labels = np.where(y_train == 0)[0]
        neg_labels = np.where(y_train == 1)[0]

        pos_features = x_train[pos_labels]
        neg_features = x_train[neg_labels]

        ids = np.arange(len(pos_features))
        # Ensure replace=True for oversampling
        choices = np.random.choice(ids, int(len(neg_features)*args.oversampling), replace=True)  

        res_pos_features = pos_features[choices]
        # Since pos_labels are all 0, no need to sample
        res_pos_labels = np.zeros(len(res_pos_features))  
        #Concatenate the resampled image and labels in X_resampled. y_resampled
        X_resampled = np.concatenate([res_pos_features, neg_features], axis=0)
        y_resampled = np.concatenate([res_pos_labels, np.ones(len(neg_features))], axis=0)
        
        
        #Shuffle the dataset
        order = np.arange(len(y_resampled))
        np.random.shuffle(order)
        x_train = X_resampled[order].copy()
        del X_resampled
        y_train = y_resampled[order].copy()
        del y_resampled

        print('\n')
        print(f'Updated information on data after split & resampling: \n ')
        print(f'Shape of the train {x_train.shape} and {x_val.shape} val image set.')
        print(f'Shape of the train {y_train.shape} and {y_val.shape} val label set.')
        # Count occurences of labels in train
        y_train_count, y_val_count = np.where(y_train==0)[0], np.where(y_val==0)[0]
        print(f'Number of 0s in train {len(y_train_count)} and in val {len(y_val_count)}.')
        print(f'Shape of data {x_train[0].shape}.')
        print('\n')

    else:
        print("Unknown method selected. Please choose 'undersampling', 'oversampling' or 'None'.")
        exit
    
    return x_train, x_val, y_train, y_val
