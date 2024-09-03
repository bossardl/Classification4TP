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

        return df.values
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
        np.ndarray: A NumPy array of indices of drawn images.
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
    

    return indices



def get_data(image_set: np.ndarray, labels: np.ndarray, indices: np.ndarray,):
    """
    Get X and y. Display information about the set.


    Parameters:
        image_set (np.ndarray): A NumPy array of images.
        label (np.ndarray): A NumPy array of labels.#
        n_draw (int): Number of samples to draw randomly from image set
        
    Returns:
        np.ndarray: A NumPy array of indices of drawn images.
    """
    image_shape = image_set[indices[0]].shape
    X = image_set
    y = np.float32(labels)
    print(f'Shape des données {image_shape}')
    return X, y 
