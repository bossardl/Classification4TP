import numpy as np
import tensorflow as tf
from tensorflow.keras import callbacks
import matplotlib.pyplot as plt
import io
from sklearn.metrics import confusion_matrix, f1_score

class SaveBestModelCallback(tf.keras.callbacks.Callback):
    def __init__(self, x_val, y_val, file_path):
        super(SaveBestModelCallback, self).__init__()
        self.x_val = x_val
        self.y_val = y_val
        self.file_path = file_path
        self.best_hter = float('inf') 

    def calculate_hter(self):
        threshold = 0.5
        y_pred_prob_val = self.model.predict(self.x_val)
        y_pred_val = (y_pred_prob_val > threshold).astype(int)
        cm = confusion_matrix(self.y_val, y_pred_val)

        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
        else:
            raise ValueError("Confusion matrix must be 2x2 for binary classification.")

        FAR = fp / (fp + tn) if (fp + tn) > 0 else 0
        FRR = fn / (fn + tp) if (fn + tp) > 0 else 0
        HTER = (FAR + FRR) / 2
        return HTER

    def on_epoch_end(self, epoch, logs=None):
        hter = self.calculate_hter()
        if hter < self.best_hter:
            self.best_hter = hter
            self.model.save(self.file_path)
            print(f"\nEpoch {epoch + 1}: HTER improved to {hter:.4f}. Model saved to {self.file_path}.")
        else:
            print(f"\nEpoch {epoch + 1}: HTER {hter:.4f}. Best HTER is {self.best_hter:.4f}.")


class PerformancePlotCallback(callbacks.Callback):
    def __init__(self, x_val, y_val, model_name, file_writer, epoch_interval=5, mode='train'):
        super(PerformancePlotCallback, self).__init__()
        self.x_val = x_val
        self.y_val = y_val
        self.model_name = model_name
        self.file_writer = file_writer
        self.mode = mode  # 'train' or 'val'
        self.epoch_interval = epoch_interval

    @staticmethod
    def plot_to_image(figure):
        """Converts the matplotlib plot specified by 'figure' to a PNG image and
        returns it. The supplied figure is closed and inaccessible after this call."""
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close(figure)
        buf.seek(0)
        image = tf.image.decode_png(buf.getvalue(), channels=4)
        image = tf.expand_dims(image, 0)
        return image

    def log_predictions_to_tensorboard(self, label, step):
        indices = np.random.choice(len(self.x_val), size=6, replace=False)
        sample_images = self.x_val[indices]
        sample_labels = self.y_val[indices]
        y_pred_prob_val = self.model.predict(sample_images)
        predictions = (y_pred_prob_val > 0.5).astype(int)

        fig, axes = plt.subplots(2, 3, figsize=(8, 6))
        for i, ax in enumerate(axes.flatten()):
            ax.imshow(sample_images[i])
            ax.axis('off')
            ax.set_title(f"True: {sample_labels[i]} - Pred: {predictions[i]}")
        
        with self.file_writer.as_default():
            tf.summary.image(f'Predictions {label}', self.plot_to_image(fig), step=step)

    def calculate_metrics(self, step):
        threshold = 0.5
        y_pred_prob_val = self.model.predict(self.x_val)
        y_pred_val = (y_pred_prob_val > threshold).astype(int)
        cm = confusion_matrix(self.y_val, y_pred_val)

        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
        else:
            raise ValueError("Confusion matrix must be 2x2 for binary classification.")

        FAR = fp / (fp + tn) if (fp + tn) > 0 else 0
        FRR = fn / (fn + tp) if (fn + tp) > 0 else 0
        HTER = (FAR + FRR) / 2
        f1_score_val = f1_score(self.y_val, y_pred_val)

        print(f'\nEpoch {step} - HTER {self.mode.capitalize()}: {HTER:.4f}')
        print(f'Epoch {step} - f1-score {self.mode.capitalize()}: {f1_score_val:.4f}')

        with self.file_writer.as_default():
            tf.summary.scalar(f'HTER {self.mode.capitalize()}', HTER, step=step)
            tf.summary.scalar(f'f1-score {self.mode.capitalize()}', f1_score_val, step=step)

    def on_epoch_end(self, epoch, logs=None):
        if self.mode == 'train' and (epoch + 1) % self.epoch_interval == 0:
            self.log_predictions_to_tensorboard(f'{self.model_name} Epoch {epoch+1}', step=epoch)
            self.calculate_metrics(step=epoch)

    def on_test_end(self, logs=None):
        if self.mode == 'val':
            self.log_predictions_to_tensorboard(f'{self.model_name} at Validation', step=0)
            self.calculate_metrics(step=0)
