import pandas as pd
import os
import shutil
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from keras.preprocessing import image
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Flatten, Dense
from sklearn.metrics import confusion_matrix
from helper_spectogram import GetSpectograms
from model import create_model
import sys

cur_dir = os.getcwd()
sys.path.append(cur_dir)

logs_path = "./Model/Logs"
checkpoint_path = "./Model/Training/model-{epoch:02d}-{val_accuracy:.4f}.keras"

Applause_path = "./Model/Data/Test/Segments/Applause"
Spectogram_Applause_path = "./Model/Data/Test/Spectogram/Applause"

Randomaudio_path = "./Model/Data/Test/Segments/RandomAudio"
Spectogram_Randomaudio_path = "./Model/Data/Test/Spectogram/RandomAudio"

num_classes = 2
batch_size = 10
epochs = 2

def create_model(num_classes):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
    model.add(MaxPooling2D(2, 2))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(2, 2))
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

if __name__ == "__main__":
    x = GetSpectograms(Spectogram_Applause_path)
    y = [0] * len(x)

    Len = len(x)
    x += GetSpectograms(Spectogram_Randomaudio_path)
    y += [1] * (len(x) - Len)

    x_test_norm = np.array(x) / 255

    y_test_encoded = to_categorical(y)

    model = create_model(num_classes)

    model.load_weights("./Model/Training/model-13-0.9687.keras")

    print("Testing Model")
    predictions = model.predict(x_test_norm)

    # Convert predicted probabilities to class labels
    predicted_labels = np.argmax(predictions, axis=1)

    # Convert one-hot encoded labels back to original labels
    true_labels = np.argmax(y_test_encoded, axis=1)

    # Calculate confusion matrix
    conf_matrix = confusion_matrix(true_labels, predicted_labels)

    # Calculate True Positives, True Negatives, False Positives, False Negatives
    TP = np.diag(conf_matrix)
    FP = np.sum(conf_matrix, axis=0) - TP
    FN = np.sum(conf_matrix, axis=1) - TP
    TN = np.sum(conf_matrix) - (TP + FP + FN)

    # Calculate Sensitivity (True Positive Rate)
    sensitivity = TP / (TP + FN)

    # Calculate Specificity (True Negative Rate)
    specificity = TN / (TN + FP)

    # Calculate Accuracy
    accuracy = (TP + TN) / np.sum(conf_matrix)

    # Plotting
    labels = ['Class 0', 'Class 1']
    x = np.arange(len(labels))  # the label locations
    width = 0.3  # the width of the bars

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Plot Metrics
    bar1 = ax1.bar(x - width, sensitivity, width, label='Sensitivity')
    bar2 = ax1.bar(x, specificity, width, label='Specificity')
    bar3 = ax1.bar(x + width, accuracy, width, label='Accuracy')

    ax1.set_xlabel('Metrics')
    ax1.set_ylabel('Score')
    ax1.set_title('Model Evaluation Metrics')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.legend()

    # Attach a text label above each bar in *rects*, displaying its height.
    def autolabel(rects, ax):
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(round(height, 2)),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(bar1, ax1)
    autolabel(bar2, ax1)
    autolabel(bar3, ax1)

    # Plot Confusion Matrix
    im = ax2.imshow(conf_matrix, cmap='Blues')

    ax2.set_xticks(np.arange(len(labels)))
    ax2.set_yticks(np.arange(len(labels)))
    ax2.set_xticklabels(labels)
    ax2.set_yticklabels(labels)
    ax2.set_xlabel('Predicted label')
    ax2.set_ylabel('True label')
    ax2.set_title('Confusion Matrix')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax2.get_xticklabels(), rotation=45, ha="right",
            rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax2.text(j, i, str(conf_matrix[i, j]), ha='center', va='center',
                    color='black')

    fig.tight_layout()
    plt.savefig("./Confusion_matrix.jpg")
    plt.show()