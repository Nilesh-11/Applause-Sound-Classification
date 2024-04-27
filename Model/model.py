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
from helper_spectogram import GetSpectograms
import sys

cur_dir = os.getcwd()
sys.path.append(cur_dir)

logs_path = "./Model/logs/train/"
checkpoint_path = "./Model/Training/model-{epoch:02d}-{val_accuracy:.4f}.keras"

Spectogram_Applause_path = "./Model/Spectogram/Applause"

Spectogram_Randomaudio_path = "./Model/Spectogram/Randomaudio1"

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

    print("Splitting train and testing data...")

    x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size=0.3, random_state=0)

    x_train_norm = np.array(x_train) / 255
    x_test_norm = np.array(x_test) / 255

    y_train_encoded = to_categorical(y_train)
    y_test_encoded = to_categorical(y_test)

    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                    monitor='val_accuracy',
                                                    save_freq='epoch',
                                                    verbose=1)

    model = create_model(num_classes)

    print("Training Model")
    hist = model.fit(x_train_norm, 
                    y_train_encoded, 
                    validation_data=(x_test_norm, y_test_encoded), 
                    batch_size=batch_size,
                    epochs=epochs,
                    callbacks = [cp_callback]
                    )

    history_df = pd.DataFrame(hist.history)
    history_df.to_csv(os.path.join(logs_path, "training_history.csv"), index=False)

    acc = hist.history['accuracy']
    val_acc = hist.history['val_accuracy']
    epochs = range(1, len(acc) + 1)

    plt.plot(epochs, acc, '-', label='Training Accuracy')
    plt.plot(epochs, val_acc, ':', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.plot()
    plt.show()
    pass