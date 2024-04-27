from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import datetime
import numpy as np
from helper_spectogram import GetSpectograms

num_classes = 2  

log_dir = "./Model/logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

Applause_path = "./Model/Data/Test/Segments/Applause"
Spectogram_Applause_path = "./Model/Data/Test/Spectogram/Applause"

Randomaudio_path = "./Model/Data/Test/Segments/RandomAudio"
Spectogram_Randomaudio_path = "./Model/Data/Test/Spectogram/RandomAudio"

if __name__ == "__main__":
    x = GetSpectograms(Spectogram_Applause_path)
    y = [0] * len(x)

    Len = len(x)
    x += GetSpectograms(Spectogram_Randomaudio_path)
    y += [1] * (len(x) - Len)

    x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size=0.3, random_state=0)

    x_train_norm = np.array(x_train) / 255
    x_test_norm = np.array(x_test) / 255

    y_train_encoded = to_categorical(y_train)
    y_test_encoded = to_categorical(y_test)

    model = load_model("./Model/Training/model-13-0.9687.keras")

    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(x_train_norm, y_train_encoded, validation_data=(x_test_norm, y_test_encoded), callbacks = [tensorboard_callback])
