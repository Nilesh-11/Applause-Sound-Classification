import pandas as pd
import os
import shutil
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from keras.preprocessing import image
import tensorflow as tf
from keras.layers import Flatten, Dense

checkpoint_path = "./Training/model-{epoch:02d}-{val_accuracy:.4f}.weights.h5"

Applause_path = "./Model/Data/Test/Segments/Applause"
Spectogram_Applause_path = "./Model/Data/Test/Spectogram/Applause"

Randomaudio_path = "./Model/Data/Test/Segments/RandomAudio"
Spectogram_Randomaudio_path = "./Model/Data/Test/Spectogram/RandomAudio"

def SortData(flush_path):
    print("This function needs to be run once...")
    fileslist = pd.read_csv("./Data/test_post_competition_scoring_clips.csv")
    files = []
    for index,row in fileslist.iterrows():
        if row["label"] != "Applause":
            files.append(row["fname"])

    def copy_files(src, dest, filename):
        if not os.path.exists(src):
            print("Invalid source folder path.")
            return
        
        if not os.path.exists(dest):
            print("Creating destination folder path...")
            os.makedirs(dest)
        
        for file in os.listdir(src):
            if file == filename:
                src_file_path = os.path.join(src, file)
                dest_file_path = os.path.join(dest, file)
                if os.path.exists(dest_file_path):
                    print(f"Overwriting File {filename}..")
                shutil.copyfile(src_file_path, dest_file_path)
                print(f"File '{filename}' copied successfully.")

    src_folder = "./Data/FSDKaggle2018.audio_test"
    dest_folder = flush_path
    for file in files:
        copy_files(src_folder, dest_folder, file)
    print("Wavfile sorted successfully...\n")
    return

def create_spectrogram(audio_file, image_file, output_folder):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    y, sr = librosa.load(audio_file)
    ms = librosa.stft(y)
    log_ms = librosa.amplitude_to_db(ms, ref=np.max)
    librosa.display.specshow(log_ms, sr=sr)
    if os.path.exists(image_file):
        print(f"Overwriting {image_file}...")
    plt.savefig(os.path.join(image_file))
    plt.close()

def GenerateSpectograms(source_folder, destination_folder):
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    print("Generating Spectograms...")
    for file in os.listdir(source_folder):
        src_image_path = os.path.join(source_folder, file)
        dest_image_path = os.path.join(destination_folder, file.replace('.wav', '.png'))
        create_spectrogram(src_image_path, dest_image_path, destination_folder)
        # shutil.copyfile(src_image_path, dest_image_path)
        print(f"File '{file}' generated spectogram successfully...")
    print("Completed generating spectrograms...\n")
    return

def preprocessingSpectograms(path):
    images = []
    for file in os.listdir(path):
        images.append(image.img_to_array(image.load_img(os.path.join(path, file), target_size=(224, 224, 3))))
    return images

def show_images(images):
    fig, axes = plt.subplots(1, 10, figsize=(20, 20), subplot_kw={'xticks': [], 'yticks': []})

    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i] / 255)
    plt.show()
    plt.close(fig)

def GetSpectograms(path):
    images = []
    for file in os.listdir(path):
        images.append(image.img_to_array(image.load_img(os.path.join(path, file), target_size=(224, 224, 3))))
        if len(images) % 1000 == 0:
            print(f"Images len :", len(images)," : ", path)
        if len(images) == 3000:
            return images
    return images

if __name__ == "__main__":
    GenerateSpectograms(Applause_path, Spectogram_Applause_path)
    GenerateSpectograms(Randomaudio_path, Spectogram_Randomaudio_path)
    pass