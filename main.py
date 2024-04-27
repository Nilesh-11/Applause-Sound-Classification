import os
import numpy as np
import tkinter as tk
from tkinter import filedialog
import wave
# from pydub import AudioSegment
from pydub.utils import mediainfo
from matplotlib.widgets import Slider
from keras.preprocessing import image
import matplotlib.pyplot as plt

from Model.model import create_model
from Model.helper_spectogram import GenerateSpectograms
from Model.helper_audio import convert_to_wav
from Model.helper_audio import save_segments
from Model.helper_audio import split_audio_into_segments
from Model.helper_audio import remove_silence
from Model.helper_audio import record_audio
from Model.model import GetSpectograms
from Model.helper_folder import create_folder
from Model.helper_folder import remove_folder
from Model.helper_folder import extract_times_from_filename

Data_path = "./Data/"
segments_path = "./Data/Segments"
Spectogram_path = "./Data/Spectogram"
Model_path = "./Model/Training/model-13-0.9687.keras"
recorded_audio_path = "./Data/"
recorded_audio_filename = "recorded_audio.wav"
stamps = {}
duration = 0
record_duration = 0
choice = "file"
segment_length_sec = None
audio_duration_sec = None

def select_file():
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    file_path = filedialog.askopenfilename()  # Prompt the user to select a file
    return file_path

def binary_search(arr, x):
    low = 0
    high = len(arr) - 1
    while low <= high:
        mid = (low + high) // 2
        if arr[mid] == x:
            return arr[mid]
        elif arr[mid] < x:
            low = mid + 1
        else:
            high = mid - 1
    
    return arr[low] if low < len(arr) else None

def calculate_values(x, predictions, keys):
    result = binary_search(keys, x)
    if result == None :
        return 0
    return predictions[result][0][0]


def on_submit():
    global segment_length_sec, audio_duration_sec
    segment_length_sec = int(segment_length_entry.get())
    if choice == "audio":
        audio_duration_sec = int(audio_duration_entry.get())
    root.quit()
    print_results()
    quit_application()

def quit_application():
    root.quit()
    root.destroy()

def upload_file_gui():
    global choice
    choice = "file"
    segment_length_label.pack()
    segment_length_entry.pack()
    submit_button.pack()
    root.geometry("300x200")

def record_audio_gui():
    global choice
    choice = "audio"
    audio_label = tk.Label(root, text="To start recording audio, close this window.")
    audio_label.pack()
    segment_length_label.pack()
    segment_length_entry.pack()
    audio_duration_label.pack()
    audio_duration_entry.pack()
    submit_button.pack()
    root.geometry("400x250")

def print_results():
    print("Choice:", choice)
    print("Segment Length:", segment_length_sec)
    if choice == "audio":
        print("Audio Duration:", audio_duration_sec)

def StartKar():
    global segments_path, Spectogram_path
    create_folder(segments_path)
    create_folder(Spectogram_path)
    return

def Donewiththis():
    global segments_path, Spectogram_path
    remove_folder(segments_path)
    remove_folder(Spectogram_path)
    return

if __name__ == "__main__":
    StartKar()
    root = tk.Tk()
    root.title("Segment Length Input")
    root.geometry("300x200")

    upload_button = tk.Button(root, text="Upload File", command=upload_file_gui)
    upload_button.pack()

    record_button = tk.Button(root, text="Record Audio", command=record_audio_gui)
    record_button.pack()

    segment_length_label = tk.Label(root, text="Segment Length (sec):")
    segment_length_entry = tk.Entry(root)

    audio_duration_label = tk.Label(root, text="Audio Duration (sec):")
    audio_duration_entry = tk.Entry(root)

    submit_button = tk.Button(root, text="Submit", command=on_submit)

    root.protocol("WM_DELETE_WINDOW", quit_application)

    root.mainloop()

    # display_message_for_time("Please select audio file : ", 2000)
    if choice == "file": 
        selected_file = select_file() 
    else :
        record_audio(audio_duration_sec, recorded_audio_path, recorded_audio_filename)
        selected_file = os.path.join(recorded_audio_path, recorded_audio_filename)  

    print("Selected file:", selected_file)

    # Convert the file to WAV format (if possible) and print the result
    if selected_file:
        converted_audio = convert_to_wav(selected_file)
        if converted_audio:
            output_file_path = os.path.join(Data_path, os.path.basename(converted_audio))
            os.rename(converted_audio, output_file_path)
            test_file_path = os.path.join(Data_path, "silenced_audio.wav")
            remove_silence(output_file_path, test_file_path)
            output_file_path = test_file_path
            selected_file = output_file_path # when audio is in different extensions
            print("File converted successfully to WAV format.")
        else:
            print("File could not be converted.")
            exit()

    segments = split_audio_into_segments(output_file_path, segment_length_sec)
    save_segments(segments, segments_path)
    GenerateSpectograms(segments_path, Spectogram_path)

    num_classes = 2
    model = create_model(num_classes)
    model.load_weights(Model_path)

    # Iterate over each spectrogram file
    for file in os.listdir(Spectogram_path):
        # Load and preprocess the image data
        data = []
        data.append(image.img_to_array(image.load_img(os.path.join(Spectogram_path, file), target_size=(224, 224, 3))))
        data = np.array(data) 
        data = data / 255 

        # Make predictions using the model
        prediction = model.predict(data)

        # Extract start and end times from the file name
        start_time, end_time = extract_times_from_filename(file)
        stamps[end_time] = prediction

        duration = max(duration, end_time)

    predictions = dict(sorted(stamps.items()))
    keys = sorted(predictions.keys())
    print("\n", keys, "\n")
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)

    # Set initial value for the slider
    initial_value = 0

    # Create the slider
    ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')
    slider = Slider(ax_slider, 'Current Time(ms)', 0, duration, valinit=initial_value)

    # Function to update plot when slider value changes
    def update(val):
        x = slider.val
        y = calculate_values(x, predictions, keys)
        print("\n", y, "\n")
        ax.clear()
        ax.plot(x, y, 'ro')
        ax.set_xlabel('Time(ms)')
        ax.set_ylabel('Applause Probability')
        ax.set_title('Applause Sound Detector')
        ax.set_ylim(0, 1)
        fig.canvas.draw_idle()

    # Attach the update function to the slider
    slider.on_changed(update)

    # Show the plot
    plt.show()

    def plot_prediction_results(duration):
        x = np.arange(1, duration+1)  # Generate x values from 1 to duration inclusive
        y = [calculate_values(val, predictions=predictions, keys=keys) for val in x]   # Calculate y values using the function

        plt.figure()
        plt.plot(x, y)
        plt.ylim(0, 1)  # Set y-axis limits from 0 to 1
        plt.xlabel('Time(ms)')
        plt.ylabel('Applause Probability')
        plt.title('Applause Sound Detector')
        plt.grid(True)
        plt.savefig(f'./Results/prediction_results-len={duration}.png')  # Save the figure as prediction_results.png in the current directory
        plt.show()

    plot_prediction_results(duration)
    Donewiththis()
    pass