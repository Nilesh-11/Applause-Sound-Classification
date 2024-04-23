import tkinter as tk
from tkinter import filedialog
from pydub import AudioSegment
import numpy as np
import simpleaudio as sa

# Initialize variables
audio_path = None
audio = None
audio_position = 0

# Function to load and play the selected audio file
def play_audio():
    global audio_path, audio, audio_position
    audio_path = filedialog.askopenfilename()  # Prompt the user to select an audio file
    if audio_path:
        audio = AudioSegment.from_file(audio_path)
        audio_position = 0
        play_next_segment()

# Function to play the next segment of audio
def play_next_segment():
    global audio, audio_position
    if audio:
        audio_length = len(audio)
        if audio_position < audio_length:
            audio_segment = audio[audio_position:audio_position+1000]  # Extract 1 second audio segment
            intensity = np.mean(np.abs(audio_segment.get_array_of_samples()))  # Calculate intensity
            current_time = audio_position / 1000  # Convert milliseconds to seconds
            print(f"Time: {current_time}s, Intensity: {intensity:.2f} dB")
            samples = audio_segment.get_array_of_samples()
            sample_rate = audio_segment.frame_rate
            play_obj = sa.play_buffer(samples, num_channels=1, bytes_per_sample=2, sample_rate=sample_rate)
            play_obj.wait_done()
            audio_position += 1000
            # Schedule the next segment to be played
            window.after(1, play_next_segment)
        else:
            print("Audio playback completed.")

# Create the Tkinter window
window = tk.Tk()
window.title("Audio Player")

# Create a button to select and play the audio file
play_button = tk.Button(window, text="Select and Play Audio", command=play_audio)
play_button.pack()

# Run the Tkinter event loop
window.mainloop()
