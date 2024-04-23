from pydub import AudioSegment
from pydub.utils import mediainfo
import numpy as np
import soundfile as sf
from pydub.silence import split_on_silence

def convert_to_wav(input_file):
    try:
        # Load the audio file
        audio = AudioSegment.from_file(input_file)

        # Check the format of the input file
        input_format = mediainfo(input_file)['format_name'].lower()

        # If the input file is already in WAV format, just return it
        if input_format == 'wav':
            return input_file

        # If the input file is not in WAV format, convert it to WAV
        output_file = input_file.split('.')[0] + '.wav'
        audio.export(output_file, format='wav')
        return output_file
    except Exception as e:
        print("Invalid file:", e)
        return None

def split_audio_into_segments(audio_path, segment_length_sec):
    try:
        # Load the audio
        audio = AudioSegment.from_file(audio_path)

        # Split into segments
        segment_length_ms = segment_length_sec * 1000
        segments = []
        for i in range(0, len(audio), segment_length_ms):
            start_time = i
            end_time = min(i + segment_length_ms, len(audio))  # Ensure end time doesn't exceed audio length
            segment = audio[start_time:end_time]
            segments.append((segment, start_time, end_time))  # Append tuple containing segment, start time, and end time

        return segments
    except Exception as e:
        print(f"Error splitting audio: {str(e)}")
        return None

def save_segments(segments, output_path):
    try:
        for i, (segment, start_time, end_time) in enumerate(segments):
            segment.export(f"{output_path}/segment_{i+1}_start_{start_time}ms_end_{end_time}ms.wav", format="wav")
        print("Segments saved successfully.")
    except Exception as e:
        print(f"Error saving segments: {str(e)}")

import matplotlib.pyplot as plt 
import numpy as np 
import wave, sys 
  
# shows the sound waves 
def visualize(path: str): 
    
    # reading the audio file 
    raw = wave.open(path) 
      
    # reads all the frames  
    # -1 indicates all or max frames 
    signal = raw.readframes(-1) 
    signal = np.frombuffer(signal, dtype ="int16") 
      
    # gets the frame rate 
    f_rate = raw.getframerate() 

    time = np.linspace( 
        0, # start 
        len(signal) / f_rate, 
        num = len(signal) 
    ) 
  
    # using matplotlib to plot 
    # creates a new figure 
    plt.figure(1) 
      
    # title of the plot 
    time_ms = time * 1000  # Convert time from seconds to milliseconds
    plt.plot(time_ms, signal, color='blue') 
    plt.xlabel("Time (ms)")  # Update x-axis label to Time (ms)
    plt.ylabel("Amplitude") 
    plt.title("Audio Waveform") 
    plt.grid(True)
    plt.show()
  
    # you can also save 
    # the plot using 
    # plt.savefig('filename') 

def remove_silence(input_file, output_file, min_silence_len=500, silence_thresh=-40):
    print("Loading the audio file...")
    audio = AudioSegment.from_mp3(input_file)

    print("Splitting the audio based on silence...")
    chunks = split_on_silence(audio, min_silence_len=min_silence_len, silence_thresh=silence_thresh)

    print("Concatenating non-silent chunks...")
    output = AudioSegment.empty()
    for chunk in chunks:
        output += chunk
        
    print("Exporting the audio without silent parts...")
    output.export(output_file, format="mp3")
    
    print("---Done!---")

# Example usage:
input_file = "./Data/shashi.wav"
output_file = "./silenced.wav"

if __name__ == "__main__": 
    # path = "./Data/shashi.wav" 
    
    visualize(input_file)
    remove_silence(input_file, output_file)  
    visualize(output_file)
  