from pytube import YouTube
from moviepy.editor import AudioFileClip
from pydub import AudioSegment, silence
import os

def download_audio(video_url, output_path):
    try:
        # Create a YouTube object
        yt = YouTube(video_url)

        # Get the highest quality audio stream
        audio_stream = yt.streams.filter(only_audio=True).first()

        # Download the audio
        audio_stream.download(output_path=output_path)
        print("Audio downloaded successfully.")
    except Exception as e:
        print(f"Error downloading audio: {str(e)}")

def convert_mp4_to_wav(mp4_file, output_wav):
  try:
    audio_clip = AudioFileClip(mp4_file)
    audio_clip.write_audiofile(output_wav)
    print("Conversion completed successfully.")
  except Exception as e:
    print(f"Error converting MP4 to WAV: {str(e)}")

def remove_silence(audio_path, output_path, silence_thresh=-40, min_silence_len=100):
    try:
        # Load the audio
        audio = AudioSegment.from_file(audio_path)

        # Detect and remove silence
        non_silent_audio = silence.detect_silence(audio, silence_thresh=silence_thresh, min_silence_len=min_silence_len)
        
        # Construct a new audio segment without silence
        result = AudioSegment.empty()
        for start, end in non_silent_audio:
            result += audio[start:end]

        # Export the result
        result.export(output_path, format="wav")
        print("Silence removed successfully.")
    except Exception as e:
        print(f"Error removing silence: {str(e)}")

def detect_silence(audio_path, threshold):
    try:
        # Load the audio
        audio = AudioSegment.from_file(audio_path)

        # Get the raw data
        raw_data = audio.raw_data

        # Convert raw data to list of amplitudes
        amplitudes = list(audio.get_array_of_samples())

        # Determine silence segments
        silence_segments = []
        is_silence = False
        silence_start = None
        for i, amp in enumerate(amplitudes):
            if abs(amp) < threshold:
                if not is_silence:
                    silence_start = i
                    is_silence = True
            else:
                if is_silence:
                    silence_segments.append((silence_start, i))
                    is_silence = False

        # Print silence segments and file name
        if silence_segments:
            print(f"Silence detected in file: {audio_path}")
            # for start, end in silence_segments:
            #     print(f"Silence from {start} ms to {end} ms")
            return 1
        else:
            # print(f"No silence detected in file: {audio_path}")
            return 0

    except Exception as e:
        print(f"Error detecting silence: {str(e)}")

if __name__ == "__main__":
    video_url = "https://www.youtube.com/watch?v=OHlDtOIZ1tE"  # Replace with your YouTube video URL
    output_path = "./Model/Data/Test/"  # Replace with your desired output directory
    download_audio(video_url, output_path)
    
    mp4_file = "./Model/Data/Dr Shashi Tharoor MP - Britain Does Owe Reparations.mp4"  # Replace with your input MP4 file
    output_wav = "./Model/Data/Test/output_audio.wav"  # Replace with your desired output WAV file
    convert_mp4_to_wav(mp4_file, output_wav)

    # audio_path = "./Data/Applause.wav"  # Replace with your input audio file path
    # output_path = "./Data/output_audio.wav"  # Replace with your desired output audio file path
    # remove_silence(audio_path, output_path)
    
    # path = "./Data/Applause/Segments"
    # threshold = 32000
    #  # Get list of files in the directory
    # audio_files = [file for file in os.listdir(path) if file.endswith(".wav")]

    # # Process each audio file
    # for file in audio_files:
    #     audio_path = os.path.join(path, file)
    #     detect_silence(audio_path, threshold)