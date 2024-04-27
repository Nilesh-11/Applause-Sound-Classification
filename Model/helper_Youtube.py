from pytube import YouTube
from moviepy.editor import AudioFileClip
from pydub import AudioSegment, silence
import os
from helper_audio import remove_silence

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


if __name__ == "__main__":
    # video_url = "https://www.youtube.com/watch?v=J43U4InntJg"  # Replace with your YouTube video URL
    # output_path = "./Model/Data/Test/"  # Replace with your desired output directory
    # download_audio(video_url, output_path)
    
    # mp4_file = "./Model/Data/Test/Applause Sound Effect  Crowd Applause  Clapping Sound Effect.mp4"  # Replace with your input MP4 file
    # output_wav = "./Model/Data/Test/output_audio.wav"  # Replace with your desired output WAV file
    # convert_mp4_to_wav(mp4_file, output_wav)

    audio_path = "./Model/Data/Test/output_audio.wav"  # Replace with your input audio file path
    output_path = "./Model/Data/Test/silenced_output_audio.wav"  # Replace with your desired output audio file path
    remove_silence(audio_path, output_path)

    # path = "./Data/Applause/Segments"
    # threshold = 32000
    #  # Get list of files in the directory
    # audio_files = [file for file in os.listdir(path) if file.endswith(".wav")]

    # # Process each audio file
    # for file in audio_files:
    #     audio_path = os.path.join(path, file)
    #     detect_silence(audio_path, threshold)