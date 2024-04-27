import sounddevice as sd
from pydub import AudioSegment

def record_audio(audio_duration_sec, recorded_audio_path, recorded_audio_filename, sample_rate=44100, channels=2):
    # Calculate the number of frames to record based on duration and sample rate
    num_frames = int(audio_duration_sec * sample_rate)

    # Start recording
    recording = sd.rec(frames=num_frames, samplerate=sample_rate, channels=channels, dtype='int16')
    sd.wait()  # Wait for recording to finish

    # Convert recorded audio to AudioSegment
    audio_segment = AudioSegment(
        recording.tobytes(),  # Convert NumPy array to bytes
        frame_rate=sample_rate,
        sample_width=recording.dtype.itemsize,  # Sample width in bytes
        channels=channels
    )

    # Save the recorded audio to the specified path and filename
    output_file = recorded_audio_path + "/" + recorded_audio_filename
    audio_segment.export(output_file, format="wav")

    return output_file

# Example usage: record audio for 5 seconds and save to specified path and filename
audio_duration_sec = 6
recorded_audio_path = "./Data"
recorded_audio_filename = "recorded_audio.wav"

output_file = record_audio(audio_duration_sec, recorded_audio_path, recorded_audio_filename)
print(f"Audio recorded and saved to {output_file}")
