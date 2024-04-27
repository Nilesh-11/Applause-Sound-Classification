# 2D CNN Model which classifies applause sound from the given audio file

## How to navigate? WILL UDPATE SOON :)

### Steps to Train Model

## Generate audio Segments and spectograms:
Assuming you have audio file
1. **selected_file** is the directory to the audio file
3. **segments_path** is the directory to save segments
2. **Spectogram_path** is the directory to save spectograms
4. **segment_length_sec** is the length of segments
Then the run the following code in *./Model/helper_audio.py*
```python
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
```

## Training
1. **Spectogram_Applause_path** is the directory contains all the spectograms of applause class
2. **Spectogram_random_path** is the directory contains all the spectograms of non-applause class
3. **log_dir** is the directory to save your model
4. you can decide **batch_size** and **epochs**
After selecting proper variables run *./Model/model.py* file to start training. Model will be saved in *./Model/Training/* and the statistics will be saved in *./Model/logs/train/*

### Testing the Model
## Check Applause from audio file
1. select model from *./Model/Training* and update **Model_path**
run *./main.py* follow the gui. Results will be saved in *./Results*