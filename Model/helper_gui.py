import tkinter as tk

choice = None
segment_length_sec = None
audio_duration_sec = None

def on_submit():
    global segment_length_sec, audio_duration_sec
    segment_length_sec = int(segment_length_entry.get())
    if choice == "audio":
        audio_duration_sec = int(audio_duration_entry.get())
    root.quit()
    print_results()

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
