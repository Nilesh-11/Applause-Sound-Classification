import tkinter as tk
from tkinter import filedialog

def select_files():
    file_paths = filedialog.askopenfilenames()
    print("Selected files:")
    for file_path in file_paths:
        print(file_path)

# Create the main window
root = tk.Tk()
root.title("File Selection")

# Create a button to trigger file selection
select_button = tk.Button(root, text="Select Files", command=select_files)
select_button.pack(pady=20)

# Run the Tkinter event loop
root.mainloop()
