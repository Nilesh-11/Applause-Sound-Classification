import os
import shutil

def create_folder(folder_path):
    try:
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            print(f"Folder created at {folder_path}")
        else:
            print(f"Folder already exists at {folder_path}")
    except OSError as e:
        print(f"Error creating folder at {folder_path}: {e}")

def remove_folder(folder_path):
    try:
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)
            print(f"Folder and its contents removed at {folder_path}")
        else:
            print(f"Folder does not exist at {folder_path}")
    except OSError as e:
        print(f"Error removing folder at {folder_path}: {e}")
        
        
remove_folder("./Data/Segments")
remove_folder("./Data/Spectogram")