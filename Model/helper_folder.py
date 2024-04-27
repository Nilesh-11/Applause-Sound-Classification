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
        

def extract_times_from_filename(filename):
    try:
        # Split the filename using '_' as delimiter
        filename = filename.rstrip('.png')
        parts = filename.split('_')
        # Extract start and end times from the filename
        start_time = int(parts[-3].split('_')[-1].replace('ms', ''))
        end_time = int(parts[-1].split('_')[-1].replace('ms', ''))
        return start_time, end_time
    except ValueError:
        print(f"Error extracting times from filename: {filename}")
        return None, None
    
remove_folder("./Data/Segments")
remove_folder("./Data/Spectogram")