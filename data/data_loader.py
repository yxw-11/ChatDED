import pandas as pd
import pickle
from pathlib import Path
from typing import Any, List, Union
import os

def read_csv_file(filepath: str) -> pd.DataFrame:
    """
    Read a CSV file and return it as a DataFrame.
    :param filepath: Path to the CSV file
    :return: pandas DataFrame
    """
    try:
        # Read the CSV file into a DataFrame
        df = pd.read_csv(filepath)
        
        # Print some basic info
        print(f"File successfully read: {filepath}")
        print(f"Data shape: {df.shape}")  # (rows, columns)
        
        return df
    except Exception as e:
        # Catch errors if file reading fails
        print(f"Error reading file: {e}")
        return None

def save_list_to_pkl(data: List[Any], file_path: Union[str, Path]) -> None:
    """
    Save a Python list to a .pkl file.
    Creates parent folders if they don't exist.
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

def load_list_from_pkl(file_path: Union[str, Path]) -> List[Any]:
    """
    Load a list from a .pkl file.
    """
    with open(file_path, "rb") as f:
        return pickle.load(f)
    
def read_txt_file(file_path):
    """
    Reads the content of a txt file and returns it as a string.
    """
    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()
    return content


def read_txt_files(directory_path):
    """
    Reads all txt files in a given directory and combines their contents into one string.
    
    Parameters:
    - directory_path: Path to the directory containing txt files.
    
    Returns:
    - A single string containing the contents of all txt files in the directory.
    """
    combined_content = ""
    
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        
        if os.path.isfile(file_path) and filename.endswith('.txt'):
            with open(file_path, "r", encoding="utf-8") as file:
                combined_content += file.read() + "\n" 
    return combined_content

def save_list_to_txt(lst, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        for i, item in enumerate(lst, start=1):
            f.write(f"CASE {i}\n{item}")
            if i < len(lst):  
                f.write("\n\n")

# Example usage
if __name__ == "__main__":
    data = read_csv_file("data.csv")
    print(data.head())  # Show first 5 rows
    
    txt = read_txt_files("txt_files")
    print(len(txt))
