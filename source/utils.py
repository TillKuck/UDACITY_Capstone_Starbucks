import pickle

def save_data(data, filepath):
    """
    Writes data to a pickle file to a specified location
    --------------
    Parameters:
    - data: data to be saved
    - filepath: The directory path where the pickle file will be saved
    """
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)

def load_data(file_path):
    """
    Loads data from a pickle file from a specified location
    --------------
    Parameters:
    - filepath: The directory path where the pickle file will be saved
    --------------
    Returns data
    """
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data
