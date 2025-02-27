import pickle


def text_file_to_string(file_path):
    """Reads the contents of a text file into a string.

    Args:
        file_path (Str): The file location of the text file.

    Returns:
        (Str): The contents of the text file.
    """
    with open(file_path, "r") as f:
        return f.read()


def save_object(object, pickle_path):
    """Saves an object to a pickle file.

    Args:
        object (Any): The object to be saved.
        pickle_path (Str): The file location to save the object.
    """

    with open(pickle_path, "wb") as f:
        pickle.dump(object, f)


def load_object(pickle_path):
    """Loads an object from a pickle file.

    Args:
        pickle_path (Str): The file location of the saved object.

    Returns:
        (Any): The loaded object.
    """
    with open(pickle_path, "rb") as f:
        return pickle.load(f)
