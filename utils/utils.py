import os


def exists_in_folder(expected_filename_part, target_dir):
    """
    This function checks if there is any file in a given directory that contains
    a specified string in its filename.

    Args:
        expected_filename_part (str): The substring to search for in the filenames.
        target_dir (str): The directory in which to search for the files.

    Returns:
        bool: True if a file containing the substring in its name is found, False otherwise.
    """
    for filename in os.listdir(target_dir):
        if expected_filename_part in filename:
            return True
    return False


def delete_files(filename_part, target_dir):
    """
    This function deletes files in a specified directory that contain a
    certain substring in their filename.

    Args:
        filename_part (str): The substring to match within the filenames.
        target_dir (str): The directory where to delete files from.
    """
    files_in_folder = os.listdir(target_dir)
    for file in files_in_folder:
        if filename_part in file:
            file_path = os.path.join(target_dir, file)
            os.remove(file_path)


def convert_to_https(url):
    """
    Convert url from http to https.
    """
    if url.startswith('http://'):
        return 'https://' + url[len('http://'):]
    return url
