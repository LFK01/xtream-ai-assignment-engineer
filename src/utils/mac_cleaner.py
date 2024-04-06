import os
from src.utils.consts import ROOT_DIR

if __name__ == '__main__':
    # Delete all the .DS_Store file found in the project directory
    # traverse root directory, and list directories as dirs and files as files
    for root, dirs, files in os.walk(ROOT_DIR):
        for file in files:
            print(f'root: {root} dirs: {dirs} file: {file}')
            if file == '.DS_Store':
                os.remove(os.path.join(root, file))
