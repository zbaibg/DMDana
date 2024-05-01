#!python
import glob
import logging
import os
import shutil


def setup_logging():
    """Set up the logging configuration."""
    logging.basicConfig(filename='copy_files.log', level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',filemode='w')

def read_config(config_file):
    """Read configuration and organize into tasks."""
    with open(config_file, 'r') as file:
        lines = file.readlines()

    tasks = []
    source = ''
    target = ''
    files = []

    for line in lines:
        clean_line = line.strip()
        if not clean_line:
            if source and target and files:
                tasks.append((source, target, files))
            source, target = '', ''
            files = []
        elif not source or not target:
            source, target = clean_line.split()
        else:
            files.append(clean_line)

    # Add the last task if not empty
    if source and target and files:
        tasks.append((source, target, files))

    return tasks

def copy_files(tasks):
    """Copy files based on the given list of tasks using wildcard support."""
    for source, target, file_patterns in tasks:
        if os.path.exists(target):
            if os.path.isdir(target):
                shutil.rmtree(target)
                logging.warning(f"{target} folder already exists, it is deleted and created again")
            else:
                logging.error(f"{target} existed but is not a folder, try manually delete it")
                exit(1)
        logging.info(f"Begin copying files from {source} to {target}")
        if not os.path.exists(target):
            os.makedirs(target)
        for pattern in file_patterns:
            # Use glob to find all files matching the pattern in the source directory
            matching_files = glob.glob(os.path.join(source, pattern))
            if not matching_files:
                logging.warning(f"No files matching {pattern} in {source}")
            for file_path in matching_files:
                target_file=os.path.join(target, os.path.basename(file_path))
                if os.path.islink(file_path):
                    linkto = os.readlink(file_path)
                    os.symlink(linkto, target_file)
                    logging.info(f"Link: {file_path.replace(source,'').strip('/')} is a link '{linkto}', successfully created ")
                    continue
                if os.path.isfile(file_path):
                    shutil.copy(file_path, target_file)
                    logging.info(f"File: {file_path.replace(source,'').strip('/')} file successfully copied ")
                    continue
                if os.path.isdir(file_path):
                    logging.info(f"Folder: {file_path.replace(source,'').strip('/')} is a folder, successfully coped with its content")
                    shutil.copytree(file_path, target_file)
                    continue

        logging.info("This folder finished\n")        

def main():
    setup_logging()
    config_file = 'config.list'
    tasks = read_config(config_file)
    copy_files(tasks)

if __name__ == "__main__":
    main()
