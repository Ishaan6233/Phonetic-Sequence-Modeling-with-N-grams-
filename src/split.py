'''
CMPUT 461 - Assignment 3: Prepare the training.txt and dev.txt files from raw transformed data files.
Author: Ishaan Meena
'''

import glob
import os
import random


# Consolidate transformed files into a single file, ensuring line breaks between sequences
def consolidate_transformed_files(transformed_dir, consolidated_output):
    if not os.path.exists(transformed_dir):
        print(f"Error: The directory {transformed_dir} does not exist.")
        return

    if os.path.exists(consolidated_output) and os.path.getsize(consolidated_output) > 0:
        print(f"{consolidated_output} already exists and contains data. Skipping consolidation.")
        return

    with open(consolidated_output, 'w', encoding='utf-8') as outfile:
        for file_path in glob.glob(f"{transformed_dir}/**/*.txt", recursive=True):
            with open(file_path, 'r', encoding='utf-8') as infile:
                for line in infile:
                    outfile.write(line.strip() + '\n')  # Ensures line breaks between sequences
    print(f"Consolidation complete. Data saved to {consolidated_output}")


# Split consolidated data into training and dev files, ensuring balanced data
def split_data(input_file, train_file="training.txt", dev_file="dev.txt", split_ratio=0.8):
    if not os.path.exists(input_file) or os.path.getsize(input_file) == 0:
        print(f"Error: The file {input_file} does not exist or is empty.")
        return

    if (os.path.exists(train_file) and os.path.getsize(train_file) > 0) and \
            (os.path.exists(dev_file) and os.path.getsize(dev_file) > 0):
        print("Training and dev files already exist and contain data. Skipping split.")
        return

    with open(input_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    if len(lines) < 2:
        print("Error: Not enough data to split.")
        return

    random.shuffle(lines)
    split_index = max(1, int(len(lines) * split_ratio))
    training_data, dev_data = lines[:split_index], lines[split_index:]

    # Write training and dev data, ensuring both are not empty
    with open(train_file, 'w', encoding='utf-8') as train_f, \
            open(dev_file, 'w', encoding='utf-8') as dev_f:
        if training_data:
            train_f.writelines(training_data)
        if dev_data:
            dev_f.writelines(dev_data)

    print(f"Data split complete. Training data saved to {train_file}, Dev data saved to {dev_file}")


# Define directory paths and filenames for processing
root = '../'
transformed_dir = os.path.join(root, 'transformed')  # Directory where transformed data files are stored
consolidated_file = os.path.join(root, 'data', 'consolidated_phonetic_data.txt')  # Combined output of all transformed files
training_file_path = os.path.join(root, 'data', 'training.txt')  # Output path for training data
dev_file_path = os.path.join(root, 'data', 'dev.txt')  # Output path for dev data

# Process and consolidate data if necessary
consolidate_transformed_files(transformed_dir, consolidated_file)

# Split data into training and dev sets if they do not already exist
split_data(consolidated_file, training_file_path, dev_file_path)
