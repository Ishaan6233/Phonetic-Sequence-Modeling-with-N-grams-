# Phonetic-Sequence-Modeling-with-N-grams

## Description

This project implements n-gram language models to evaluate the likelihood of various phonetic sequences in English using cleaned and transformed CHILDES data. By applying unigram, bigram, and trigram models with Laplace smoothing, the code can distinguish between sequences that resemble English and those that do not, based on computed perplexity scores.

The project is structured as follows:
1. **Data Preparation**: Load, clean, and transform the input data, splitting it into training and dev sets.
2. **N-gram Model Training**: Generate and train unigram, bigram, and trigram models using Laplace smoothing.
3. **Model Evaluation**: Calculate perplexity scores for each model type to assess their effectiveness on unseen data.

---
## Requirements

### Python Version
This project was developed and tested using **Python 3.8.10**. Please ensure you are using this version to avoid discrepancies.

### Python Packages
The following packages are required to run the code:
- **collections**: for efficient counting and storage of n-gram data
- **argparse**: for command-line argument parsing.
- **math**: for mathematical operations.

**No external packages** are needed. All modules are part of the Python Standard Library.

---
## File Structure
- **`src/main.py`**: Main script for data loading, model training, and evaluation.
- **`src/split.py`**: Script for preprocessing and splitting the transformed data for training and evaluation.
- **`data/`**: Contains `training.txt` and `dev.txt` files used for n-gram model training and evaluation.

---
## Script Overview

### main.py
- **Purpose**: Trains unigram, bigram, and trigram n-gram models on phonetic data and evaluates them by calculating perplexity scores.
- **Key Functions**:
  - `load_data(file_path)`: Loads and preprocesses data from a specified file path.
  - `generate_unigrams`, `generate_bigrams`, `generate_trigrams`: Create unigram, bigram, and trigram models with optional Laplace smoothing.
  - `calculate_perplexity`: Computes the perplexity score for a given n-gram model on test data.

### split.py
- **Purpose**: Prepares `training.txt` and `dev.txt` files from raw transformed data files.
- **Functions**:
  - `consolidate_transformed_files(transformed_dir, consolidated_output)`: Consolidates multiple transformed data files into one.
  - `split_data(input_file, train_file, dev_file, split_ratio=0.8)`: Splits consolidated data into training and dev files based on a specified ratio.

---
## Data
The `data` directory includes:
- `training.txt`: The training set used for n-gram model training.
- `dev.txt`: The dev set for evaluating perplexity on unseen data.

**Note**: Use `src/split.py` to generate `training.txt` and `dev.txt` from transformed data files.

---
### Execution

#### Running the Code
1. **Prepare the Environment**:
   - Ensure Python 3.8.10 is installed.
   - Navigate to the directory containing `main.py`.
   - Verify that `data/` contains both `training.txt` and `dev.txt`. If not, then use the split.py script.

Use the following command in the current directory.
`python3 src/main.py <model_type> <train_file> <test_file> [--laplace]`

### Parameters:

* <model_type>: Choose from unigram, bigram, or trigram.
* <train_file>: Path to the training data file.
* <test_file>: Path to the dev/test data file.
* --laplace: (Optional) Include this flag to enable Laplace smoothing.

Example:
`python3 src/main.py bigram data/training.txt data/dev.txt --laplace`

---
### Input & Output
1. Input: `data/training.txt`: Phonetic sequences for model training.
          `data/dev.txt`: Phonetic sequences for model evaluation.
2. Output: The console displays unigram, bigram, and trigram perplexity scores.

---
## Evaluation

The following table summarizes the perplexity scores for each model on the training and dev sets.

|Model           | Smoothing  | Training set PPL | Dev set PPL |
|----------------|----------- | ---------------- | ----------- |
|unigram         | -          |       70.63      |    193.07   |
|bigram          | unsmoothed |       22.65      |    182.72   |
|bigram          | Laplace    |       111.0      |    195.13   |
|trigram         | unsmoothed |       9.25       |    259.47   |
|trigram         | Laplace    |       650.98     |    2075.50  |

