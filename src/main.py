'''
CMPUT 461 - Assignment 3: Training and Evaluating N-gram Models
Author: Ishaan Meena
Date: 05/11/2024
'''

import argparse
import math
import os
from collections import defaultdict, Counter


# Load data from a file, line by line
def get_data(path):
    if not os.path.exists(path):
        print(f"File {path} not found")
        return []
    with open(path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]


# Generate unigrams
def unigrams(utterances):
    unigram_counts = Counter(word for line in utterances for word in line.split())
    total_unigrams = sum(unigram_counts.values())
    # Return unigram frequency counts and the total count of unigrams
    return unigram_counts, total_unigrams


# Generate bigrams
def bigrams(utterances):
    bigram_counts = defaultdict(Counter)
    unigram_counts, ignored_var = unigrams(utterances)
    # Calculate vocabulary size, which is the number of unique unigrams
    v_size = len(unigram_counts)

    # Process each line in the utterances to build bigram counts
    for line in utterances:
        # Add start ('<s>') and end ('</s>') symbols around each line to mark boundaries
        words = ['<s>'] + line.split() + ['</s>']
        for i in range(len(words) - 1):
            bigram_counts[words[i]][words[i + 1]] += 1

    return bigram_counts, unigram_counts, v_size


# Generate trigrams
def trigrams(utterances):
    # Initialize trigram_counts as a nested defaultdict where each item is a Counter for trigram counts
    trigram_counts = defaultdict(lambda: Counter())
    bigram_counts, unigram_counts, v_size = bigrams(utterances)

    # Process each line in the utterances to build trigram counts
    for line in utterances:
        words = ['<s>', '<s>'] + line.split() + ['</s>']
        # Count each set of three consecutive words to populate the trigram counts
        for i in range(len(words) - 2):
            trigram_counts[(words[i], words[i + 1])][words[i + 2]] += 1

    return trigram_counts, bigram_counts, unigram_counts, v_size


# Calculate the perplexity of a model on a dataset with optional smoothing
def calculate_perplexity(utterances, ngram_counts, model='unigram', v_size=None, smoothing=True, epsilon=1e-10):
    # Initialize the total log probability to accumulate log-probabilities across all words
    total_log_prob = 0
    total_words = sum(len(line.split()) for line in utterances)

    if total_words == 0:
        print("Warning: No words found in dev set. Returning infinite perplexity.")
        return float('inf')

    if model == 'unigram':
        unigram_sum = sum(ngram_counts.values())
    else:
        unigram_sum = None

    for line in utterances:
        words = line.split()

        # Unigram model: Calculate probability for each word individually
        if model == 'unigram':
            # Sum log-probabilities of each word
            total_log_prob += sum(math.log((ngram_counts[word] + 1) / (unigram_sum + v_size))
                                  if smoothing else math.log((ngram_counts[word] + epsilon) / unigram_sum)
                                  for word in words)

        # Bigram model: Calculate probability for each pair of consecutive words
        elif model == 'bigram':
            words = ['<s>'] + words + ['</s>']
            for i in range(len(words) - 1):
                bigram_count = ngram_counts[0][words[i]][words[i + 1]]
                unigram_count = ngram_counts[1][words[i]]  # Get unigram count for the first word

                # Calculate probability
                prob = ((bigram_count + 1) / (unigram_count + v_size) if smoothing
                        else (bigram_count + epsilon) / (unigram_count + epsilon)) if unigram_count > 0 else epsilon
                # Accumulate log-probabilities
                total_log_prob += math.log(prob)

        # Trigram model: Calculate probability for each triplet of consecutive words
        elif model == 'trigram':
            words = ['<s>', '<s>'] + words + ['</s>']
            for i in range(len(words) - 2):
                trigram_count = ngram_counts[0][(words[i], words[i + 1])][words[i + 2]]  # Get trigram count
                bigram_count = ngram_counts[1][words[i]][words[i + 1]]  # Get bigram count

                prob = ((trigram_count + 1) / (bigram_count + v_size) if smoothing
                        else (trigram_count + epsilon) / (bigram_count + epsilon)) if bigram_count > 0 else epsilon
                # Accumulate log-probabilities
                total_log_prob += math.log(prob)

    # Calculate and return perplexity by exponentiating the negative average log probability
    return math.exp(-total_log_prob / total_words)


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train and evaluate n-gram models.")
    parser.add_argument("model", choices=["unigram", "bigram", "trigram"], help="Type of n-gram model to use")
    parser.add_argument("train_file", type=str, help="Path to the training data file")
    parser.add_argument("test_file", type=str, help="Path to the test (dev) data file")
    parser.add_argument("--laplace", action="store_true", help="Apply Laplace smoothing for bigram and trigram models")

    args = parser.parse_args()

    # Load training and dev data
    training_data = get_data(args.train_file)
    dev_data = get_data(args.test_file)

    if not training_data or not dev_data:
        print("Error: Training or dev data is empty. Ensure data files are correctly prepared.")
    else:
        # Train models based on the specified model
        if args.model == "unigram":
            unigram_counts, _ = unigrams(training_data)
            # Calculate perplexity on both training and dev sets
            train_perplexity = calculate_perplexity(training_data, unigram_counts, model="unigram", v_size=len(unigram_counts), smoothing=args.laplace)
            dev_perplexity = calculate_perplexity(dev_data, unigram_counts, model="unigram", v_size=len(unigram_counts), smoothing=args.laplace)
            print(f"Unigram Model Perplexity - Training Set: {train_perplexity}")
            print(f"Unigram Model Perplexity - Dev Set: {dev_perplexity}")

        elif args.model == "bigram":
            bigram_counts, unigram_counts, v_size = bigrams(training_data)
            # Calculate perplexity on both training and dev sets
            train_perplexity = calculate_perplexity(training_data, (bigram_counts, unigram_counts), model="bigram", v_size=v_size, smoothing=args.laplace)
            dev_perplexity = calculate_perplexity(dev_data, (bigram_counts, unigram_counts), model="bigram", v_size=v_size, smoothing=args.laplace)
            print(f"Bigram Model Perplexity (Laplace Smoothing: {args.laplace}) - Training Set: {train_perplexity}")
            print(f"Bigram Model Perplexity (Laplace Smoothing: {args.laplace}) - Dev Set: {dev_perplexity}")

        elif args.model == "trigram":
            trigram_counts, bigram_counts, unigram_counts, v_size = trigrams(training_data)
            # Calculate perplexity on both training and dev sets
            train_perplexity = calculate_perplexity(training_data, (trigram_counts, bigram_counts), model="trigram", v_size=v_size, smoothing=args.laplace)
            dev_perplexity = calculate_perplexity(dev_data, (trigram_counts, bigram_counts), model="trigram", v_size=v_size, smoothing=args.laplace)
            print(f"Trigram Model Perplexity (Laplace Smoothing: {args.laplace}) - Training Set: {train_perplexity}")
            print(f"Trigram Model Perplexity (Laplace Smoothing: {args.laplace}) - Dev Set: {dev_perplexity}")
