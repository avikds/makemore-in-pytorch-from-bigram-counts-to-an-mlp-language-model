"""
makemore in PyTorch: From Bigram Counts to an MLP Language Model

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - load_names
import os
import tempfile
import urllib.request

def load_names():
    """Download and cache makemore's names.txt; return a list of lowercase names."""

    # Define the dataset URL and its cache location.
    url = "https://raw.githubusercontent.com/karpathy/makemore/master/names.txt"
    cache_path = os.path.join(tempfile.gettempdir(), "makemore_names.txt")

    # Download the dataset only if it is not already cached.
    if not os.path.isfile(cache_path):
        urllib.request.urlretrieve(url, cache_path)

    # Read the dataset using UTF-8 encoding and normalize each name.
    with open(cache_path, "r", encoding="utf-8") as file:
        names = [
            line.strip().lower()
            for line in file
            if line.strip()
        ]

    return names

# Step 2 - build_vocab
def build_vocab(words):
    """Return (stoi, itos): '.' -> 0, the sorted characters -> 1, 2, ..."""

    # Collect all distinct characters and sort them alphabetically.
    # Reserve '.' exclusively as the special start/end token.
    chars = sorted(set("".join(words)) - {"."})

    # Map each character to a unique integer, starting from 1.
    stoi = {ch: i + 1 for i, ch in enumerate(chars)}

    # Assign 0 to the special token representing the start/end of a name.
    stoi["."] = 0

    # Create the exact inverse mapping: integer IDs to characters.
    itos = {i: ch for ch, i in stoi.items()}

    return stoi, itos

# Step 3 - bigram_counts (not yet solved)
# TODO: implement

# Step 4 - bigram_probs (not yet solved)
# TODO: implement

# Step 5 - sample_bigram (not yet solved)
# TODO: implement

# Step 6 - bigram_nll (not yet solved)
# TODO: implement

# Step 7 - bigram_dataset (not yet solved)
# TODO: implement

# Step 8 - train_neural_bigram (not yet solved)
# TODO: implement

# Step 9 - build_context_dataset (not yet solved)
# TODO: implement

# Step 10 - split_words (not yet solved)
# TODO: implement

# Step 11 - MLP (not yet solved)
# TODO: implement

# Step 12 - evaluate_loss (not yet solved)
# TODO: implement

# Step 13 - train_mlp (not yet solved)
# TODO: implement

# Step 14 - sample_mlp (not yet solved)
# TODO: implement

