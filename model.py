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

# Step 3 - bigram_counts
import torch

def bigram_counts(words, stoi):
    """Return a (V, V) int32 tensor of bigram counts, with '.' around every word."""

    # Get the vocabulary size.
    V = len(stoi)

    # Initialize a matrix to store bigram transition counts.
    # N[i, j] represents how often token j follows token i.
    N = torch.zeros((V, V), dtype=torch.int32)

    # Count character transitions for every word.
    for word in words:

        # Add the special boundary token at the beginning and end.
        chars = ["."] + list(word) + ["."]

        # Convert characters into their corresponding integer IDs.
        ids = [stoi[ch] for ch in chars]

        # Count each consecutive pair of tokens.
        for ch1, ch2 in zip(ids, ids[1:]):
            N[ch1, ch2] += 1

    return N

# Step 4 - bigram_probs
def bigram_probs(N, smoothing=1):
    """Return row-normalized float probabilities after adding smoothing to every count."""

    # Convert the integer count matrix to floating-point values
    # and add the smoothing value to every cell.
    counts = N.float() + smoothing

    # Calculate the total count for each row.
    # keepdim=True preserves the shape for broadcasting.
    row_totals = counts.sum(dim=1, keepdim=True)

    # Normalize each row to obtain transition probabilities.
    # Each row represents the probability distribution of the next character.
    P = counts / row_totals

    return P

# Step 5 - sample_bigram
def sample_bigram(P, itos, n, seed):
    """Sample n names from the bigram model using a seeded generator."""

    # Create one random number generator for reproducible sampling.
    g = torch.Generator().manual_seed(seed)

    names = []

    # Generate the requested number of names.
    for _ in range(n):

        # Every name starts with the special boundary token (ID 0).
        current = 0
        name_chars = []

        while True:

            # Sample the next token from the current token's probability row.
            next_token = torch.multinomial(
                P[current],
                num_samples=1,
                replacement=True,
                generator=g
            ).item()

            # Stop generating when the boundary token appears again.
            if next_token == 0:
                break

            # Convert the sampled token ID back into its character.
            name_chars.append(itos[next_token])

            # Update the current token for the next sampling iteration.
            current = next_token

        # Join the sampled characters into a name.
        names.append("".join(name_chars))

    return names

# Step 6 - bigram_nll
def bigram_nll(P, words, stoi):
    """Return the average negative log-likelihood per bigram as a Python float."""

    # Initialize the total negative log-likelihood and bigram counter.
    total_nll = 0.0
    total_bigrams = 0

    # Calculate the negative log-likelihood for every word.
    for word in words:

        # Add the boundary token at the beginning and end of the word.
        chars = ["."] + list(word) + ["."]

        # Convert characters into their corresponding integer IDs.
        ids = [stoi[ch] for ch in chars]

        # Calculate the negative log-likelihood of every consecutive bigram.
        for first, second in zip(ids, ids[1:]):

            # Add -log(P[first, second]) to the total loss.
            total_nll -= torch.log(P[first, second]).item()

            # Count the processed bigram.
            total_bigrams += 1

    # Prevent division by zero when no bigrams are available.
    if total_bigrams == 0:
        raise ValueError("Cannot calculate NLL because no bigrams were found.")

    # Return the average negative log-likelihood as a Python float.
    return total_nll / total_bigrams

# Step 7 - bigram_dataset
def bigram_dataset(words, stoi):
    """Return (xs, ys): input and next-token IDs for every bigram, as long tensors."""

    # Store the input token IDs and their corresponding next-token IDs.
    xs = []
    ys = []

    # Process every word in the dataset.
    for word in words:

        # Add the boundary token at the beginning and end of each word.
        chars = ["."] + list(word) + ["."]

        # Convert characters into their corresponding integer IDs.
        ids = [stoi[ch] for ch in chars]

        # Create input-target pairs from consecutive token IDs.
        # Each input token predicts the token immediately following it.
        xs.extend(ids[:-1])
        ys.extend(ids[1:])

    # Convert the lists into one-dimensional PyTorch long tensors.
    xs = torch.tensor(xs, dtype=torch.long)
    ys = torch.tensor(ys, dtype=torch.long)

    return xs, ys

# Step 8 - train_neural_bigram
import torch.nn.functional as F

def train_neural_bigram(xs, ys, vocab_size, steps, lr, reg, seed):
    """Train a single linear layer using full-batch gradient descent.
    
    Returns:
        W: Learned weight matrix of shape (vocab_size, vocab_size).
        losses: List of training losses recorded before each update.
    """

    # Initialize a seeded random number generator for reproducibility.
    g = torch.Generator().manual_seed(seed)

    # Initialize the weight matrix with random values.
    # Each row represents an input character, and each column represents
    # the logits for predicting the next character.
    W = torch.randn(
        (vocab_size, vocab_size),
        generator=g,
        requires_grad=True
    )

    # Store the loss from each training iteration.
    losses = []

    # Create row indices for selecting the correct target probabilities.
    row_indices = torch.arange(xs.shape[0])

    # Perform full-batch gradient descent.
    for _ in range(steps):

        # Convert input token IDs into one-hot encoded floating-point vectors.
        xenc = F.one_hot(xs, num_classes=vocab_size).float()

        # Forward pass: calculate logits using the linear layer.
        logits = xenc @ W

        # Convert logits into unnormalized probabilities.
        counts = logits.exp()

        # Apply softmax normalization so each probability row sums to 1.
        probs = counts / counts.sum(dim=1, keepdim=True)

        # Calculate the average negative log-likelihood for correct targets.
        nll_loss = -torch.log(probs[row_indices, ys]).mean()

        # Add L2 regularization to penalize large weight values.
        reg_loss = reg * W.pow(2).mean()

        # Combine the negative log-likelihood and regularization losses.
        loss = nll_loss + reg_loss

        # Record the current loss before updating the weights.
        losses.append(loss.item())

        # Reset the gradients from the previous iteration.
        W.grad = None

        # Backpropagate to calculate gradients of the loss with respect to W.
        loss.backward()

        # Update the weights using gradient descent without tracking gradients.
        with torch.no_grad():
            W -= lr * W.grad

    return W, losses

# Step 9 - build_context_dataset
def build_context_dataset(words, stoi, block_size):
    """Return (X, Y): '.'-padded rolling contexts and next-token IDs."""

    # Store the context windows and their corresponding target IDs.
    X = []
    Y = []

    # Process each word independently.
    for word in words:

        # Initialize the context with block_size boundary tokens.
        context = [0] * block_size

        # Process every character, including the final boundary token.
        for ch in list(word) + ["."]:

            # Convert the current character into its integer ID.
            ix = stoi[ch]

            # Record a copy of the current context.
            X.append(context.copy())

            # Record the current character as the prediction target.
            Y.append(ix)

            # Slide the context window one position to the left
            # and append the current character ID.
            context = context[1:] + [ix]

    # Convert the collected contexts and targets into long tensors.
    # Reshape X to ensure the required (n, block_size) dimensions.
    X = torch.tensor(X, dtype=torch.long).reshape(-1, block_size)
    Y = torch.tensor(Y, dtype=torch.long)

    return X, Y

# Step 10 - split_words
import random

def split_words(words, seed):
    """Shuffle a copy with random.Random(seed); return the 80/10/10 (train, dev, test) split."""

    # Create a copy so that the original words list remains unchanged.
    shuffled_words = words.copy()

    # Shuffle the copied list using a seeded random generator.
    # This ensures reproducible and consistent dataset splitting.
    random.Random(seed).shuffle(shuffled_words)

    # Calculate the boundaries for the 80/10/10 split.
    n1 = int(0.8 * len(shuffled_words))
    n2 = int(0.9 * len(shuffled_words))

    # Divide the shuffled dataset into training, development, and test sets.
    train = shuffled_words[:n1]
    dev = shuffled_words[n1:n2]
    test = shuffled_words[n2:]

    return train, dev, test

# Step 11 - MLP
class MLP:
    def __init__(self, vocab_size, block_size, emb_dim, hidden, seed):
        """Draw C, W1, b1, W2, b2 (in that order) from one seeded generator; all require grad."""

        # Store the context window size.
        self.block_size = block_size

        # Create one seeded generator for reproducible parameter initialization.
        g = torch.Generator().manual_seed(seed)

        # Character embedding table:
        # Maps each character ID to an embedding vector.
        self.C = torch.randn(
            (vocab_size, emb_dim),
            generator=g,
            requires_grad=True
        )

        # First layer weights:
        # Maps the concatenated character embeddings to the hidden layer.
        self.W1 = torch.randn(
            (block_size * emb_dim, hidden),
            generator=g,
            requires_grad=True
        )

        # First layer bias.
        self.b1 = torch.randn(
            (hidden,),
            generator=g,
            requires_grad=True
        )

        # Second layer weights:
        # Maps hidden layer activations to vocabulary-sized output logits.
        self.W2 = torch.randn(
            (hidden, vocab_size),
            generator=g,
            requires_grad=True
        )

        # Second layer bias.
        self.b2 = torch.randn(
            (vocab_size,),
            generator=g,
            requires_grad=True
        )

    def parameters(self):
        """Return all trainable parameters in the required order."""

        return [self.C, self.W1, self.b1, self.W2, self.b2]

    def __call__(self, X):
        """Convert context token IDs into vocabulary-sized output logits."""

        # Look up the embedding vector for every character in each context.
        emb = self.C[X]

        # Flatten each example's embeddings into a single feature vector.
        # Resulting shape: (n, block_size * emb_dim).
        emb = emb.reshape(emb.shape[0], -1)

        # Hidden layer: linear transformation followed by tanh activation.
        h = torch.tanh(emb @ self.W1 + self.b1)

        # Output layer: transform hidden activations into vocabulary logits.
        logits = h @ self.W2 + self.b2

        return logits

# Step 12 - evaluate_loss
def evaluate_loss(model, X, Y):
    """Return the mean cross-entropy loss as a Python float, without gradients."""

    # Disable gradient tracking during evaluation.
    with torch.no_grad():

        # Perform the forward pass to obtain the model's output logits.
        logits = model(X)

        # Calculate the mean cross-entropy loss against the target IDs.
        loss = F.cross_entropy(logits, Y)

    # Convert the loss tensor into a standard Python float.
    return loss.item()

# Step 13 - train_mlp (not yet solved)
# TODO: implement

# Step 14 - sample_mlp (not yet solved)
# TODO: implement

