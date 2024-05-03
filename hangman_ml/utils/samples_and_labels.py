"""Create samples and labels for training the hangman model.

The hangman model is trained to predict the probability of each unguessed letter in a masked word.
The model is trained using a dataset of masked words and the incorrect guesses.
The dataset is created by generating up to `max_samples_per_word` possible combinations of masked words and incorrect guesses.

Samples:
- `masked_word`:
    A masked word with the guessed letters filled in.
    Represented as an array of one-hot encoded characters.
    The masked word is padded to the maximum word length.
- `previous_guesses`:
    A one-hot encoded vector of the guessed letters.

Labels:
- `next_guess_probs`:
    A vector of probabilities for each unguessed letter in the masked word.
    The probability is calculated using n-grams of length `MIN_N_GRAM` to `MAX_N_GRAM`.
    The probability is weighted by the length of the n-gram.

Notes:
- While generating the dataset, the most common two letters in words of the same length are guessed first. Rest of the guesses are random.
- The dataset is generated in parallel using multiple processes.
- Validation set is contaminated since `len_to_words_trie` has information about the training set while generating the labels.

The dataset is saved to an HDF5 file.
"""

import collections
import itertools
from multiprocessing.managers import DictProxy
import random
import string
from typing import Any

import h5py
from keras.preprocessing.sequence import pad_sequences
import numpy as np

from .n_gram_trie import NGramTrie
from .trie import Trie

CHAR_TO_IDX = {char: i for i, char in enumerate(string.ascii_lowercase + "_")}
IDX_TO_CHAR = {i: char for char, i in CHAR_TO_IDX.items()}
CHAR_TO_ENCODING = {
    char: np.eye(len(CHAR_TO_IDX))[idx] for char, idx in CHAR_TO_IDX.items()
}

MAX_WORD_LEN = 30
ALPHABET_LEN = len(CHAR_TO_IDX) - 1

MIN_N_GRAM = 2
MAX_N_GRAM = 6

N_GRAM_LEN_TO_FACTOR = {
    2: 1,
    3: 2,
    4: 4,
    5: 5,
    6: 6,
}
"""Factor to weight the n-gram probabilities by the length of the n-gram."""


def build_n_grams(words: list[str]) -> dict[str, NGramTrie]:
    """Create a dictionary mapping n-grams to their counts.

    Only n-grams of length `MIN_N_GRAM` to `MAX_N_GRAM` are considered.
    """
    return {n: NGramTrie(words, n) for n in range(MIN_N_GRAM, MAX_N_GRAM + 1)}


def build_len_to_most_frequent_letters(
    words: list[str],
) -> dict[int, list[tuple[str, int]]]:
    """Create a dictionary mapping word lengths to the most frequent letters.

    The most frequent letters are sorted by their frequency.
    """
    len_to_words = build_len_to_words_dict(words)
    len_to_most_frequent_letters = {}
    for length, words in len_to_words.items():
        all_words_str = "".join(words)
        counter = collections.Counter(all_words_str).most_common()
        len_to_most_frequent_letters[length] = counter
    return len_to_most_frequent_letters


def build_len_to_words_dict(words: list[str]) -> dict[int, list[str]]:
    """Create a dictionary mapping word lengths to words."""
    len_to_words = collections.defaultdict(list)
    for word in words:
        len_to_words[len(word)].append(word)
    return len_to_words


def build_len_to_words_trie(words: list[str]) -> dict[int, Trie]:
    """Create a dictionary mapping word lengths to Trie."""
    len_to_words = build_len_to_words_dict(words)
    len_to_words_trie = {}
    for length, words in len_to_words.items():
        len_to_words_trie[length] = Trie(words)
    return len_to_words_trie


def find_matching_words_in_trie(
    masked_word: str,
    incorrect_guesses: set[str],
    len_to_words_trie: dict[int, Trie],
) -> list[str]:
    """Find words that match a given masked word."""
    len_word = len(masked_word)
    if len_word not in len_to_words_trie:
        raise ValueError(f"No words of length {len_word} found in tries.")

    trie = len_to_words_trie[len(masked_word)]
    return trie.get_matches(masked_word, incorrect_guesses)


def _get_n_gram_probabilities(
    masked_word: str,
    n: int,
    n_grams: dict[str, NGramTrie],
    incorrect_guesses: set[str],
) -> np.ndarray:
    """Get the letter probabilities for a masked word using mathcing n-grams of length `n`.

    The n-gram is only considered if there is only one missing letter.
    """
    vector = np.zeros(ALPHABET_LEN)

    all_matches_with_count: list[dict[str, int]] = []
    for i in range(len(masked_word) - n + 1):
        n_gram = masked_word[i : i + n]

        count_visible_leters = n - n_gram.count("_")
        if count_visible_leters == n - 1:
            matches_with_count = n_grams[n].get_matching_ngrams_with_count(
                n_gram, incorrect_guesses
            )
            all_matches_with_count.append(matches_with_count)

    masked_word_chars = set(masked_word) - {"_"}
    for matches_with_count in all_matches_with_count:
        for n_gram, count in matches_with_count.items():
            for letter in n_gram:
                if letter not in incorrect_guesses and letter not in masked_word_chars:
                    idx = CHAR_TO_IDX[letter]
                    vector[idx] += count

    total_count = np.sum(vector)
    return vector / total_count if total_count > 0 else vector


def _label_from_masked_word_and_guesses(
    masked_word: str,
    guesses: set[str],
    n_grams: dict[str, NGramTrie],
) -> np.ndarray:
    """Create a label for a masked word.

    The label is the probability of each unguessed letter in the masked word.
    It is calculated using n-grams of length MIN_N_GRAM to MAX_N_GRAM.

    The probability is weighted by using a factor based on the length of the n-gram.
    Longer n-grams are given more weight.
    """
    vector = np.zeros(ALPHABET_LEN)

    incorrect_guesses = guesses - set(masked_word)

    for n in range(MIN_N_GRAM, MAX_N_GRAM + 1):
        n_gram_vector = (
            _get_n_gram_probabilities(masked_word, n, n_grams, incorrect_guesses)
            * N_GRAM_LEN_TO_FACTOR[n]
        )
        vector += n_gram_vector

    total_count = np.sum(vector)
    return vector / total_count if total_count > 0 else vector


def _get_guesses_encoding(guesses: set[str]) -> np.ndarray:
    """Create a one-hot encoded vector of the guessed letters."""
    vector = np.zeros(ALPHABET_LEN)
    idxs = [CHAR_TO_IDX[guess] for guess in guesses]
    vector[idxs] = 1
    return vector


def sample_from_masked_word_and_guesses(
    masked_word: str, guesses: set[str]
) -> tuple[np.ndarray, np.ndarray]:
    """Create a sample from a masked word and guesses."""
    guesses_encoding = _get_guesses_encoding(guesses)

    masked_word_encoding = np.zeros((len(masked_word), ALPHABET_LEN + 1))
    for r, char in enumerate(masked_word):
        masked_word_encoding[r, :] = CHAR_TO_ENCODING[char]

    return masked_word_encoding, guesses_encoding


def _generate_random_guess_combinations(
    all_guessable_chars: set, max_guess_combinations: int
) -> list[tuple[str]]:
    """Generate random guess combinations."""
    guess_combinations = []
    # Generate all combinations
    for i in range(len(all_guessable_chars)):
        for guessed_letters_combo in itertools.combinations(all_guessable_chars, i):
            guess_combinations.append(guessed_letters_combo)

    random.shuffle(guess_combinations)
    return guess_combinations[:max_guess_combinations]


def _get_samples_and_labels_for_word(
    word: str,
    guessed_chars: set[str],
    guessable_chars: set[str],
    max_samples: int,
    cache: DictProxy,
    len_to_most_frequent_letters: dict[int, list[tuple[str, int]]],
    n_grams: dict[str, NGramTrie],
    lock: Any,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """Create samples and labels for a given word."""
    if len(word) > MAX_WORD_LEN:
        raise ValueError(
            f"Word {word} is longer than the maximum word length of {MAX_WORD_LEN}"
        )

    word_chars = set(word)

    # Initial 2 guesses are the most common chars in the same length words
    num_most_common_guesses = 2
    guessed_most_common_chars: set[str] = set()
    for char, _ in len_to_most_frequent_letters[len(word)]:
        guessed_most_common_chars.add(char)
        correct_guesses = word_chars.intersection(guessed_most_common_chars)
        if len(correct_guesses) == num_most_common_guesses:
            break

    all_guessed_chars = guessed_chars.union(guessed_most_common_chars)

    all_guessable_chars = word_chars.union(guessable_chars) - all_guessed_chars
    guess_combinations = _generate_random_guess_combinations(
        all_guessable_chars=all_guessable_chars,
        max_guess_combinations=max_samples,
    )

    masked_word_encodings = []
    guesses_encodings = []
    labels = []
    # Generate all possible combinations of word_chars except all chars
    for guess_combination in guess_combinations:
        guessed_chars_for_sample = set(guess_combination).union(all_guessed_chars)

        masked_word = "".join(
            [char if char in guessed_chars_for_sample else "_" for char in word]
        )
        cache_key = f"{masked_word}-{''.join(guessed_chars_for_sample)}"

        if "_" not in masked_word:
            continue

        with lock:
            if cache_key in cache:
                continue
            cache[cache_key] = True

        masked_word_encoding, guesses_encoding = sample_from_masked_word_and_guesses(
            masked_word,
            guessed_chars_for_sample,
        )
        label = _label_from_masked_word_and_guesses(
            masked_word,
            guessed_chars_for_sample,
            n_grams,
        )

        masked_word_encodings.append(masked_word_encoding)
        guesses_encodings.append(guesses_encoding)
        labels.append(label)
    return masked_word_encodings, guesses_encodings, labels


def get_samples_and_labels(
    words: list[str],
    guessed_chars: set[str],
    guessable_chars: set[str],
    max_samples_per_word: int,
    cache: DictProxy,
    len_to_most_frequent_letters: dict[int, list[tuple[str, int]]],
    n_grams: dict[str, NGramTrie],
    lock: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create samples and labels for a list of words."""
    masked_words = []
    previous_guesses = []
    next_guess_probs = []
    for word in words:
        masked_word_encodings, guesses_encodings, labels = (
            _get_samples_and_labels_for_word(
                word=word,
                guessed_chars=guessed_chars,
                guessable_chars=guessable_chars,
                max_samples=max_samples_per_word,
                cache=cache,
                len_to_most_frequent_letters=len_to_most_frequent_letters,
                n_grams=n_grams,
                lock=lock,
            )
        )
        masked_words.extend(masked_word_encodings)
        previous_guesses.extend(guesses_encodings)
        next_guess_probs.extend(labels)

    padded_masked_words = pad_sequences(
        masked_words, maxlen=MAX_WORD_LEN, padding="post"
    )
    previous_guesses = np.array(previous_guesses)
    next_guess_probs = np.array(next_guess_probs)
    return padded_masked_words, previous_guesses, next_guess_probs


def save_data(
    words: list[str],
    guessed_chars: set[str],
    guessable_chars: set[str],
    max_samples_per_word: int,
    filename: str,
    start: int,
    end: int,
    batch_size: int,
    cache: DictProxy,
    len_to_most_frequent_letters: dict[int, list[tuple[str, int]]],
    n_grams: dict[str, NGramTrie],
    lock: Any,
):
    """Save samples and labels to a file."""
    for i in range(start, end, batch_size):
        batch_end = min(i + batch_size, end)
        print(
            f"Processing: {i}-{batch_end}. End: {end}. Length of all words to process: {len(words)}"
        )

        words_batch = words[i:batch_end]
        masked_words, previous_guesses, next_guess_probs = get_samples_and_labels(
            words=words_batch,
            guessed_chars=guessed_chars,
            guessable_chars=guessable_chars,
            max_samples_per_word=max_samples_per_word,
            cache=cache,
            len_to_most_frequent_letters=len_to_most_frequent_letters,
            n_grams=n_grams,
            lock=lock,
        )
        if len(masked_words) == 0:
            continue

        with lock:
            with h5py.File(filename, "a") as hf:
                if "masked_words" in hf and "previous_guesses" in hf and "next_guess_probs" in hf:
                    hf["masked_words"].resize((hf["masked_words"].shape[0] + masked_words.shape[0]), axis = 0)
                    hf["masked_words"][-masked_words.shape[0]:] = masked_words

                    hf["previous_guesses"].resize((hf["previous_guesses"].shape[0] + previous_guesses.shape[0]), axis = 0)
                    hf["previous_guesses"][-previous_guesses.shape[0]:] = previous_guesses

                    hf["next_guess_probs"].resize((hf["next_guess_probs"].shape[0] + next_guess_probs.shape[0]), axis = 0)
                    hf["next_guess_probs"][-next_guess_probs.shape[0]:] = next_guess_probs
                else:
                    hf.create_dataset("masked_words", data=masked_words, maxshape=(None, *masked_words.shape[1:]), compression="gzip", compression_opts=2)
                    hf.create_dataset("previous_guesses", data=previous_guesses, maxshape=(None, *previous_guesses.shape[1:]), compression="gzip", compression_opts=2)
                    hf.create_dataset("next_guess_probs", data=next_guess_probs, maxshape=(None, *next_guess_probs.shape[1:]), compression="gzip", compression_opts=2)

    print(f"Finished processing: {start}-{end}")
