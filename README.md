# Hangman Machine Learning Model

This Hangman Machine Learning Model is designed to play the hangman game. The model is designed to predict the next letter in a masked word, given the current state of the word and the guessed letters. The model is trained using a dataset generated from a dictionary of 250,000 words, leveraging both trie data structures and n-gram analysis.


## Solution Structure

The solution comprises several components:

1. `Trie` Data Structure: Implemented as `Trie` class, it is used to efficiently store words and quickly search for matches based on a given masked word.

2. `NGramTrie` Data Structure: Similar to `Trie`, it's used to store n-grams of words along with their counts, enabling probabilistic analysis for predicting the next letter.

3. Dataset Generation: The utilities in `samples_and_labels.py` are used to generate a dataset to train the model.

4. Model Architecture: An LSTM/NN model is built incorporating both the masked word and guessed letters as inputs to predict the probability distribution of the next letter.

5. Training and Evaluation: The model is trained using the generated dataset and evaluated on a validation set to ensure its effectiveness. The model is also tested on a separate test set.


## Dataset Generation

- Parallel Processing: To handle large datasets efficiently, dataset generation is parallelized using multiple processes, speeding up the process. 7,665,680 training and 1,912,161 validation samples and labels are generated and stored in chunks in a .h5 file, allowing for efficient access to the samples and labels during training. A cache is used to prevent duplicate samples from being generated.

- Optimized Guessing Strategy: The dataset generation algorithm prioritizes the most common letters in words of the same length as the masked word, followed by random guesses, ensuring a balanced dataset.

- Samples and Labels: Masked words are represented as an array of one-hot encoded characters while the previous guesses are represented as a one-hot encoded vector. Labels are represented as a vector of probabilities for each unguessed letter in the masked word, calculated using n-grams of length 2 to 6.

- Further Processing: The dataset is then shuffled and further processed to apply smoothing to labels, ensuring a more robust training process and better generalization.

- Validation Set Contamination: We note that the validation set is contaminated since it has information about the training set during label generation.


## Model Architecture

- The architecture is designed to handle the dynamic nature of the Hangman game, where the model must predict the likelihood of each letter given the current state of the word and the guesses made.

- Bidirectional LSTM: The model architecture includes a bidirectional LSTM layer to capture both past and future contexts of the hidden word, aiding in predicting the next letter.

- Concatenating the LSTM output with the guessed letters combines the contextual information from the word with the knowledge of previous guesses.

- L2 regularization is applied and Dropout layers are incorporated to mitigate overfitting and enhance model's generalization capabilities.

- Output Layer: Softmax activation is used in the output layer for predicting the probabilities of unguessed letters.


## Letter Prediction

`predict_next_letter` function predicts the next letter given the masked word and guesses. It incorporates multiple strategies for making the next guess. The algorithm is as follows:

1. If there are less than 2 correct guesses and 75% of the word is unknown, use the trie to predict the next letter, utilizing similar length words that do not contain incorrect guesses.
2. Otherwise, predict the next letter using the model.
3. Set the probabilities of guessed letters to 0.
4. If there are no valid predictions left, guess the most frequent letter for the given word length.


## Conclusion

The Hangman Machine Learning Model demonstrates an effective approach to predicting the next letter in a word based on the current state and guessed letters. By leveraging trie and n-gram data structures along with LSTMs and neural network modeling, the solution achieves a balance between efficiency and accuracy. The `predict_next_letter` algorithm was tested on a separate dataset and achieved a success rate of 50% in winning the games it played within 6 guesses. With further optimization and fine-tuning, the model has the potential to serve as a robust and scalable solution for Hangman game applications.


## Contact

For any questions or inquiries, please contact [atayardimci@outlook.com](mailto:atayardimci@outlook.com).
