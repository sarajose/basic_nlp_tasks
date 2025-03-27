import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, TimeDistributed, Bidirectional

import numpy as np
from conllu import parse_incr 
from sklearn.metrics import classification_report, confusion_matrix, f1_score, recall_score, accuracy_score
import matplotlib.pyplot as plt
import itertools
import os
import pickle

class POSTagger:
    def __init__(self, embedding_dim=50, lstm_units=100, recurrent_dropout=0.1):
        # Hyperparameters
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units
        self.recurrent_dropout = recurrent_dropout
        # Placeholders
        self.word2idx = None
        self.tag2idx = None
        self.max_len = None
        # Model
        self.model = None

    def load_data_conllu(self, file_path):
        """
        Load sentences from a CoNLL-U formatted file
        Each sentence is represented as a list of tuples (word, upos_tag)

        Returns:
            list: A list of sentences, where each sentence is a list of (word, upos_tag) tuples
        """
        sentences = []
        with open(file_path, 'r', encoding='utf-8') as f:
            # parse_incr reads one sentence (token list) at a time.
            for tokenlist in parse_incr(f):
                sentence = []
                for token in tokenlist:
                    # token["form"]: word, token["upos"]: universal POS tag
                    sentence.append((token["form"], token["upos"]))
                sentences.append(sentence)
        return sentences

    def prepare_data(self, sentences, word2idx=None, tag2idx=None, max_len=None):
        """
        Convert sentences (each sentence is a list of (word, tag) tuples)
        into padded sequences of indices and one-hot encoded tag sequences
        
        Returns:
            X: Padded array of word indices.
            y: Padded array of one-hot encoded tag vectors
            word2idx: Dictionary mapping words to indices
            tag2idx: Dictionary mapping tags to indices
            max_len: Maximum sequence length used for padding
        """
        # Build dictionaries if not provided (typically during training)
        if word2idx is None or tag2idx is None:
            # Get unique words and tags from sentences
            words = list({word for sentence in sentences for word, _ in sentence})
            tags = list({tag for sentence in sentences for _, tag in sentence})
            # Start indices from 2 to leave room for 'PAD' (0) and 'UNK' (1)
            word2idx = {w: i + 2 for i, w in enumerate(words)}
            word2idx['UNK'] = 1  # Unknown words
            word2idx['PAD'] = 0  # Padding token
            tag2idx = {t: i + 1 for i, t in enumerate(tags)}
            tag2idx['PAD'] = 0  # Padding tag
        
        # Convert sentences to sequences of indices for words and tags
        X = [[word2idx.get(word, word2idx['UNK']) for word, _ in s] for s in sentences]
        y = [[tag2idx.get(tag, tag2idx['PAD']) for _, tag in s] for s in sentences]

        # Determine the maximum sentence length
        if max_len is None:
            max_len = max(len(seq) for seq in X)
        
        # Save max_len and dictionaries for later use
        self.word2idx = word2idx
        self.tag2idx = tag2idx
        self.max_len = max_len

        # Pad word sequences and sequence similarity, pad with idx for tag 'PAD'
        X = pad_sequences(X, maxlen=max_len, padding='post', value=word2idx['PAD'])
        y = pad_sequences(y, maxlen=max_len, padding='post', value=tag2idx['PAD'])
        
        # Convert tag indices to one-hot encoded vectors for each token
        y = [to_categorical(seq, num_classes=len(tag2idx)) for seq in y]
        
        return np.array(X), np.array(y), word2idx, tag2idx, max_len

    def build_model(self):
        """
        Build the POS tagging model using an Embedding layer followed by a Bidirectional LSTM
        The output layer uses TimeDistributed Dense with softmax activation
        """
        input_dim = len(self.word2idx)
        output_dim = len(self.tag2idx)
        input_length = self.max_len

        self.model = Sequential()
        self.model.add(Embedding(input_dim=input_dim, output_dim=self.embedding_dim,
                                 input_length=input_length, mask_zero=True))
        self.model.add(Bidirectional(LSTM(units=self.lstm_units, return_sequences=True,
                                          recurrent_dropout=self.recurrent_dropout)))
        self.model.add(TimeDistributed(Dense(output_dim, activation='softmax')))
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        self.model.summary()

    def train(self, X_train, y_train, batch_size=32, epochs=5, validation_split=0.1):
        """
        Train the model using the provided training data
        Returns:
            History: A history object containing training loss and metrics per epoch
        """
        history = self.model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs,
                                 validation_split=validation_split, verbose=1)
        return history

    def evaluate(self, X_test, y_test):
        """
        Evaluate the model on test data. Computes accuracy, F1 score, recall, and displays a confusion matrix
        """
        # Get predictions for test data (shape: [num_sentences, max_len, num_tags])
        y_pred_probs = self.model.predict(X_test, verbose=1)
        # Convert probabilities to predicted indices
        y_pred = np.argmax(y_pred_probs, axis=-1)
        # Convert one-hot ground truth to label indices
        y_true = np.argmax(y_test, axis=-1)
        
        # Flatten the predictions and ground truth while filtering out PAD tokens
        y_pred_flat = []
        y_true_flat = []
        for i in range(len(y_true)):
            for j in range(len(y_true[i])):
                # Only include non-PAD tokens
                if y_true[i][j] != self.tag2idx['PAD']:
                    y_pred_flat.append(y_pred[i][j])
                    y_true_flat.append(y_true[i][j])
                    
        # Metrics
        acc = accuracy_score(y_true_flat, y_pred_flat)
        f1 = f1_score(y_true_flat, y_pred_flat, average='weighted')
        recall = recall_score(y_true_flat, y_pred_flat, average='weighted')
        report = classification_report(y_true_flat, y_pred_flat, target_names=self.get_tag_list())
        cm = confusion_matrix(y_true_flat, y_pred_flat)
        
        print("Test Accuracy: {:.4f}".format(acc))
        print("Weighted F1 Score: {:.4f}".format(f1))
        print("Weighted Recall: {:.4f}".format(recall))
        print("\nClassification Report:\n", report)
        self.plot_confusion_matrix(cm, classes=self.get_tag_list(), title='Confusion Matrix')
        
    def get_tag_list(self):
        """
        Get a list of tag names in order of their indices
        Returns:
            list: A list of tag names (excluding PAD) sorted by their index
        """
        # Create a reverse mapping (from index to tag) and return tags sorted
        idx2tag = {idx: tag for tag, idx in self.tag2idx.items()}
        tag_list = [idx2tag[i] for i in sorted(idx2tag) if i != self.tag2idx['PAD']]
        return tag_list

    def plot_confusion_matrix(self, cm, classes, title='Confusion Matrix'):
        """
        Plot the confusion matrix using matplotlib.
        """
        plt.figure(figsize=(10, 8))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
    
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()
        #plt.savefig("confusion_matrix_plt.jpg")

    def predict_sentence(self, sentence):
        """
        Predict the POS tags for a single sentence
        The padded positions (beyond the actual sentence length) might receive arbitrary predictions
        
        Returns:
            list: A list of predicted POS tags for the sentence
        """
        # Create mapping for tags
        idx2tag = {i: tag for tag, i in self.tag2idx.items()}
        # Convert sentence into indices, using 'UNK' for unknown words
        seq = [self.word2idx.get(w, self.word2idx['UNK']) for w in sentence.split()]
        # Pad sequence to self.max_len. Add predictions for extra padded tokens
        padded_seq = pad_sequences([seq], maxlen=self.max_len, padding='post', value=self.word2idx['PAD'])
        pred_probs = self.model.predict(padded_seq)
        pred_indices = np.argmax(pred_probs, axis=-1)[0]
        # Only take predictions for the actual len of the sentence (no padded)
        pred_indices = pred_indices[:len(seq)]
        tags = [idx2tag[idx] for idx in pred_indices]
        return tags

if __name__ == '__main__':
    # Choose model language (change to any locally saved conllu dataset)
    model_language = 'english'

    # File paths and model filenames based on the language
    if model_language == 'english':
        train_file_path = 'data/en_ewt-ud-train.conllu'
        test_file_path = 'data/en_ewt-ud-test.conllu'
        model_filename = 'english_pos_tagger.h5'
        vocab_filename = 'english_vocab.pkl'
    elif model_language == 'catalan':
        train_file_path = 'data/ca_ancora-ud-test.conllu'
        test_file_path = 'data/ca_ancora-ud-test.conllu'
        model_filename = 'catalan_pos_tagger.h5'
        vocab_filename = 'catalan_vocab.pkl'

    # Instantiate POSTagger
    pos_tagger = POSTagger(embedding_dim=50, lstm_units=100, recurrent_dropout=0.1)

    # Check saved model and vocabulary exist
    if os.path.exists(model_filename) and os.path.exists(vocab_filename):
        print("Loading saved model and vocabulary...")
        pos_tagger.model = tf.keras.models.load_model(model_filename)
        with open(vocab_filename, 'rb') as f:
            vocab_data = pickle.load(f)
            pos_tagger.word2idx = vocab_data['word2idx']
            pos_tagger.tag2idx = vocab_data['tag2idx']
            pos_tagger.max_len = vocab_data['max_len']
    else:
        # Training and Evaluation
        print("Loading data...")
        train_sentences = pos_tagger.load_data_conllu(train_file_path)
        test_sentences = pos_tagger.load_data_conllu(test_file_path)

        print("Preparing training data...")
        X_train, y_train, word2idx, tag2idx, max_len = pos_tagger.prepare_data(train_sentences)
        print("Preparing test data...")
        X_test, y_test, _, _, _ = pos_tagger.prepare_data(test_sentences, word2idx=word2idx, tag2idx=tag2idx, max_len=max_len)

        print("Building the model...")
        pos_tagger.build_model()

        print("Training the model...")
        history = pos_tagger.train(X_train, y_train, batch_size=32, epochs=5, validation_split=0.1)
      
        # Metrics per epoch
        print("\nTraining metrics per epoch:")
        for i in range(len(history.history['loss'])):
            print(f"Epoch {i+1}: loss = {history.history['loss'][i]:.4f}, accuracy = {history.history['accuracy'][i]:.4f}, "
                f"val_loss = {history.history['val_loss'][i]:.4f}, val_accuracy = {history.history['val_accuracy'][i]:.4f}")
        
        # Model evaluation on test data
        print("\nEvaluating the model on test data...")
        pos_tagger.evaluate(X_test, y_test)

        # Save the trained model and vocabulary
        pos_tagger.model.save(model_filename)
        with open(vocab_filename, 'wb') as f:
            pickle.dump({'word2idx': pos_tagger.word2idx, 'tag2idx': pos_tagger.tag2idx, 'max_len': pos_tagger.max_len}, f)
    
    # Prediction example
    sentence = "This is a test sentence to check if the POS tagger works"
    #sentence = "Aquesta frase serveix per comprovar si el programa funciona."
    predicted_tags = pos_tagger.predict_sentence(sentence)
    print(f'\nSentence: "{sentence}"')
    print("Predicted Tags:", predicted_tags)
