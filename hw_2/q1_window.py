#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Q1: A window into NER (PyTorch Implementation)
"""

import argparse
import sys
import time
import logging
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from .util import print_sentence, write_conll
from .data_util import load_and_preprocess_data, load_embeddings, read_conll, ModelHelper
from .defs import LBLS

logger = logging.getLogger("hw3.q1")
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)


class Config:
    """Holds model hyperparams and data information."""
    n_word_features = 2  # Number of features for every word in the input.
    window_size = 1  # The size of the window to use.
    n_window_features = n_word_features * (2 * window_size + 1)  # Total number of features for each window.
    n_classes = 5
    dropout = 0.5
    embed_size = 50
    hidden_size = 200
    batch_size = 2048
    n_epochs = 10
    lr = 0.001

    def __init__(self, output_path=None):
        if output_path:
            self.output_path = output_path
        else:
            self.output_path = f"results/window/{datetime.now():%Y%m%d_%H%M%S}/"
        self.model_output = self.output_path + "model.weights"
        self.eval_output = self.output_path + "results.txt"
        self.log_output = self.output_path + "log"
        self.conll_output = self.output_path + "window_predictions.conll"


def make_windowed_data(data, start, end, window_size=1):
    """Creates windowed data points from the input sequences."""
    windowed_data = []
    for sentence, labels in data:
        padded_sentence = [start] * window_size + sentence + [end] * window_size
        for i in range(len(sentence)):
            window = []
            for j in range(-window_size, window_size + 1):
                window.extend(padded_sentence[i + window_size + j])
            windowed_data.append((window, labels[i]))
    return windowed_data


class WindowModel(nn.Module):
    """
    Feedforward neural network with an embedding layer and single hidden layer.
    """

    def __init__(self, helper, config, pretrained_embeddings):
        super(WindowModel, self).__init__()
        self.helper = helper
        self.config = config

        # Initialize embeddings
        self.embeddings = nn.Embedding.from_pretrained(
            torch.FloatTensor(pretrained_embeddings),
            padding_idx=0
        )

        # Define layers
        self.dropout = nn.Dropout(p=config.dropout)
        self.hidden = nn.Linear(config.n_window_features * config.embed_size, config.hidden_size)
        self.output = nn.Linear(config.hidden_size, config.n_classes)

        # Initialize weights
        nn.init.xavier_uniform_(self.hidden.weight)
        nn.init.xavier_uniform_(self.output.weight)

    def forward(self, inputs):
        """
        Forward pass of the model.
        Args:
            inputs: Tensor of shape (batch_size, n_window_features)
        Returns:
            out: Tensor of shape (batch_size, n_classes)
        """
        # Get embeddings for all words in all windows
        embeds = self.embeddings(inputs)  # shape: (batch_size, n_window_features, embed_size)
        embeds = embeds.view(embeds.size(0), -1)  # shape: (batch_size, n_window_features * embed_size)

        # Hidden layer with ReLU activation
        hidden = torch.relu(self.hidden(embeds))
        hidden = self.dropout(hidden)

        # Output layer (no activation, handled by loss function)
        out = self.output(hidden)
        return out

    def preprocess_sequence_data(self, examples):
        """Convert examples into windowed data."""
        return make_windowed_data(examples,
                                  start=self.helper.START,
                                  end=self.helper.END,
                                  window_size=self.config.window_size)

    def predict(self, inputs):
        """
        Make predictions for the provided batch of data
        Args:
            inputs: torch.Tensor of shape (n_samples, n_features)
        Returns:
            predictions: np.ndarray of shape (n_samples,)
        """
        self.eval()  # Set model to evaluation mode
        with torch.no_grad():
            logits = self.forward(inputs)
            predictions = torch.argmax(logits, dim=1)
        return predictions.cpu().numpy()

    def predict_on_batch(self, inputs_batch):
        """Make predictions for the provided batch of data"""
        inputs_tensor = torch.LongTensor(inputs_batch)
        return self.predict(inputs_tensor)

    def train_on_batch(self, inputs, labels):
        """
        Train model on a batch of data
        Args:
            inputs: torch.Tensor of shape (batch_size, n_features)
            labels: torch.Tensor of shape (batch_size,)
        Returns:
            loss: training loss
        """
        self.train()  # Set model to training mode
        self.optimizer.zero_grad()

        logits = self.forward(inputs)
        loss = self.criterion(logits, labels)

        loss.backward()
        self.optimizer.step()

        return loss.item()

    def consolidate_predictions(self, examples_raw, examples, preds):
        """Batch the predictions into groups of sentence length."""
        ret = []
        i = 0
        for sentence, labels in examples_raw:
            labels_ = preds[i:i + len(sentence)]
            i += len(sentence)
            ret.append([sentence, labels, labels_])
        return ret

    def output(self, sess, inputs):
        """
        Reports the output of the model on examples (uses helper to featurize each example).
        """
        inputs_preprocessed = self.preprocess_sequence_data(inputs)
        return self.consolidate_predictions(
            inputs,
            inputs_preprocessed,
            self.predict_on_batch(inputs_preprocessed)
        )

    def fit(self, train_data, dev_data, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Train model on training data and evaluate on dev data
        """
        self.to(device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters(), lr=self.config.lr)

        best_dev_acc = 0

        for epoch in range(self.config.n_epochs):
            # Training
            self.train()
            train_loss = 0
            train_batches = 0

            # Create batches
            indices = np.random.permutation(len(train_data))
            for batch_start in range(0, len(indices), self.config.batch_size):
                batch_indices = indices[batch_start:batch_start + self.config.batch_size]
                batch_inputs = torch.LongTensor([train_data[i][0] for i in batch_indices]).to(device)
                batch_labels = torch.LongTensor([train_data[i][1] for i in batch_indices]).to(device)

                loss = self.train_on_batch(batch_inputs, batch_labels)
                train_loss += loss
                train_batches += 1

            # Evaluation
            dev_acc = self.evaluate(dev_data, device)

            logger.info(f"Epoch {epoch + 1}/{self.config.n_epochs}")
            logger.info(f"Average training loss: {train_loss / train_batches:.4f}")
            logger.info(f"Dev accuracy: {dev_acc:.4f}")

            # Save best model
            if dev_acc > best_dev_acc:
                best_dev_acc = dev_acc
                torch.save(self.state_dict(), self.config.model_output)

    def evaluate(self, data, device):
        """
        Evaluate model on data
        Returns:
            accuracy: Accuracy on data
        """
        self.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            # Create batches
            for batch_start in range(0, len(data), self.config.batch_size):
                batch_end = batch_start + self.config.batch_size
                batch_inputs = torch.LongTensor([data[i][0] for i in range(batch_start, min(batch_end, len(data)))]).to(
                    device)
                batch_labels = torch.LongTensor([data[i][1] for i in range(batch_start, min(batch_end, len(data)))]).to(
                    device)

                outputs = self.forward(batch_inputs)
                predictions = torch.argmax(outputs, dim=1)

                total += batch_labels.size(0)
                correct += (predictions == batch_labels).sum().item()

        return correct / total


def test_make_windowed_data():
    sentences = [[[1, 1], [2, 0], [3, 3]]]
    sentence_labels = [[1, 2, 3]]
    data = list(zip(sentences, sentence_labels))
    w_data = make_windowed_data(data, start=[5, 0], end=[6, 0], window_size=1)

    assert len(w_data) == sum(len(sentence) for sentence in sentences)

    assert w_data == [
        ([5, 0] + [1, 1] + [2, 0], 1,),
        ([1, 1] + [2, 0] + [3, 3], 2,),
        ([2, 0] + [3, 3] + [6, 0], 3,),
    ]


def do_test1(_):
    logger.info("Testing make_windowed_data")
    test_make_windowed_data()
    logger.info("Passed!")


def do_test2(args):
    logger.info("Testing implementation of WindowModel")
    config = Config()
    helper, train, dev, train_raw, dev_raw = load_and_preprocess_data(args)
    embeddings = load_embeddings(args, helper)
    config.embed_size = embeddings.shape[1]

    logger.info("Building model...")
    start = time.time()
    model = WindowModel(helper, config, embeddings)
    logger.info("took %.2f seconds", time.time() - start)

    model.fit(train, dev)
    logger.info("Model did not crash!")
    logger.info("Passed!")


def do_train(args):
    config = Config()
    helper, train, dev, train_raw, dev_raw = load_and_preprocess_data(args)
    embeddings = load_embeddings(args, helper)
    config.embed_size = embeddings.shape[1]
    helper.save(config.output_path)

    # Setup logging
    handler = logging.FileHandler(config.log_output)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
    logging.getLogger().addHandler(handler)

    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = WindowModel(helper, config, embeddings)
    model.to(device)

    # Train model
    logger.info("Training model...")
    start = time.time()
    model.fit(train, dev, device)
    logger.info(f"Training took {time.time() - start:.2f} seconds")

    # Save predictions
    output = model.output(None, dev_raw)  # Note: sess parameter not needed for PyTorch
    sentences, labels, predictions = zip(*output)
    predictions = [[LBLS[l] for l in preds] for preds in predictions]
    output = zip(sentences, labels, predictions)

    with open(model.config.conll_output, 'w') as f:
        write_conll(f, output)
    with open(model.config.eval_output, 'w') as f:
        for sentence, labels, predictions in output:
            print_sentence(f, sentence, labels, predictions)


def do_evaluate(args):
    config = Config(args.model_path)
    helper = ModelHelper.load(args.model_path)
    input_data = read_conll(args.data)
    embeddings = load_embeddings(args, helper)
    config.embed_size = embeddings.shape[1]

    logger.info("Building model...")
    start = time.time()
    model = WindowModel(helper, config, embeddings)
    logger.info("took %.2f seconds", time.time() - start)

    model.load_state_dict(torch.load(model.config.model_output))
    for sentence, labels, predictions in model.output(None, input_data):
        predictions = [LBLS[l] for l in predictions]
        print_sentence(args.output, sentence, labels, predictions)


def do_shell(args):
    config = Config(args.model_path)
    helper = ModelHelper.load(args.model_path)
    embeddings = load_embeddings(args, helper)
    config.embed_size = embeddings.shape[1]

    logger.info("Building model...")
    start = time.time()
    model = WindowModel(helper, config, embeddings)
    logger.info("took %.2f seconds", time.time() - start)

    model.load_state_dict(torch.load(model.config.model_output))
    model.eval()  # Set to evaluation mode

    print("""Welcome!
You can use this shell to explore the behavior of your model.
Please enter sentences with spaces between tokens, e.g.,
input> Germany 's representative to the European Union 's veterinary committee .""")

    while True:
        try:
            sentence = input("input> ")
            tokens = sentence.strip().split(" ")
            for sentence, _, predictions in model.output(None, [(tokens, ["O"] * len(tokens))]):
                predictions = [LBLS[l] for l in predictions]
                print_sentence(sys.stdout, sentence, [""] * len(tokens), predictions)
        except EOFError:
            print("Closing session.")
            break