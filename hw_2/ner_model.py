#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A model for named entity recognition using PyTorch.
"""
import logging
import torch
import torch.nn as nn
from torch.optim import AdamW
from typing import List, Tuple, Set
from .util import ConfusionMatrix, Progbar, minibatches
from .data_util import get_chunks
from .defs import LBLS

logger = logging.getLogger("hw3")
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)


class NERModel(nn.Module):
    """
    Implements special functionality for NER models.
    """

    def __init__(self, helper, config, report=None):
        super(NERModel, self).__init__()
        self.helper = helper
        self.config = config
        self.report = report
        self.optimizer = AdamW(self.parameters(), lr=self.config.lr)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def preprocess_sequence_data(self, examples):
        """Preprocess sequence data for the model.

        Args:
            examples: A list of vectorized input/output sequences.
        Returns:
            A new list of vectorized input/output pairs appropriate for the model.
        """
        raise NotImplementedError("Each Model must re-implement this method.")

    def consolidate_predictions(self, data_raw, data, preds):
        """
        Convert a sequence of predictions according to the batching
        process back into the original sequence.
        """
        raise NotImplementedError("Each Model must re-implement this method.")

    def evaluate(self, examples, examples_raw):
        """Evaluates model performance on @examples.

        This function uses the model to predict labels for @examples and constructs a confusion matrix.

        Args:
            examples: A list of vectorized input/output pairs.
            examples_raw: A list of the original input/output sequence pairs.
        Returns:
            The F1 score for predicting tokens as named entities.
        """
        token_cm = ConfusionMatrix(labels=LBLS)

        self.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            correct_preds, total_correct, total_preds = 0., 0., 0.
            for _, labels, labels_ in self.output(examples_raw, examples):
                for l, l_ in zip(labels, labels_):
                    token_cm.update(l, l_)
                gold = set(get_chunks(labels))
                pred = set(get_chunks(labels_))
                correct_preds += len(gold.intersection(pred))
                total_preds += len(pred)
                total_correct += len(gold)

        p = correct_preds / total_preds if correct_preds > 0 else 0
        r = correct_preds / total_correct if correct_preds > 0 else 0
        f1 = 2 * p * r / (p + r) if correct_preds > 0 else 0
        return token_cm, (p, r, f1)

    def output(self, inputs_raw, inputs=None):
        """
        Reports the output of the model on examples (uses helper to featurize each example).
        """
        if inputs is None:
            inputs = self.preprocess_sequence_data(self.helper.vectorize(inputs_raw))

        preds = []
        prog = Progbar(target=1 + int(len(inputs) / self.config.batch_size))

        self.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            for i, batch in enumerate(minibatches(inputs, self.config.batch_size, shuffle=False)):
                # Ignore predict
                batch = batch[:1] + batch[2:]
                preds_ = self.predict_on_batch(*batch)
                preds += list(preds_)
                prog.update(i + 1, [])

        return self.consolidate_predictions(inputs_raw, inputs, preds)

    def fit(self, train_examples_raw, dev_set_raw, optimizer, scheduler=None):
        """
        Fit the model on the training data and evaluate on dev set.

        Args:
            train_examples_raw: Raw training examples
            dev_set_raw: Raw development set examples
            optimizer: PyTorch optimizer
            scheduler: Optional learning rate scheduler
        """
        best_score = 0.
        best_state = None

        train_examples = self.preprocess_sequence_data(train_examples_raw)
        dev_set = self.preprocess_sequence_data(dev_set_raw)

        for epoch in range(self.config.n_epochs):
            logger.info("Epoch %d out of %d", epoch + 1, self.config.n_epochs)
            prog = Progbar(target=1 + int(len(train_examples) / self.config.batch_size))

            # Set model to training mode
            self.train()

            # Training loop
            for i, batch in enumerate(minibatches(train_examples, self.config.batch_size)):
                optimizer.zero_grad()
                loss = self.train_on_batch(*batch)
                loss.backward()
                optimizer.step()
                prog.update(i + 1, [("loss", loss.item())])

            if scheduler:
                scheduler.step()

            # Evaluation on dev set
            logger.info("Evaluating on development data")
            token_cm, entity_scores = self.evaluate(dev_set, dev_set_raw)
            logger.debug("Token-level confusion matrix:\n" + token_cm.as_table())
            logger.debug("Token-level scores:\n" + token_cm.summary())
            logger.info("Entity level P/R/F1: %.2f/%.2f/%.2f", *entity_scores)

            score = entity_scores[-1]

            if score > best_score:
                best_score = score
                best_state = self.state_dict()
                logger.info("New best score! Saving model")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'score': best_score,
                }, self.config.model_output)

            print("")
            if self.report:
                self.report.log_epoch()
                self.report.save()

        # Load best model
        if best_state is not None:
            self.load_state_dict(best_state)

        return best_score

    def predict_on_batch(self, *args):
        """
        Make predictions for the given batch of inputs.
        """
        raise NotImplementedError("Each Model must re-implement this method.")

    def train_on_batch(self, *args):
        """
        Perform one step of gradient descent on the provided batch of data.
        """
        raise NotImplementedError("Each Model must re-implement this method.")