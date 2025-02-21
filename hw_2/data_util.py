#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility functions to process data for PyTorch models.
"""
import os
import pickle
import logging
from collections import Counter
from typing import List, Tuple, Dict, Any

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from .util import read_conll, load_word_vector_mapping
from .defs import LBLS, NONE, LMAP, NUM, UNK, EMBED_SIZE

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

FDIM = 4
P_CASE = "CASE:"
CASES = ["aa", "AA", "Aa", "aA"]
START_TOKEN = "<s>"
END_TOKEN = "</s>"


class NERDataset(Dataset):
    """Custom Dataset for NER data"""

    def __init__(self, data, device='cuda'):
        self.data = data
        self.device = device

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sentence, labels = self.data[idx]
        # Convert to tensors
        sentence = torch.tensor(sentence, device=self.device)
        labels = torch.tensor(labels, device=self.device)
        return sentence, labels


def create_data_loader(data, batch_size, shuffle=True, device='cuda'):
    """Create a DataLoader for the NER data"""
    dataset = NERDataset(data, device)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def one_hot(n_classes: int, idx: int) -> torch.Tensor:
    """Create a one-hot vector"""
    one_hot_vec = torch.zeros(n_classes)
    one_hot_vec[idx] = 1
    return one_hot_vec


def casing(word: str) -> str:
    """Determine the casing of a word."""
    if len(word) == 0:
        return word
    if word.islower():
        return "aa"
    elif word.isupper():
        return "AA"
    elif word[0].isupper():
        return "Aa"
    else:
        return "aA"


def normalize(word: str) -> str:
    """Normalize words that are numbers or have casing."""
    if word.isdigit():
        return NUM
    else:
        return word.lower()


def featurize(embeddings: Dict[str, torch.Tensor], word: str) -> torch.Tensor:
    """Featurize a word given embeddings."""
    case = casing(word)
    word = normalize(word)
    case_mapping = {c: one_hot(FDIM, i) for i, c in enumerate(CASES)}
    wv = embeddings.get(word, embeddings[UNK])
    fv = case_mapping[case]
    return torch.cat((wv, fv))


def evaluate(model, data_loader):
    """Evaluate model performance"""
    device = next(model.parameters()).device
    model.eval()

    confusion = torch.zeros((len(LBLS), len(LBLS)), device=device)
    with torch.no_grad():
        for X, Y in data_loader:
            Y_pred = model(X)
            Y_pred = Y_pred.argmax(dim=-1)
            Y = Y.argmax(dim=-1)

            for y_true, y_pred in zip(Y.view(-1), Y_pred.view(-1)):
                confusion[y_true, y_pred] += 1

    # Calculate metrics
    tp = confusion.diag()
    fp = confusion.sum(dim=0) - tp
    fn = confusion.sum(dim=1) - tp

    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-10)

    return {
        'precision': precision.mean().item(),
        'recall': recall.mean().item(),
        'f1': f1.mean().item(),
        'confusion_matrix': confusion.cpu().numpy()
    }


class ModelHelper:
    """
    Helper class for preprocessing data and constructing embeddings.
    """

    def __init__(self, tok2id: Dict[str, int], max_length: int):
        self.tok2id = tok2id
        self.START = [tok2id[START_TOKEN], tok2id[P_CASE + "aa"]]
        self.END = [tok2id[END_TOKEN], tok2id[P_CASE + "aa"]]
        self.max_length = max_length

    def vectorize_example(self, sentence: List[str], labels: List[str] = None) -> Tuple[List[List[int]], List[int]]:
        sentence_ = [
            [self.tok2id.get(normalize(word), self.tok2id[UNK]),
             self.tok2id[P_CASE + casing(word)]]
            for word in sentence
        ]
        if labels:
            labels_ = [LBLS.index(l) for l in labels]
            return sentence_, labels_
        else:
            return sentence_, [LBLS.index(NONE) for _ in sentence]

    def vectorize(self, data: List[Tuple[List[str], List[str]]]) -> List[Tuple[List[List[int]], List[int]]]:
        return [self.vectorize_example(sentence, labels) for sentence, labels in data]

    @classmethod
    def build(cls, data: List[Tuple[List[str], List[str]]]) -> 'ModelHelper':
        # Build dictionary from data
        tok2id = build_dict(
            (normalize(word) for sentence, _ in data for word in sentence),
            offset=1,
            max_words=10000
        )
        tok2id.update(build_dict([P_CASE + c for c in CASES], offset=len(tok2id)))
        tok2id.update(build_dict([START_TOKEN, END_TOKEN, UNK], offset=len(tok2id)))

        max_length = max(len(sentence) for sentence, _ in data)
        logger.info("Built dictionary for %d features.", len(tok2id))

        return cls(tok2id, max_length)

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "features.pkl"), "wb") as f:
            pickle.dump([self.tok2id, self.max_length], f)

    @classmethod
    def load(cls, path: str) -> 'ModelHelper':
        features_path = os.path.join(path, "features.pkl")
        assert os.path.exists(path) and os.path.exists(features_path)
        with open(features_path, "rb") as f:
            tok2id, max_length = pickle.load(f)
        return cls(tok2id, max_length)


def load_and_preprocess_data(with_test=False):
    class Args:
        def __init__(self):
            self.data_train = "lev_nlp_course/hw_2/data/tiny.conll"
            self.data_dev = "lev_nlp_course/hw_2/data/tiny.conll"
            self.data_test = "lev_nlp_course/hw_2/data/test.masked"
            self.vocab = "lev_nlp_course/hw_2/data/vocab.txt"
            self.vectors = "lev_nlp_course/hw_2/data/wordVectors.txt"

        def parse_args(self):
            return self
    args = Args()
    args = args.parse_args()
    """Load and preprocess data for training"""
    logger.info("Loading training data...")
    train = read_conll(args.data_train)
    logger.info("Done. Read %d sentences", len(train))
    logger.info("Loading dev data...")
    dev = read_conll(args.data_dev)
    logger.info("Done. Read %d sentences", len(dev))

    helper = ModelHelper.build(train)

    train_data = helper.vectorize(train)
    dev_data = helper.vectorize(dev)

    if with_test:
        logger.info("Loading test data...")
        test = read_conll(args.data_test)
        logger.info("Done. Read %d sentences", len(test))
        test_data = helper.vectorize(test)
        return helper, train_data, dev_data, test_data, train, dev, test
    else:
        return helper, train_data, dev_data, train, dev


def load_embeddings(args, helper: ModelHelper) -> torch.Tensor:
    """Load word embeddings"""
    embeddings = torch.randn(len(helper.tok2id) + 1, EMBED_SIZE)
    embeddings[0] = 0.

    word_vectors = load_word_vector_mapping(args.vocab, args.vectors)
    for word, vec in word_vectors.items():
        word = normalize(word)
        if word in helper.tok2id:
            embeddings[helper.tok2id[word]] = torch.tensor(vec)

    logger.info("Initialized embeddings.")
    return embeddings


def build_dict(words: List[str], max_words: int = None, offset: int = 0) -> Dict[str, int]:
    """Build a dictionary from a list of words"""
    cnt = Counter(words)
    if max_words:
        words = cnt.most_common(max_words)
    else:
        words = cnt.most_common()
    return {word: offset + i for i, (word, _) in enumerate(words)}


def get_chunks(seq: List[int], default: int = LBLS.index(NONE)) -> List[Tuple[int, int, int]]:
    """Breaks input sequence into chunks"""
    chunks = []
    chunk_type, chunk_start = None, None
    for i, tok in enumerate(seq):
        # End of a chunk 1
        if tok == default and chunk_type is not None:
            chunk = (chunk_type, chunk_start, i)
            chunks.append(chunk)
            chunk_type, chunk_start = None, None
        # End of a chunk + start of a chunk!
        elif tok != default:
            if chunk_type is None:
                chunk_type, chunk_start = tok, i
            elif tok != chunk_type:
                chunk = (chunk_type, chunk_start, i)
                chunks.append(chunk)
                chunk_type, chunk_start = tok, i
    # end condition
    if chunk_type is not None:
        chunk = (chunk_type, chunk_start, len(seq))
        chunks.append(chunk)
    return chunks


def test_get_chunks():
    """Test the get_chunks function"""
    assert get_chunks([4, 4, 4, 0, 0, 4, 1, 2, 4, 3], 4) == [(0, 3, 5), (1, 6, 7), (2, 7, 8), (3, 9, 10)]