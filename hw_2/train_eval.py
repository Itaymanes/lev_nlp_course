import torch
import torch.nn as nn
import numpy as np
from torch import optim
import logging
from sklearn.metrics import f1_score
from typing import List, Tuple, Any


def compute_f1(predictions: np.ndarray, labels: np.ndarray, n_classes: int) -> Tuple[float, List[float]]:
    """
    Compute overall and per-class F1 scores.

    Args:
        predictions: Predicted labels
        labels: True labels
        n_classes: Number of classes

    Returns:
        Tuple of (macro_f1, list of per-class f1 scores)
    """
    # Compute macro F1 score
    macro_f1 = f1_score(labels, predictions, average='macro')

    # Compute per-class F1 scores
    per_class_f1 = f1_score(labels, predictions, average=None)

    return macro_f1, per_class_f1.tolist()


def evaluate_model(model: nn.Module, data: List[Tuple], batch_size: int, device: str) -> Tuple[
    float, float, List[float]]:
    """
    Evaluate model on given data.

    Returns:
        Tuple of (accuracy, macro_f1, per_class_f1_scores)
    """
    model.eval()
    all_predictions = []
    all_labels = []
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_start in range(0, len(data), batch_size):
            batch_end = batch_start + batch_size
            batch_inputs = torch.LongTensor([data[i][0] for i in range(batch_start, min(batch_end, len(data)))]).to(
                device)
            batch_labels = torch.LongTensor([data[i][1] for i in range(batch_start, min(batch_end, len(data)))]).to(
                device)

            outputs = model(batch_inputs)
            predictions = torch.argmax(outputs, dim=1)

            total += batch_labels.size(0)
            correct += (predictions == batch_labels).sum().item()

            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())

    accuracy = correct / total
    macro_f1, per_class_f1 = compute_f1(np.array(all_predictions), np.array(all_labels), model.config.n_classes)

    return accuracy, macro_f1, per_class_f1


def train_model(model: nn.Module,
                train_data: List[Tuple],
                dev_data: List[Tuple],
                config: Any,
                logger: logging.Logger) -> Tuple[List[float], List[float], List[float], List[float]]:
    """
    Train the model and return training history.

    Args:
        model: The neural network model
        train_data: List of (input, label) tuples for training
        dev_data: List of (input, label) tuples for validation
        config: Configuration object with training parameters
        logger: Logger object for printing progress

    Returns:
        Tuple of (train_losses, train_f1s, dev_accs, dev_f1s)
    """
    model.to(model.device)
    model.criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.lr)

    best_dev_f1 = 0
    train_losses = []
    train_f1s = []
    dev_accs = []
    dev_f1s = []

    for epoch in range(config.n_epochs):
        # Training
        model.train()
        train_loss = 0
        train_batches = 0
        epoch_predictions = []
        epoch_labels = []

        # Create batches with random permutation
        indices = np.random.permutation(len(train_data))
        for batch_start in range(0, len(indices), config.batch_size):
            batch_indices = indices[batch_start:batch_start + config.batch_size]
            batch_inputs = torch.LongTensor([train_data[i][0] for i in batch_indices]).to(model.device)
            batch_labels = torch.LongTensor([train_data[i][1] for i in batch_indices]).to(model.device)

            # Training step
            optimizer.zero_grad()
            logits = model(batch_inputs)

            # todo: check
            batch_labels = batch_labels.view(-1)  # Reshape to [batch_size * max_length]

            loss_batch = model.criterion(logits, batch_labels)

            loss_batch.backward()
            optimizer.step()

            # Collect predictions for F1 computation
            predictions = torch.argmax(logits, dim=1)
            epoch_predictions.extend(predictions.cpu().numpy())
            epoch_labels.extend(batch_labels.cpu().numpy())

            train_loss += loss_batch.item()
            train_batches += 1

        # Compute training metrics
        avg_train_loss = train_loss / train_batches
        train_macro_f1, train_per_class_f1 = compute_f1(
            np.array(epoch_predictions),
            np.array(epoch_labels),
            config.n_classes
        )

        # Evaluation
        dev_acc, dev_macro_f1, dev_per_class_f1 = evaluate_model(
            model, dev_data, config.batch_size, model.device
        )

        # Store metrics
        train_losses.append(avg_train_loss)
        train_f1s.append(train_macro_f1)
        dev_accs.append(dev_acc)
        dev_f1s.append(dev_macro_f1)

        # Logging
        logger.info(f"Epoch {epoch + 1}/{config.n_epochs}")
        logger.info(f"Average training loss: {avg_train_loss:.4f}")
        logger.info(f"Training Macro F1: {train_macro_f1:.4f}")
        logger.info(f"Training per-class F1: {[f'{f1:.4f}' for f1 in train_per_class_f1]}")
        logger.info(f"Dev accuracy: {dev_acc:.4f}")
        logger.info(f"Dev Macro F1: {dev_macro_f1:.4f}")
        logger.info(f"Dev per-class F1: {[f'{f1:.4f}' for f1 in dev_per_class_f1]}")

        # Save best model based on dev F1 score
        if dev_macro_f1 > best_dev_f1:
            best_dev_f1 = dev_macro_f1
            torch.save(model.state_dict(), config.model_output)
            logger.info("Saved new best model")

        logger.info("----------------------")

    return train_losses, train_f1s, dev_accs, dev_f1s

def train_model_rnn(model: nn.Module,
                train_data: List[Tuple],
                dev_data: List[Tuple],
                config: Any,
                logger: logging.Logger) -> Tuple[List[float], List[float], List[float], List[float]]:
    """
    Train the model and return training history.

    Args:
        model: The neural network model
        train_data: List of (input, label) tuples for training
        dev_data: List of (input, label) tuples for validation
        config: Configuration object with training parameters
        logger: Logger object for printing progress

    Returns:
        Tuple of (train_losses, train_f1s, dev_accs, dev_f1s)
    """
    model.to(model.device)
    model.criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.lr)

    best_dev_f1 = 0
    train_losses = []
    train_f1s = []
    dev_accs = []
    dev_f1s = []

    for epoch in range(config.n_epochs):
        # Training
        model.train()
        train_loss = 0
        train_batches = 0
        epoch_predictions = []
        epoch_labels = []

        # Create batches with random permutation
        indices = np.random.permutation(len(train_data))
        for batch_start in range(0, len(indices), config.batch_size):
            batch_indices = indices[batch_start:batch_start + config.batch_size]
            batch_inputs = torch.LongTensor([train_data[i][0] for i in batch_indices]).to(model.device)
            batch_labels = torch.LongTensor([train_data[i][1] for i in batch_indices]).to(model.device)

            # Training step
            optimizer.zero_grad()
            logits = model(batch_inputs)

            # todo: check
            batch_labels = batch_labels.view(-1)  # Reshape to [batch_size * max_length]
            loss_batch = model.criterion(logits, batch_labels)
            loss_batch.backward()
            optimizer.step()

            # Reshape logits back to [batch_size, max_length, n_classes] for predictions
            logits = logits.view(batch_inputs.size(0), batch_inputs.size(1), -1)
            predictions = torch.argmax(logits, dim=2)  # [batch_size, max_length]

            # Now predictions and batch_labels should have compatible shapes for F1 computation
            epoch_predictions.extend(predictions.cpu().numpy().flatten())
            epoch_labels.extend(batch_labels.cpu().numpy().flatten())

            # # Collect predictions for F1 computation
            # predictions = torch.argmax(logits, dim=1)
            # epoch_predictions.extend(predictions.cpu().numpy())
            # epoch_labels.extend(batch_labels.cpu().numpy())

            train_loss += loss_batch.item()
            train_batches += 1

        # Compute training metrics
        avg_train_loss = train_loss / train_batches
        print(f"avg_train_loss: {avg_train_loss}")
        train_macro_f1, train_per_class_f1 = compute_f1(
            np.array(epoch_predictions),
            np.array(epoch_labels),
            config.n_classes
        )
        print(f"train_macro_f1: {train_macro_f1}")
        print(f"train_per_class_f1: {train_per_class_f1}")
        # Evaluation
        dev_acc, dev_macro_f1, dev_per_class_f1 = evaluate_model(
            model, dev_data, config.batch_size, model.device
        )

        # Store metrics
        train_losses.append(avg_train_loss)
        train_f1s.append(train_macro_f1)
        dev_accs.append(dev_acc)
        dev_f1s.append(dev_macro_f1)

        # Logging
        logger.info(f"Epoch {epoch + 1}/{config.n_epochs}")
        logger.info(f"Average training loss: {avg_train_loss:.4f}")
        logger.info(f"Training Macro F1: {train_macro_f1:.4f}")
        logger.info(f"Training per-class F1: {[f'{f1:.4f}' for f1 in train_per_class_f1]}")
        logger.info(f"Dev accuracy: {dev_acc:.4f}")
        logger.info(f"Dev Macro F1: {dev_macro_f1:.4f}")
        logger.info(f"Dev per-class F1: {[f'{f1:.4f}' for f1 in dev_per_class_f1]}")

        # Save best model based on dev F1 score
        if dev_macro_f1 > best_dev_f1:
            best_dev_f1 = dev_macro_f1
            torch.save(model.state_dict(), config.model_output)
            logger.info("Saved new best model")

        logger.info("----------------------")

    return train_losses, train_f1s, dev_accs, dev_f1s