import torch
import torch.nn as nn


class Model(nn.Module):
    """Abstracts a PyTorch model for a learning task.

    This serves as a base class for implementing various learning algorithms.
    By inheriting from nn.Module, we get PyTorch's automatic differentiation
    and GPU support built in.
    """

    def __init__(self):
        super(Model, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def build(self):
        """Initialize the model architecture.

        Note: Unlike TensorFlow, PyTorch models typically define their architecture
        in __init__ rather than needing a separate build step. This method is kept
        for compatibility with existing code structure but should generally be empty
        or removed in favor of proper __init__ implementation.
        """
        pass

    def add_prediction_op(self, x):
        """Implements the core of the model that transforms input data into predictions.

        This method should be implemented in the forward() method of subclasses.

        Args:
            x: Input tensor of shape (batch_size, n_features)
        Returns:
            pred: A tensor of shape (batch_size, n_classes)
        """
        raise NotImplementedError("Each Model must implement forward() method.")

    def add_loss_op(self, pred, labels):
        """Adds the loss function.

        Args:
            pred: A tensor of shape (batch_size, n_classes)
            labels: A tensor of shape (batch_size, n_classes)
        Returns:
            loss: A scalar tensor
        """
        raise NotImplementedError("Each Model must re-implement this method.")

    def forward(self, x):
        """Forward pass of the model.

        Args:
            x: Input tensor of shape (batch_size, n_features)
        Returns:
            pred: A tensor of shape (batch_size, n_classes)
        """
        return self.add_prediction_op(x)

    def train_on_batch(self, inputs_batch, labels_batch):
        """Perform one step of gradient descent on the provided batch of data.

        Args:
            inputs_batch: tensor of shape (n_samples, n_features)
            labels_batch: tensor of shape (n_samples, n_classes)
        Returns:
            loss: loss over the batch (a scalar)
        """
        self.train()  # Set model to training mode
        inputs = torch.tensor(inputs_batch, device=self.device)
        labels = torch.tensor(labels_batch, device=self.device)

        # Forward pass
        predictions = self.forward(inputs)
        loss = self.add_loss_op(predictions, labels)

        return loss

    def predict_on_batch(self, inputs_batch):
        """Make predictions for the provided batch of data

        Args:
            inputs_batch: tensor or numpy array of shape (n_samples, n_features)
        Returns:
            predictions: tensor of shape (n_samples, n_classes)
        """
        self.eval()  # Set model to evaluation mode
        with torch.no_grad():
            inputs = torch.tensor(inputs_batch, device=self.device)
            predictions = self.forward(inputs)
        return predictions

    # The following TensorFlow-specific methods are removed as they're not needed in PyTorch:
    # - add_placeholders()
    # - create_feed_dict()
    # - add_training_op()