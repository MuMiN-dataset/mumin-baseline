'''Custom Trainer class with class weights'''

from transformers import Trainer
from typing import List
import torch.nn as nn
import torch


class TrainerWithClassWeights(Trainer):
    def __init__(self, class_weights: List[float], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = (torch.tensor(class_weights)
                                   .to(self.model.device))

    def compute_loss(self, model, inputs, return_outputs=False):
        '''Compute the loss for a batch of inputs.

        Args:
            model (PyTorch model):
                The model to train.
            inputs (PyTorch tensor):
                Tuple containing the inputs.
            return_outputs (bool, optional):
                Whether or not to return the outputs. Defaults to False.

        Returns:
            tuple or float:
                If return_outputs is True, returns a tuple containing the loss
                and the outputs. Otherwise, returns the loss.
        '''
        # Fetch the labels
        labels = inputs.get("labels")

        # Forward pass
        outputs = model(**inputs)
        logits = outputs.get('logits')

        # Reshape the logits
        logits = logits.view(-1, self.model.config.num_labels)

        # Compute custom loss
        loss_fct = nn.CrossEntropyLoss(weight=self.class_weights)
        loss = loss_fct(logits, labels.view(-1))

        return (loss, outputs) if return_outputs else loss

