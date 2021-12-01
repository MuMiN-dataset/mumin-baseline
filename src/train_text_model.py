'''Finetune a transformer model on the dataset'''

from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          AutoConfig, TrainingArguments, Trainer)
from datasets import Dataset
from typing import Dict
import sys
import pandas as pd


def main(model_id: str) -> Dict[str, float]:
    '''Train a transformer model on the dataset.

    Args:
        model_id (str): The model id to use.

    Returns:
        dict:
            The results of the training, with keys the names of the metrics and
            values the scores.
    '''
    # Load the dataset
    claim_df = pd.read_pickle('claim_dump.pkl')
    train_df = claim_df.query('train_mask == True')
    val_df = claim_df.query('val_mask == True')

    # Convert the dataset to the HuggingFace format
    train = Dataset.from_dict(dict(text=train_df.claim.tolist(),
                                   labels=train_df.verdict.tolist()))
    val = Dataset.from_dict(dict(text=val_df.claim.tolist(),
                                 labels=val_df.verdict.tolist()))

    # Load the tokenizer and model
    config_dict = dict(num_labels=2,
                       id2label={0: 'misinformation', 1: 'factual'},
                       label2id=dict(misinformation=0, factual=1))
    config = AutoConfig.from_pretrained(model_id, **config_dict)
    model = AutoModelForSequenceClassification.from_pretrained(model_id,
                                                               config=config)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Set up the training arguments
    training_args = TrainingArguments(
        output_dir='models',
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=100,
        evaluation_strategy='epoch',
        logging_strategy='epoch',
        save_strategy='epoch',
        report_to='none',
        save_total_limit=1,
        learning_rate=2e-5,
        warmup_steps=len(train),
        gradient_accumulation_steps=4,
        metric_for_best_model='loss',
        load_best_model_at_end=True,
    )

    # Set up early stopping callback
    early_stopping = EarlyStoppingCallback(early_stopping_patience=2)

    # Initialise the Trainer
    trainer = Trainer(model=model,
                      args=training_args,
                      train_dataset=train,
                      eval_dataset=val,
                      tokenizer=tokenizer,
                      callbacks=[early_stopping])

    # Train the model
    trainer.train()

    # Evaluate the model
    results = trainer.evaluate()

    return results


if __name__ == '__main__':
    model_id = sys.argv[1]
    main(model_id)
