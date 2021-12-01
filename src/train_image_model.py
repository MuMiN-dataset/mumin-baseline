'''Finetune a vision transformer model on the dataset'''

from transformers import (AutoModelForImageClassification,
                          AutoFeatureExtractor, AutoConfig, TrainingArguments,
                          Trainer)
from datasets import Dataset, load_metric
from typing import Dict
import sys
import pandas as pd
import numpy as np


def main(model_id: str) -> Dict[str, float]:
    '''Train a vision transformer model on the dataset.

    Args:
        model_id (str): The model id to use.

    Returns:
        dict:
            The results of the training, with keys the names of the metrics and
            values the scores.
    '''
    # Load the dataset
    image_df = pd.read_pickle('image_dump.pkl')
    train_df = image_df.query('train_mask == True')
    val_df = image_df.query('val_mask == True')

    # Convert the dataset to the HuggingFace format
    train = Dataset.from_dict(dict(pixels=train_df.claim.tolist(),
                                   orig_label=train_df.verdict.tolist()))
    val = Dataset.from_dict(dict(pixels=val_df.claim.tolist(),
                                 orig_label=val_df.verdict.tolist()))

    # Load the feature extractor and model
    config_dict = dict(num_labels=2,
                       id2label={0: 'misinformation', 1: 'factual'},
                       label2id=dict(misinformation=0, factual=1))
    config = AutoConfig.from_pretrained(model_id, **config_dict)
    model = AutoModelForImageClassification.from_pretrained(model_id,
                                                            config=config)
    feat_extractor = AutoFeatureExtractor.from_pretrained(model_id)

    # Preprocess the datasets
    def preprocess(examples: dict) -> dict:

        # Set up the labels
        labels = ['misinformation', 'factual']
        examples['labels'] = [labels.index(lbl)
                              for lbl in examples['orig_label']]

        # Extract the features
        images = [np.moveaxis(np.array(image, dtype=np.uint8),
                              source=-1,
                              destination=0)
                  for image in examples['pixels']]
        inputs = feat_extractor(images=images)
        examples['pixel_values'] = inputs['pixel_values']

        return examples

    train = train.map(preprocess, batched=True)
    val = val.map(preprocess, batched=True)

    # Set up compute_metrics function
    def compute_metrics(preds_and_labels: tuple) -> Dict[str, float]:
        metric = load_metric('f1')
        predictions, labels = preds_and_labels
        predictions = predictions.argmax(axis=-1)
        factual_results = metric.compute(predictions=predictions,
                                 references=labels)
        misinfo_results = metric.compute(predictions=1-predictions,
                                 references=1-labels)
        return dict(factual_f1=factual_results['f1'],
                    misinfo_f1=misinfo_results['f1'])

    # Set up the training arguments
    training_args = TrainingArguments(
        output_dir='models',
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=1000,
        evaluation_strategy='steps',
        logging_strategy='steps',
        save_strategy='steps',
        eval_steps=100,
        logging_steps=100,
        save_steps=100,
        report_to='none',
        save_total_limit=1,
        learning_rate=2e-5,
        warmup_ratio=0.01,  # 10 epochs
        gradient_accumulation_steps=4,
        metric_for_best_model='factual_f1',
        load_best_model_at_end=True,
    )

    # Initialise the Trainer
    trainer = Trainer(model=model,
                      args=training_args,
                      train_dataset=train,
                      eval_dataset=val,
                      tokenizer=tokenizer,
                      compute_metrics=compute_metrics)

    # Train the model
    trainer.train()

    # Evaluate the model
    results = trainer.evaluate()

    return results


if __name__ == '__main__':
    patch_model_id = 'google/vit-base-patch16-224-in21k'
    model_id = sys.argv[-1] if len(sys.argv) > 1 else patch_model_id
    main(model_id)