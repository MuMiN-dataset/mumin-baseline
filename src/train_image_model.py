'''Finetune a vision transformer model on the dataset'''

from transformers import (AutoModelForImageClassification,
                          AutoFeatureExtractor, AutoConfig, TrainingArguments)
from datasets import Dataset, load_metric, set_caching_enabled
from typing import Dict
import sys
import numpy as np
from mumin import MuminDataset
import os
from dotenv import load_dotenv
import gc

from trainer_with_class_weights import TrainerWithClassWeights


load_dotenv()


def train_image_model(model_id: str,
                      size: str,
                      random_split: bool = False,
                      num_epochs: int = 300,
                      **_) -> Dict[str, float]:
    '''Train a vision transformer model on the dataset.

    Args:
        model_id (str):
            The model id to use.
        size (str):
            The size of the dataset.
        random_split (bool, optional):
            Whether to use a random split. Defaults to False.
        num_epochs (int, optional):
            The number of epochs to train for. Defaults to 300.

    Returns:
        dict:
            The results of the training, with keys the names of the metrics and
            values the scores.
    '''
    # Load the dataset
    mumin_dataset = MuminDataset(os.environ['TWITTER_API_KEY'], size=size)
    mumin_dataset.compile()
    image_df = mumin_dataset.nodes['image']
    tweet2image_df = mumin_dataset.rels[('tweet', 'has_image', 'image')]
    tweet2claim_df = mumin_dataset.rels[('tweet', 'discusses', 'claim')]
    claim_df = mumin_dataset.nodes['claim']
    df = (image_df.merge(tweet2image_df.rename(columns=dict(src='tweet_idx',
                                                            tgt='image_idx')),
                         left_index=True,
                         right_on='image_idx')
                  .merge(tweet2claim_df.rename(columns=dict(src='tweet_idx',
                                                            tgt='claim_idx')),
                         on='tweet_idx')
                  .merge(claim_df, left_on='claim_idx', right_index=True))
    image_df = df[['pixels', 'label', 'train_mask', 'val_mask', 'test_mask']]

    # If we are performing a random split then split the dataset into a
    # 80/10/10 train/val/test split, with a fixed random seed
    if random_split:
        train_df = image_df.sample(frac=0.8, random_state=42)
        val_test_df = image_df.drop(train_df.index)
        val_df = val_test_df.sample(frac=0.5, random_state=42)
        test_df  = val_test_df.drop(val_df.index)

    # Otherwise, use the train/val/test split that is already in the dataset
    else:
        train_df = image_df.query('train_mask == True')
        val_df = image_df.query('val_mask == True')
        test_df = image_df.query('test_mask == True')

    # Convert dataset to dictionaries
    train_dict = dict(pixels=train_df.pixels.map(lambda x: x.tobytes()),
                      width=train_df.pixels.map(lambda x: x.shape[0]),
                      height=train_df.pixels.map(lambda x: x.shape[1]),
                      orig_label=train_df.label)
    val_dict = dict(pixels=val_df.pixels.map(lambda x: x.tobytes()),
                    width=val_df.pixels.map(lambda x: x.shape[0]),
                    height=val_df.pixels.map(lambda x: x.shape[1]),
                    orig_label=val_df.label)
    test_dict = dict(pixels=test_df.pixels.map(lambda x: x.tobytes()),
                     width=test_df.pixels.map(lambda x: x.shape[0]),
                     height=test_df.pixels.map(lambda x: x.shape[1]),
                     orig_label=test_df.label)

    # Convert the dataset to the HuggingFace format
    train = Dataset.from_dict(train_dict)
    val = Dataset.from_dict(val_dict)
    test = Dataset.from_dict(test_dict)

    # Garbage collection
    del df, image_df, claim_df, train_df, val_df, test_df, mumin_dataset
    gc.collect()

    # Load the feature extractor and model
    config_dict = dict(num_labels=2,
                       id2label={0: 'misinformation', 1: 'factual'},
                       label2id=dict(misinformation=0, factual=1),
                       hidden_dropout_prob=0.2,
                       attention_probs_dropout_prob=0.2,
                       classifier_dropout_prob=0.2)
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
        images = [
            np.moveaxis(np.frombuffer(buf, dtype='uint8')
                          .reshape(width, height, 3),
                        source=-1,
                        destination=0)
            for buf, width, height in zip(examples['pixels'],
                                          examples['width'],
                                          examples['height'])
        ]
        inputs = feat_extractor(images=images)
        examples['pixel_values'] = inputs['pixel_values']

        return examples

    train = train.map(preprocess, batched=True, batch_size=32)
    val = val.map(preprocess, batched=True, batch_size=32)
    test = test.map(preprocess, batched=True, batch_size=32)

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
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=num_epochs,
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
        gradient_accumulation_steps=1,
        metric_for_best_model='factual_f1',
    )

    # Initialise the Trainer
    trainer = TrainerWithClassWeights(model=model,
                                      args=training_args,
                                      train_dataset=train,
                                      eval_dataset=val,
                                      compute_metrics=compute_metrics,
                                      class_weights=[1., 20.])

    # Train the model
    trainer.train()

    # Evaluate the model
    results = dict(train=trainer.evaluate(train),
                   val=trainer.evaluate(val),
                   test=trainer.evaluate(test))

    return results
