'''Finetune a transformer model on the claim part of the dataset'''

from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          AutoConfig, TrainingArguments)
from datasets import Dataset, load_metric
from typing import Dict
import pandas as pd
from dotenv import load_dotenv

from trainer_with_class_weights import TrainerWithClassWeights


load_dotenv()


def train_claim_model(model_id: str,
                      size: str,
                      frozen: bool = False,
                      random_split: bool = False,
                      num_epochs: int = 300,
                      **_) -> Dict[str, Dict[str, float]]:
    '''Train a transformer model on the dataset.

    Args:
        model_id (str):
            The model id to use.
        size (str):
            The size of the dataset to use.
        frozen (bool, optional):
            Whether to freeze the model weights. Defaults to False.
        random_split (bool, optional):
            Whether to use a random split of the dataset. Defaults to False.
        num_epochs (int, optional):
            The number of epochs to train for. Defaults to 300.

    Returns:
        dict:
            The results of the training, with keys 'train', 'val' and 'split',
            with dictionaries with the split scores as values.
    '''
    # Load the dataset
    claim_df = pd.read_pickle(f'claim_dump_{size}.pkl.xz')

    # If we are performing a random split then split the dataset into a
    # 80/10/10 train/val/test split, with a fixed random seed
    if random_split:
        train_df = claim_df.sample(frac=0.8, random_state=42)
        val_test_df = claim_df.drop(train_df.index)
        val_df = val_test_df.sample(frac=0.5, random_state=42)
        test_df = val_test_df.drop(val_df.index)

    # Otherwise, use the train/val/test split that is already in the dataset
    else:
        train_df = claim_df.query('train_mask == True')
        val_df = claim_df.query('val_mask == True')
        test_df = claim_df.query('test_mask == True')

    # Convert the dataset to the HuggingFace format
    train = Dataset.from_dict(dict(text=train_df.claim.tolist(),
                                   orig_label=train_df.verdict.tolist()))
    val = Dataset.from_dict(dict(text=val_df.claim.tolist(),
                                 orig_label=val_df.verdict.tolist()))
    test = Dataset.from_dict(dict(text=test_df.claim.tolist(),
                                 orig_label=test_df.verdict.tolist()))

    # Load the tokenizer and model
    config_dict = dict(num_labels=2,
                       id2label={0: 'misinformation', 1: 'factual'},
                       label2id=dict(misinformation=0, factual=1),
                       hidden_dropout_prob=0.2,
                       attention_probs_dropout_prob=0.2,
                       classifier_dropout_prob=0.5)
    config = AutoConfig.from_pretrained(model_id, **config_dict)
    model = AutoModelForSequenceClassification.from_pretrained(model_id,
                                                               config=config)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Freeze layers if required
    if frozen:
        for param in model.bert.parameters():
            param.requires_grad = False

    # Preprocess the datasets
    def preprocess(examples: dict) -> dict:
        labels = ['misinformation', 'factual']
        examples['labels'] = [labels.index(lbl)
                              for lbl in examples['orig_label']]
        examples = tokenizer(examples['text'],
                              truncation=True,
                              padding=True,
                              max_length=512)
        return examples
    train = train.map(preprocess, batched=True)
    val = val.map(preprocess, batched=True)
    test = test.map(preprocess, batched=True)

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
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
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
        warmup_ratio=0.01,
        gradient_accumulation_steps=32,
    )

    # Initialise the Trainer
    trainer = TrainerWithClassWeights(model=model,
                                      args=training_args,
                                      train_dataset=train,
                                      eval_dataset=val,
                                      tokenizer=tokenizer,
                                      compute_metrics=compute_metrics,
                                      class_weights=[1., 20.])

    # Train the model
    trainer.train()

    # Evaluate the model
    results = dict(train=trainer.evaluate(train),
                   val=trainer.evaluate(val),
                   test=trainer.evaluate(test))

    return results
