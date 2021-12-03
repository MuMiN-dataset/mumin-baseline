'''Finetune a transformer model on the dataset'''

from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          AutoConfig, TrainingArguments)
from datasets import Dataset, load_metric
from typing import Dict
import sys
import os
from dotenv import load_dotenv

from trainer_with_class_weights import TrainerWithClassWeights


load_dotenv()


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
    mumin_dataset = MuminDataset(os.environ['TWITTER_API_KEY'])
    mumin_dataset.compile()
    claim_df = mumin_dataset.nodes['claim']
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
                       hidden_dropout_prob=0.5,
                       attention_probs_dropout_prob=0.5,
                       classifier_dropout_prob=0.5)
    config = AutoConfig.from_pretrained(model_id, **config_dict)
    model = AutoModelForSequenceClassification.from_pretrained(model_id,
                                                               config=config)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

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


if __name__ == '__main__':
    labse_model = 'sentence-transformers/LaBSE'
    model_id = sys.argv[-1] if len(sys.argv) > 1 else labse_model
    results = main(model_id)
    print(results)
