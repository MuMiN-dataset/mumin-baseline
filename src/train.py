'''General script that calls the training scripts of various baselines'''

import click
from train_claim_model import train_claim_model
from train_tweet_model import train_tweet_model
from train_image_model import train_image_model
from train_graph_model import train_graph_model
import logging


# Set up logging
fmt = '%(asctime)s [%(levelname)s] %(message)s'
logging.basicConfig(level=logging.INFO, format=fmt)
logger = logging.getLogger(__name__)


@click.command()
@click.option('--model_type',
              type=click.Choice(['claim', 'tweet', 'image', 'graph']),
              help='The type of model to train.')
@click.option('--size',
              type=click.Choice(['small', 'medium', 'large']),
              help='The size of the MuMiN dataset to use.')
@click.option('--task',
              type=click.Choice(['claim', 'tweet']),
              help=('What task to finetune the model for. Only used if '
                    '`model_type`==`graph`.'))
@click.option('--text_model_id',
              default='sentence-transformers/LaBSE',
              type=str,
              help=('The HuggingFace model ID of the text model to finetune. '
                    'Only relevant if `model_type` is \'claim\' or '
                    '\'tweet\'.'))
@click.option('--image_model_id',
              default='google/vit-base-patch16-224-in21k',
              type=str,
              help=('The HuggingFace model ID of the image model to finetune. '
                    'Only relevant if `model_type` is \'image\'.'))
@click.option('--frozen',
              is_flag=True,
              show_default=True,
              help=('Whether to freeze the weights of the pretrained model. '
                    'Only relevant if `model_type` is \'claim\' or '
                    '\'tweet\'.'))
@click.option('--random_split',
              is_flag=True,
              show_default=True,
              help=('Whether the model should be benchmarked on a random '
                    'split of the data, as opposed to splits based on the '
                    'claim clusters.'))
@click.option('--num_epochs',
              default=300,
              show_default=True,
              type=int,
              help='The amount of epochs to train for. ')
def main(model_type: str, **kwargs):
    '''Benchmark models on the MuMiN dataset.'''
    if model_type == 'claim':
        kwargs['model_id'] = kwargs.pop('text_model_id')
        scores = train_claim_model(**kwargs)
    elif model_type == 'tweet':
        kwargs['model_id'] = kwargs.pop('text_model_id')
        scores = train_tweet_model(**kwargs)
    elif model_type == 'image':
        kwargs['model_id'] = kwargs.pop('image_model_id')
        scores = train_image_model(**kwargs)
    elif model_type == 'graph':
        scores = train_graph_model(**kwargs)
    else:
        raise ValueError(f'Invalid model type: {model_type}')

    # Report statistics
    log = 'Final evaluation\n'
    for split, dct in scores.items():
        for statistic, value in dct.items():
            statistic = split + '_' + statistic.replace('eval_', '')
            log += f'> {statistic}: {value}\n'
    logger.info(log)


if __name__ == '__main__':
    main()
