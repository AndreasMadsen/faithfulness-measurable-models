import json
import os
import pathlib
import argparse
from timeit import default_timer as timer

from tqdm import tqdm
import numpy as np
import tensorflow as tf

from ecoroar.util import generate_experiment_id, model_name_to_huggingface_repo, default_jit_compile, default_max_epochs
from ecoroar.dataset import datasets
from ecoroar.tokenizer import HuggingfaceTokenizer
from ecoroar.model import huggingface_model_from_local
from ecoroar.transform import BucketedPaddedBatch, ExplainerMasking
from ecoroar.explain import explainers

parser = argparse.ArgumentParser()
parser.add_argument('--persistent-dir',
                    action='store',
                    default=pathlib.Path(__file__).absolute().parent.parent,
                    type=pathlib.Path,
                    help='Directory where all persistent data will be stored')
parser.add_argument('--seed',
                    action='store',
                    default=0,
                    type=int,
                    help='Random seed')
parser.add_argument('--max-epochs',
                    action='store',
                    default=None,
                    type=int,
                    help='The max number of epochs to use')
parser.add_argument('--batch-size',
                    action='store',
                    default=16,
                    type=int,
                    help='The batch size to use for training and evaluation')
parser.add_argument('--model',
                    action='store',
                    default='roberta-sb',
                    type=str,
                    help='Model name')
parser.add_argument('--huggingface-repo',
                    action='store',
                    default=None,
                    type=str,
                    help='Valid huggingface repo')
parser.add_argument('--dataset',
                    action='store',
                    default='IMDB',
                    choices=datasets.keys(),
                    type=str,
                    help='The dataset to fine-tune on')
parser.add_argument('--deterministic',
                    action='store_true',
                    help='Use determinstic computations')
parser.add_argument('--jit-compile',
                    action=argparse.BooleanOptionalAction,
                    default=None,
                    help='Use XLA JIT complication')
parser.add_argument('--precision',
                    action='store',
                    default='mixed_float16',
                    choices=['mixed_float16', 'mixed_bfloat16', 'float32'],
                    help='Set the precision policy. mixed_bfloat16 only works on Ampere (A100) and better.')
parser.add_argument('--max-masking-ratio',
                    action='store',
                    default=0,
                    type=int,
                    help='The maximum masking ratio (percentage integer) to apply on the training dataset')
parser.add_argument('--masking-strategy',
                    default='uni',
                    choices=['uni', 'half-det', 'half-ran'],
                    type=str,
                    help='The masking strategy to use for masking during fune-tuning')
parser.add_argument('--explainer',
                    default='uni',
                    choices=explainers.keys(),
                    type=str,
                    help='The masking strategy to use for masking during fune-tuning')

if __name__ == '__main__':
    durations = {}
    setup_time_start = timer()

    # Parse arguments
    args = parser.parse_args()
    if args.huggingface_repo is None:
        args.huggingface_repo = model_name_to_huggingface_repo(args.model)
    args.jit_compile = default_jit_compile(args)
    args.max_epochs = default_max_epochs(args)

    # Generate job id
    experiment_id = generate_experiment_id(
        'faithfulness',
        model=args.model, dataset=args.dataset,
        seed=args.seed, max_epochs=args.max_epochs,
        max_masking_ratio=args.max_masking_ratio, masking_strategy=args.masking_strategy
    )

    # Print configuration
    print(f'Configuration [{experiment_id}]:')
    print('  Seed:', args.seed)
    print('  Model:', args.model)
    print('  Dataset:', args.dataset)
    print('  Huggingface Repo:', args.huggingface_repo)
    print('  Max masking ratio:', args.max_masking_ratio)
    print('  Masking strategy:', args.masking_strategy)
    print('  Explainer:', args.explainer)
    print('')
    print('  Batch size:', args.batch_size)
    print('  Max epochs:', args.max_epochs)
    print('')
    print('  JIT compile:', args.jit_compile)
    print('  Deterministic:', args.deterministic)
    print('  Precision:', args.precision)
    print('')

    # Set global configuration options
    if args.deterministic:
        tf.config.experimental.enable_op_determinism()
    tf.keras.utils.set_random_seed(args.seed)
    tf.keras.mixed_precision.set_global_policy(args.precision)

    # Initialize tokenizer, dataset, and model
    tokenizer = HuggingfaceTokenizer(args.huggingface_repo, persistent_dir=args.persistent_dir)
    dataset = datasets[args.dataset](persistent_dir=args.persistent_dir, seed=args.seed)
    model = huggingface_model_from_local(args.persistent_dir / 'checkpoints' / generate_experiment_id(
        'masking',
        model=args.model, dataset=args.dataset,
        seed=args.seed, max_epochs=args.max_epochs,
        max_masking_ratio=args.max_masking_ratio, masking_strategy=args.masking_strategy
    ))
    explainer = explainers[args.explainer](tokenizer, model, run_eagerly=False, jit_compile=args.jit_compile)
    masker = ExplainerMasking(explainer, tokenizer)

    # Load datasets
    dataset_test = dataset.test(tokenizer)

    # Setup batching routine
    if args.jit_compile:
        batcher = BucketedPaddedBatch([dataset_test], batch_size=args.batch_size)
    else:
        batcher = lambda batch_size, padding_values, num_parallel_calls: \
            lambda dataset: dataset.padded_batch(batch_size, padding_values=padding_values)

    # Configure model
    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, name='cross_entropy'),
        metrics=dataset.metrics(),
        run_eagerly=False,
        jit_compile=args.jit_compile
    )
    durations['setup'] = timer() - setup_time_start
    results_time_start = timer()

    # Compute faithfulness curve
    dataset_test_masked = dataset_test \
        .apply(batcher(args.batch_size,
                        padding_values=(tokenizer.padding_values, None),
                        num_parallel_calls=tf.data.AUTOTUNE)) \
        .prefetch(tf.data.AUTOTUNE)

    # Evalute test performance at different masking ratios
    results = []
    for masking_ratio in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:

        # Create masked dataset. This is cached because it is used twice.
        # 1. To evaluate method
        # 2. To create next masking dataset
        dataset_test_masked = dataset_test_masked \
            .apply(masker(masking_ratio)) \
            .cache() \
            .prefetch(tf.data.AUTOTUNE)

        results.append({
            'masking_ratio': masking_ratio,
            **model.evaluate(dataset_test_masked, return_dict=True)
        })
    durations['results'] = timer() - results_time_start

    os.makedirs(args.persistent_dir / 'results', exist_ok=True)
    with open(args.persistent_dir / 'results' / f'{experiment_id}.json', "w") as f:
        del args.persistent_dir
        json.dump({
            'args': vars(args),
            'results': results,
            'durations': durations
        }, f)
