import os
import sys
import pathlib
import argparse
import json
import shutil
import tempfile
from timeit import default_timer as timer

import tensorflow as tf
from ecoroar.util import generate_experiment_id, model_name_to_huggingface_repo, default_jit_compile, default_max_epochs
from ecoroar.dataset import datasets
from ecoroar.tokenizer import HuggingfaceTokenizer
from ecoroar.model import HuggingfaceModel
from ecoroar.transform import RandomFixedMasking, RandomMaxMasking, BucketedPaddedBatch, TransformSampler
from ecoroar.optimizer import AdamW
from ecoroar.scheduler import LinearSchedule

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
parser.add_argument('--weight-decay',
                    action='store',
                    default=0.01,
                    type=float,
                    help='Weight decay')
parser.add_argument('--lr',
                    action='store',
                    default=2e-5,
                    type=float,
                    help='Learning rate')
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
        'masking',
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
    print('')
    print('  Weight Decay:', args.weight_decay)
    print('  Learning rate:', args.lr)
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
    model = HuggingfaceModel(args.huggingface_repo, persistent_dir=args.persistent_dir, num_classes=dataset.num_classes)

    # Load datasets
    dataset_train = dataset.train(tokenizer)
    dataset_valid = dataset.valid(tokenizer)
    dataset_test = dataset.test(tokenizer)

    # Setup batching routine
    if args.jit_compile:
        batcher = BucketedPaddedBatch([dataset_train, dataset_valid, dataset_test], batch_size=args.batch_size)
    else:
        batcher = lambda batch_size, padding_values, num_parallel_calls: \
            lambda dataset: dataset.padded_batch(batch_size, padding_values=padding_values)

    # Setup masking routine for training
    match args.masking_strategy:
        case 'uni':
            masker_train = RandomMaxMasking(args.max_masking_ratio / 100, tokenizer, seed=args.seed)
        case 'half-det':
            masker_train = TransformSampler([
                RandomMaxMasking(0, tokenizer, seed=args.seed),
                RandomMaxMasking(args.max_masking_ratio / 100, tokenizer, seed=args.seed)
            ], seed=args.seed, stochastic=False)
        case 'half-ran':
            masker_train = TransformSampler([
                RandomMaxMasking(0, tokenizer, seed=args.seed),
                RandomMaxMasking(args.max_masking_ratio / 100, tokenizer, seed=args.seed)
            ], seed=args.seed, stochastic=True)

    # Mask training dataset and batch it
    dataset_train_batched = dataset_train \
        .shuffle(dataset.train_num_examples, seed=args.seed) \
        .apply(batcher(args.batch_size,
                       padding_values=(tokenizer.padding_values, None),
                       num_parallel_calls=tf.data.AUTOTUNE)) \
        .map(lambda x, y: (masker_train(x), y), num_parallel_calls=tf.data.AUTOTUNE) \
        .prefetch(tf.data.AUTOTUNE)

    # Batch validation dataset
    dataset_valid_batched = dataset_valid \
        .apply(batcher(args.batch_size,
                       padding_values=(tokenizer.padding_values, None),
                       num_parallel_calls=tf.data.AUTOTUNE)) \
        .prefetch(tf.data.AUTOTUNE)

    # Configure model
    model.compile(
        optimizer=AdamW(
            learning_rate=LinearSchedule(
                learning_rate=args.lr,
                num_training_steps=args.max_epochs * len(dataset_train)
            ),
            weight_decay=args.weight_decay
        ),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, name='cross_entropy'),
        metrics=dataset.metrics(),
        run_eagerly=False,
        jit_compile=args.jit_compile
    )

    checkpoint_dir = tempfile.mkdtemp()
    tensorboard_dir = tempfile.mkdtemp()


    # Train models and collect validation performance at each epoch
    durations['setup'] = timer() - setup_time_start
    train_time_start = timer()
    history = model.fit(dataset_train_batched, validation_data=dataset_valid_batched, epochs=args.max_epochs, callbacks=[
        tf.keras.callbacks.ModelCheckpoint(
            filepath=f'{checkpoint_dir}/model.ckpt',
            monitor=dataset.early_stopping_metric, mode='max',
            save_weights_only=True,
            save_best_only=True
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=tensorboard_dir,
            write_graph=False
        )
    ])
    durations['train'] = timer() - train_time_start
    test_time_start = timer()

    # Load the model weights that are considered the best checkpoint by the validation metric
    model.load_weights(f'{checkpoint_dir}/model.ckpt')

    # Evalute test performance at different masking ratios
    results_test = []
    for test_masking_ratio in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        masker_test = RandomFixedMasking(test_masking_ratio, tokenizer, seed=args.seed)
        dataset_test_batched = dataset_test \
            .apply(batcher(args.batch_size,
                           padding_values=(tokenizer.padding_values, None),
                           num_parallel_calls=tf.data.AUTOTUNE)) \
            .map(lambda x, y: (masker_test(x), y), num_parallel_calls=tf.data.AUTOTUNE) \
            .prefetch(tf.data.AUTOTUNE)

        results_test.append({
            'masking_ratio': test_masking_ratio,
            **model.evaluate(dataset_test_batched, return_dict=True)
        })
    durations['test'] = timer() - test_time_start

    # Save results
    os.makedirs(args.persistent_dir / 'checkpoints', exist_ok=True)
    shutil.rmtree(args.persistent_dir / 'checkpoints' / experiment_id, ignore_errors=True)
    model.save_weights(args.persistent_dir / 'checkpoints' / experiment_id / 'model.h5', save_format="h5")

    os.makedirs(args.persistent_dir / 'tensorboard', exist_ok=True)
    shutil.rmtree(args.persistent_dir / 'tensorboard' / experiment_id, ignore_errors=True)
    shutil.move(tensorboard_dir, args.persistent_dir / 'tensorboard' / experiment_id)

    os.makedirs(args.persistent_dir / 'results', exist_ok=True)
    with open(args.persistent_dir / 'results' / f'{experiment_id}.json', "w") as f:
        del args.persistent_dir
        json.dump({
            'args': vars(args),
            'history': [
                { key: values[epoch] for key, values in history.history.items() } | { 'epoch': epoch }
                for epoch in range(history.params['epochs'])
            ],
            'results': results_test,
            'durations': durations
        }, f)
