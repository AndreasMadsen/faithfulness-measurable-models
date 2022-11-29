import os
import sys
import pathlib
import argparse
import json
import shutil
import tempfile
from timeit import default_timer as timer

import tensorflow as tf
from ecoroar.util import generate_experiment_id, model_name_to_huggingface_repo
from ecoroar.dataset import datasets
from ecoroar.tokenizer import HuggingfaceTokenizer
from ecoroar.model import HuggingfaceModel
from ecoroar.transform import RandomMasking
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
                    default=3,
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

if __name__ == '__main__':
    args = parser.parse_args()
    if args.huggingface_repo is None:
        args.huggingface_repo = model_name_to_huggingface_repo(args.model)

    if args.deterministic:
        tf.config.experimental.enable_op_determinism()
    tf.keras.utils.set_random_seed(args.seed)
    tf.keras.mixed_precision.set_global_policy(args.precision)

    durations = {}
    experiment_id = generate_experiment_id(
        'masking',
        model=args.model, dataset=args.dataset,
        seed=args.seed, max_masking_ratio=args.max_masking_ratio
    )

    tokenizer = HuggingfaceTokenizer(args.huggingface_repo, persistent_dir=args.persistent_dir)
    dataset = datasets[args.dataset](persistent_dir=args.persistent_dir, seed=args.seed)
    model = HuggingfaceModel(args.huggingface_repo, persistent_dir=args.persistent_dir, num_classes=dataset.num_classes)

    masker_train = RandomMasking(args.max_masking_ratio / 100, tokenizer, seed=args.seed)
    dataset_train = dataset.train(tokenizer) \
        .shuffle(dataset.train_num_examples, seed=args.seed) \
        .map(lambda x, y: (masker_train(x), y), num_parallel_calls=tf.data.AUTOTUNE) \
        .padded_batch(args.batch_size, padding_values=(tokenizer.padding_values, None)) \
        .prefetch(tf.data.AUTOTUNE)

    dataset_valid = dataset.valid(tokenizer) \
        .padded_batch(args.batch_size, padding_values=(tokenizer.padding_values, None)) \
        .prefetch(tf.data.AUTOTUNE)


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
        run_eagerly=False
    )

    checkpoint_dir = tempfile.mkdtemp()
    tensorboard_dir = tempfile.mkdtemp()

    train_time_start = timer()
    model.fit(dataset_train, validation_data=dataset_valid, epochs=args.max_epochs, callbacks=[
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_dir,
            monitor=dataset.early_stopping_metric, mode='max',
            save_weights_only=True,
            save_best_only=True
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=tensorboard_dir,
            write_graph=False
        )
    ])
    durations['train_time'] = timer() - train_time_start

    dataset_test = dataset.test(tokenizer)
    results_test = []
    for test_masking_ratio in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        masker_test = RandomMasking(test_masking_ratio, tokenizer, seed=args.seed)
        dataset_test_with_masking = dataset_test \
            .map(lambda x, y: (masker_train(x), y), num_parallel_calls=tf.data.AUTOTUNE) \
            .padded_batch(args.batch_size, padding_values=(tokenizer.padding_values, None)) \
            .prefetch(tf.data.AUTOTUNE)

        test_time_start = timer()
        results_test.append({
            'masking_ratio': test_masking_ratio,
            **model.evaluate(dataset_test_with_masking, return_dict=True)
        })
        durations['test_time_{test_masking_ratio}'] = timer() - test_time_start

    # Save results
    os.makedirs(args.persistent_dir / 'checkpoints', exist_ok=True)
    shutil.move(checkpoint_dir, args.persistent_dir / 'checkpoints' / experiment_id)

    os.makedirs(args.persistent_dir / 'results', exist_ok=True)
    shutil.move(tensorboard_dir, args.persistent_dir / 'tensorboard' / experiment_id)

    os.makedirs(args.persistent_dir / 'results', exist_ok=True)
    with open(args.persistent_dir / 'results' / f'{experiment_id}.json', "w") as f:
        del args.persistent_dir
        json.dump({'args': vars(args), 'results': results_test, 'durations': durations}, f)
