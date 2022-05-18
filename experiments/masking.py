import os
import sys
import pathlib
import argparse
import json
import shutil
import tempfile

import tensorflow as tf
from ecoroar.util import generate_experiment_id
from ecoroar.dataset import datasets
from ecoroar.tokenizer import HuggingfaceTokenizer
from ecoroar.model import HuggingfaceModel
from ecoroar.metric import AUROC, F1Score
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
                    default='roberta-base',
                    type=str,
                    help='Model type to use')
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
                    default=False,
                    help='Use determinstic computations')
parser.add_argument('--precision',
                    action='store',
                    default='mixed_float16',
                    choices=['mixed_float16', 'mixed_bfloat16', 'float32'],
                    help='Set the precision policy')
parser.add_argument('--max-masking-ratio',
                    action='store',
                    default=0,
                    type=int,
                    help='The maximum masking ratio (percentage integer) to apply on the training dataset')
parser.add_argument('--experiment-name',
                    action='store_true',
                    default=False,
                    help='Output the experiment name, do nothing else.')

if __name__ == '__main__':
    args = parser.parse_args()

    experiment_id = generate_experiment_id(
        'masking',
        dataset=args.dataset, seed=args.seed, max_masking_ratio=args.max_masking_ratio
    )
    if args.experiment_name:
        print(experiment_id)
        sys.exit(0)

    if args.deterministic:
        tf.config.experimental.enable_op_determinism()
    tf.keras.utils.set_random_seed(args.seed)
    tf.keras.mixed_precision.set_global_policy(args.precision)

    tokenizer = HuggingfaceTokenizer(args.model, persistent_dir=args.persistent_dir)
    dataset = datasets[args.dataset](persistent_dir=args.persistent_dir, seed=args.seed)
    model = HuggingfaceModel(args.model, persistent_dir=args.persistent_dir, num_classes=dataset.num_classes)
    masker = RandomMasking(args.max_masking_ratio / 100, tokenizer, seed=args.seed)

    dataset_train = dataset.train(tokenizer) \
        .shuffle(dataset.train_num_examples, seed=args.seed) \
        .map(lambda x, y: (masker(x), y), num_parallel_calls=tf.data.AUTOTUNE) \
        .padded_batch(args.batch_size, padding_values=(tokenizer.padding_values, None)) \
        .prefetch(tf.data.AUTOTUNE)

    dataset_valid = dataset.valid(tokenizer) \
        .padded_batch(args.batch_size, padding_values=(tokenizer.padding_values, None)) \
        .prefetch(tf.data.AUTOTUNE)

    dataset_test = dataset.test(tokenizer) \
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
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, name='ce'),
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(name='acc'),
            AUROC(name='auroc', from_logits=True),
            F1Score(num_classes=dataset.num_classes, average='macro', name='macro_f1'),
            F1Score(num_classes=dataset.num_classes, average='micro', name='micro_f1')
        ],
        run_eagerly=False
    )

    checkpoint_dir = tempfile.mkdtemp()
    tensorboard_dir = tempfile.mkdtemp()
    model.fit(dataset_train, validation_data=dataset_valid, epochs=args.max_epochs, callbacks=[
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_dir,
            monitor=f'val_{dataset.metric}', mode='max',
            save_weights_only=True,
            save_best_only=True
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=tensorboard_dir,
            write_graph=False
        )
    ])
    results = model.evaluate(dataset_test, return_dict=True)

    # Save results
    os.makedirs(args.persistent_dir / 'checkpoints', exist_ok=True)
    shutil.move(checkpoint_dir, args.persistent_dir / 'checkpoints' / experiment_id)

    os.makedirs(args.persistent_dir / 'results', exist_ok=True)
    shutil.move(tensorboard_dir, args.persistent_dir / 'tensorboard' / experiment_id)

    os.makedirs(args.persistent_dir / 'results', exist_ok=True)
    with open(args.persistent_dir / 'results' / f'{experiment_id}.json', "w") as f:
        del args.persistent_dir
        json.dump({'dataset': dataset.name, **vars(args), **results}, f)
