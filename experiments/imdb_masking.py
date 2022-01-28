import os
import os.path as path
import argparse
import json
from functools import partial

import tensorflow as tf
from transformers import TFBertForSequenceClassification

from ecoroar.util import generate_experiment_id
from ecoroar.dataset import IMDBDataset
from ecoroar.tokenizer import BertTokenizer
from ecoroar.metric import AUROC, F1Score
from ecoroar.transform import RandomMasking

thisdir = path.dirname(path.realpath(__file__))
parser = argparse.ArgumentParser()
parser.add_argument('--persistent-dir',
                    action='store',
                    default=path.realpath(path.join(thisdir, '..')),
                    type=str,
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
parser.add_argument('--model',
                    action='store',
                    default='base-cased', # bert_uncased_L-2_H-128_A-2
                    type=str,
                    help='Model type to use')
parser.add_argument('--max-masking-ratio',
                    action='store',
                    default=0,
                    type=int,
                    help='The maximum masking ratio (percentage integer) to apply on the training dataset')

if __name__ == '__main__':
    args = parser.parse_args()
    # tf.config.experimental.enable_op_determinism()
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    tf.keras.utils.set_random_seed(args.seed)

    experiment_id = generate_experiment_id('imdb', seed=args.seed, max_masking_ratio=args.max_masking_ratio)

    dataset = IMDBDataset(persistent_dir=args.persistent_dir, seed=args.seed)
    tokenizer = BertTokenizer(f'bert-{args.model}', persistent_dir=args.persistent_dir)
    model = TFBertForSequenceClassification.from_pretrained(f'bert-{args.model}',
        num_labels=dataset.num_classes, cache_dir=f'{args.persistent_dir}/download/transformers')
    masker = RandomMasking(args.max_masking_ratio / 100, tokenizer, seed=args.seed)

    dataset_train = dataset.train \
        .map(lambda item: (tokenizer(item['text']), item['label']),
             num_parallel_calls=tf.data.AUTOTUNE) \
        .cache() \
        .shuffle(dataset.train_num_examples, seed=args.seed) \
        .map(lambda x, y: (masker(x), y),
             num_parallel_calls=tf.data.AUTOTUNE) \
        .padded_batch(8, padding_values=(tokenizer.padding_values, None)) \
        .prefetch(tf.data.AUTOTUNE)

    dataset_valid = dataset.valid \
        .map(lambda item: (tokenizer(item['text']), item['label']),
             num_parallel_calls=tf.data.AUTOTUNE) \
        .cache() \
        .padded_batch(8, padding_values=(tokenizer.padding_values, None)) \
        .prefetch(tf.data.AUTOTUNE)

    dataset_test = dataset.test \
        .map(lambda item: (tokenizer(item['text']), item['label']),
             num_parallel_calls=tf.data.AUTOTUNE) \
        .cache() \
        .padded_batch(8, padding_values=(tokenizer.padding_values, None)) \
        .prefetch(tf.data.AUTOTUNE)

    # TODO: add macro and micro F1Scores when compute-canada issue regarding tensorflow_addons is solved
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, name='ce'),
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(name='acc'),
            AUROC(name='auroc', from_logits=True),
            F1Score(num_classes=dataset.num_classes, average='macro', name='macro_f1'),
            F1Score(num_classes=dataset.num_classes, average='micro', name='micro_f1')
        ]
    )

    model.fit(dataset_train, validation_data=dataset_valid, epochs=args.max_epochs, callbacks=[
        tf.keras.callbacks.ModelCheckpoint(
            filepath=f'{args.persistent_dir}/checkpoints/{experiment_id}',
            monitor='val_auroc', mode='max',
            save_weights_only=True,
            save_best_only=True
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=f'{args.persistent_dir}/tensorboard/{experiment_id}',
            write_graph=False
        )
    ])

    results = model.evaluate(dataset_test, return_dict=True)
    print(results)

    os.makedirs(f'{args.persistent_dir}/results/masking-effect/', exist_ok=True)
    with open(f'{args.persistent_dir}/results/masking-effect/{experiment_id}.json', "w") as f:
        json.dump({'dataset': dataset.name, **args, **results}, f)
