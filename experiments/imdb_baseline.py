import argparse
import os.path as path
from functools import partial

import tensorflow as tf
from transformers import TFBertForSequenceClassification

from ecoroar.util import generate_experiment_id
from ecoroar.dataset import IMDBDataset
from ecoroar.tokenizer import BertTokenizer

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

if __name__ == '__main__':
    args = parser.parse_args()
    tf.keras.utils.set_random_seed(args.seed)
    experiment_id = generate_experiment_id('imdb', seed=args.seed)

    dataset = IMDBDataset(persistent_dir=args.persistent_dir, seed=args.seed)
    tokenizer = BertTokenizer('bert-base-cased', persistent_dir=args.persistent_dir)
    model = TFBertForSequenceClassification.from_pretrained('bert-base-cased',
        num_labels=dataset.num_classes, cache_dir=f'{args.persistent_dir}/cache/transformers')

    dataset_train = dataset.train \
        .map(lambda item: (tokenizer(item['text']), item['label']),
             num_parallel_calls=tf.data.AUTOTUNE, deterministic=True) \
        .cache() \
        .shuffle(dataset.train_num_examples, seed=args.seed) \
        .padded_batch(8, padding_values=(tokenizer.padding_values, None)) \
        .prefetch(tf.data.AUTOTUNE)

    dataset_valid = dataset.valid \
        .map(lambda item: (tokenizer(item['text']), item['label']),
             num_parallel_calls=tf.data.AUTOTUNE, deterministic=True) \
        .cache() \
        .padded_batch(8, padding_values=(tokenizer.padding_values, None)) \
        .prefetch(tf.data.AUTOTUNE)

    dataset_test = dataset.test \
        .map(lambda item: (tokenizer(item['text']), item['label']),
             num_parallel_calls=tf.data.AUTOTUNE, deterministic=True) \
        .cache() \
        .padded_batch(8, padding_values=(tokenizer.padding_values, None)) \
        .prefetch(tf.data.AUTOTUNE)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=tf.metrics.SparseCategoricalAccuracy()
    )

    model.fit(dataset_train, validation_data=dataset_valid, epochs=args.max_epochs, callbacks=[
        tf.keras.callbacks.ModelCheckpoint(
            filepath=f'{args.persistent_dir}/checkpoints/{experiment_id}',
            monitor='val_acc',
            save_best_only=True
        )
    ])

    results = model.evaluate(dataset_test)
    print(results)

    os.makedirs(f'{args.persistent_dir}/results/', exist_ok=True)
    with open(f'{args.persistent_dir}/results/{experiment_id}.json', "w") as f:
        json.dump({'seed': args.seed, 'dataset': dataset.name, **results}, f)
