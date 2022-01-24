import argparse
import os.path as path

import tensorflow as tf
from datasets import load_dataset
from transformers import BertTokenizerFast, TFBertForSequenceClassification

thisdir = path.dirname(path.realpath(__file__))
parser = argparse.ArgumentParser()
parser.add_argument('--persistent-dir',
                    action='store',
                    default=path.realpath(path.join(thisdir, '..')),
                    type=str,
                    help='Directory where all persistent data will be stored')

if __name__ == "__main__":
    args = parser.parse_args()

    raw_datasets = load_dataset("imdb", cache_dir=f'{args.persistent_dir}/cache/datasets')
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased", cache_dir=f'{args.persistent_dir}/cache/tokenizer')
    model = TFBertForSequenceClassification.from_pretrained("bert-base-cased", num_labels=2, cache_dir=f'{args.persistent_dir}/cache/transformers')

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
    tf_train_dataset = tokenized_datasets["train"].remove_columns(["text"]).with_format("tensorflow")
    tf_eval_dataset =  tokenized_datasets["test"].remove_columns(["text"]).with_format("tensorflow")

    train_features = {x: tf_train_dataset[x] for x in tokenizer.model_input_names}
    train_tf_dataset = tf.data.Dataset.from_tensor_slices((train_features, tf_train_dataset["label"]))

    eval_features = {x: tf_eval_dataset[x] for x in tokenizer.model_input_names}
    eval_tf_dataset = tf.data.Dataset.from_tensor_slices((eval_features, tf_eval_dataset["label"]))


    train_tf_dataset = train_tf_dataset \
        .cache() \
        .shuffle(len(tf_train_dataset), seed=0) \
        .batch(8) \
        .prefetch(tf.data.AUTOTUNE)
    eval_tf_dataset = eval_tf_dataset  \
        .cache() \
        .batch(8) \
        .prefetch(tf.data.AUTOTUNE)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=tf.metrics.SparseCategoricalAccuracy(),
    )

    model.fit(train_tf_dataset, validation_data=eval_tf_dataset, epochs=3)
