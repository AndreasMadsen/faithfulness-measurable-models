
import argparse
import pathlib
import json
import os

from tqdm import tqdm
import pandas as pd
import tensorflow as tf

from ecoroar.dataset import datasets
from ecoroar.util import generate_experiment_id, default_max_epochs, model_name_to_huggingface_repo
from ecoroar.tokenizer import HuggingfaceTokenizer
from ecoroar.model import huggingface_model_from_local

parser = argparse.ArgumentParser(
    description = 'Exports masked datasets as .csv'
)
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
parser.add_argument('--max-masking-ratio',
                    action='store',
                    default=100,
                    type=int,
                    help='The maximum masking ratio (percentage integer) to apply on the training dataset')
parser.add_argument('--masking-strategy',
                    default='half-det',
                    choices=['uni', 'half-det', 'half-ran'],
                    type=str,
                    help='The masking strategy to use for masking during fune-tuning')
parser.add_argument('--max-batches',
                    default=None,
                    type=int,
                    help='The number of batches to extract')

if __name__ == "__main__":
    pd.set_option('display.max_rows', None)
    args = parser.parse_args()
    args.max_epochs = default_max_epochs(args)
    if args.huggingface_repo is None:
        args.huggingface_repo = model_name_to_huggingface_repo(args.model)

    experiment_id = generate_experiment_id(
        'faithfulness',
        model=args.model, dataset=args.dataset,
        seed=args.seed, max_epochs=args.max_epochs,
        max_masking_ratio=args.max_masking_ratio, masking_strategy=args.masking_strategy,
    )

    # Load componenets
    dataset = datasets[args.dataset](persistent_dir=args.persistent_dir, seed=args.seed)
    tokenizer = HuggingfaceTokenizer(args.huggingface_repo, persistent_dir=args.persistent_dir)
    model = huggingface_model_from_local(args.persistent_dir / 'checkpoints' / generate_experiment_id(
        'masking',
        model=args.model, dataset=args.dataset,
        seed=args.seed, max_epochs=args.max_epochs,
        max_masking_ratio=args.max_masking_ratio, masking_strategy=args.masking_strategy
    ))
    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, name='cross_entropy'),
        run_eagerly=False,
        jit_compile=False
    )

    @tf.function(reduce_retracing=True)
    def model_predict_batch(x, y):
        y_logits = model(x, training=False).logits
        y_preds = tf.nn.softmax(y_logits, axis=-1)
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(y, y_logits)
        return (losses, y_preds)

    dataframe = []
    # Read intermediate datasets into dataframe
    experiment_id_glob = generate_experiment_id(
        'faithfulness',
        model=args.model, dataset=args.dataset,
        seed=args.seed, max_epochs=args.max_epochs,
        max_masking_ratio=args.max_masking_ratio, masking_strategy=args.masking_strategy,
        explainer='*'
    )
    files = sorted((args.persistent_dir / 'intermediate' / 'faithfulness').glob(f'{experiment_id_glob}.*.tfds'))
    for file in tqdm(files, desc='Loading .tfds files'):
        experiment_id_full, masking_ratio = file.stem.split('.')
        masking_ratio = int(masking_ratio)

        # Load main results, used for parsing the experiment_id
        result_file = (args.persistent_dir / 'results' / experiment_id_full).with_suffix('.json')
        if not result_file.exists():
            tqdm.write(f'Could not find {result_file}')
            continue

        with open(result_file, 'r') as fp:
            try:
                explainer = json.load(fp)['args']['explainer']
            except json.decoder.JSONDecodeError:
                print(f'{file} has a format error')

        # Load datasets
        intermediate_dataset = tf.data.Dataset.load(str(file))
        if args.max_batches is not None:
            intermediate_dataset = intermediate_dataset.take(args.max_batches)
        intermediate_dataset = intermediate_dataset.prefetch(tf.data.AUTOTUNE)

        for batch_i, (x, y) in enumerate(intermediate_dataset):
            losses, y_preds = model_predict_batch(x, y)

            # Convert input to text
            input_ids = tf.RaggedTensor.from_tensor(x['input_ids'], padding=tokenizer.pad_token_id)
            input_tokens = tokenizer.covert_ids_to_tokens(input_ids)
            input_texts = tf.strings.reduce_join(input_tokens, axis=-1, separator=' ')

            # Save batch into dataframe
            for observation_i, (text, label, y_pred, loss) in enumerate(zip(input_texts, y, y_preds, losses)):
                y_pred_annotated = {
                    f'pred_{name}': prop.numpy() for name, prop in zip(dataset.class_names, y_pred)
                }

                dataframe.append({
                    'IM': explainer,
                    'ratio': masking_ratio,
                    'obs': f'{batch_i}.{observation_i}',
                    'text': text.numpy().decode('utf-8'),
                    'target': dataset.class_names[label.numpy()],
                    'loss': loss.numpy(),
                    'pred': y_pred[label].numpy(),
                    **y_pred_annotated
                })

    # Save dataframe
    os.makedirs(args.persistent_dir / 'tables', exist_ok=True)
    pd.DataFrame(dataframe).to_parquet(
        (args.persistent_dir / 'tables' / experiment_id).with_suffix('.parquet'),
        index=False
    )
