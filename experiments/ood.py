import json
import os
import pathlib
import argparse
from timeit import default_timer as timer

from tqdm import tqdm
import tensorflow as tf

from ecoroar.util import generate_experiment_id, model_name_to_huggingface_repo, default_jit_compile, default_max_epochs
from ecoroar.dataset import datasets
from ecoroar.tokenizer import HuggingfaceTokenizer
from ecoroar.model import huggingface_model_from_local
from ecoroar.transform import BucketedPaddedBatch, RandomMaxMasking, TransformSampler, ExplainerMasking, MapOnGPU
from ecoroar.explain import explainers
from ecoroar.ood import MaSF

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
parser.add_argument('--save-annotated-datasets',
                    action=argparse.BooleanOptionalAction,
                    default=False,
                    help='Save annotated dataset')
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
parser.add_argument('--validation-dataset',
                    default='both',
                    choices=['nomask', 'mask', 'both'],
                    type=str,
                    help='The transformation applied to the validation dataset used for early stopping.')
parser.add_argument('--explainer',
                    default='grad',
                    choices=explainers.keys(),
                    type=str,
                    help='The importance measure algorithm to use for explanation')
parser.add_argument('--ood',
                    default='MaSF',
                    choices=['MaSF'],
                    type=str,
                    help='The OOD detection method')
parser.add_argument('--split',
                    default='test',
                    choices=['train', 'valid', 'test'],
                    type=str,
                    help='The dataset split to evaluate faithfulness on')


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
        'ood',
        model=args.model, dataset=args.dataset,
        seed=args.seed, max_epochs=args.max_epochs,
        max_masking_ratio=args.max_masking_ratio, masking_strategy=args.masking_strategy,
        validation_dataset=args.validation_dataset,
        explainer=args.explainer, ood=args.ood,
        split=args.split
    )

    # Print configuration
    print(f'Configuration [{experiment_id}]:')
    print('  Seed:', args.seed)
    print('  Model:', args.model)
    print('  Dataset:', args.dataset)
    print('  Huggingface Repo:', args.huggingface_repo)
    print('  Max masking ratio:', args.max_masking_ratio)
    print('  Masking strategy:', args.masking_strategy)
    print('  Validation dataset:', args.validation_dataset)
    print('')
    print('  Explainer:', args.explainer)
    print('  OOD:', args.ood)
    print('  Split:', args.split)
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
    if tuple(map(int, tf.__version__.split('.'))) >= (2, 12,  0):
        tf.keras.backend.experimental.enable_tf_random_generator()
    tf.keras.utils.set_random_seed(args.seed)
    tf.keras.mixed_precision.set_global_policy(args.precision)

    # Initialize tokenizer, dataset, and model
    tokenizer = HuggingfaceTokenizer(args.huggingface_repo, persistent_dir=args.persistent_dir)
    dataset = datasets[args.dataset](persistent_dir=args.persistent_dir, seed=args.seed)
    model = huggingface_model_from_local(args.persistent_dir / 'checkpoints' / generate_experiment_id(
        'masking',
        model=args.model, dataset=args.dataset,
        seed=args.seed, max_epochs=args.max_epochs,
        max_masking_ratio=args.max_masking_ratio, masking_strategy=args.masking_strategy,
        validation_dataset=args.validation_dataset
    ))
    explainer = explainers[args.explainer](tokenizer, model,
                                           seed=args.seed,
                                           run_eagerly=False, jit_compile=args.jit_compile)
    masker = ExplainerMasking(explainer, tokenizer)
    ood_detector = MaSF(tokenizer, model,
                        run_eagerly=False, jit_compile=args.jit_compile)

    # Load datasets
    dataset_valid = dataset.valid(tokenizer)
    dataset_split = dataset.load(args.split, tokenizer)

    # Setup batching routine
    if args.jit_compile:
        batcher = BucketedPaddedBatch([dataset_valid, dataset_split], batch_size=args.batch_size)
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

    # Configure model
    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, name='cross_entropy'),
        metrics=dataset.metrics(),
        run_eagerly=False,
        jit_compile=args.jit_compile
    )

    durations['setup'] = timer() - setup_time_start

    # Train distributional representation using  validation dataset
    # Note, this dataset needs to be masked the same way as the training dataset during training.
    odd_fit_time_start = timer()
    # TODO: figure out what is the appropiate number of repeats
    dataset_valid_masked = dataset_valid \
        .repeat(1) \
        .apply(batcher(args.batch_size,
                        padding_values=(tokenizer.padding_values, None),
                        num_parallel_calls=tf.data.AUTOTUNE)) \
        .map(lambda x, y: (masker_train(x), y), num_parallel_calls=tf.data.AUTOTUNE) \
        .prefetch(tf.data.AUTOTUNE)

    # TODO: Rerunning this for every --explainer argument is quite the waste,
    #   since the distributed representation will be the same for every --explainer
    ood_detector.fit(dataset_valid_masked)
    durations['ood_fit'] = timer() - odd_fit_time_start

    # Evalute statistical OOD test at each masking ratios
    results = []
    measure_time = 0
    summary_time = 0
    masked_dataset_prefix = args.persistent_dir / 'intermediate' / 'faithfulness' / generate_experiment_id(
            'faithfulness',
            model=args.model, dataset=args.dataset,
            seed=args.seed, max_epochs=args.max_epochs,
            max_masking_ratio=args.max_masking_ratio, masking_strategy=args.masking_strategy,
            validation_dataset=args.validation_dataset,
            explainer=args.explainer,
            split=args.split
        )
    ood_intermediate_dir = args.persistent_dir / 'intermediate' / 'ood'
    os.makedirs(ood_intermediate_dir, exist_ok=True)
    for masking_ratio in [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
        dataset_split_masked = tf.data.Dataset.load(
            str(masked_dataset_prefix.with_suffix(f'.{masking_ratio}.tfds'))
        )

        measure_time_start = timer()
        # assigns p-values to each observation
        dataset_split_annotated = dataset_split_masked \
            .apply(MapOnGPU(
                lambda x, y: (x, y, ood_detector(x)),
                lambda dataset: (*tf.data.experimental.get_structure(dataset), tf.TensorSpec(shape=[None], dtype=tf.dtypes.float32))
            )) \
            .cache()

        for x, y, ood in tqdm(dataset_split_annotated, desc=f'OOD annotating dataset ({masking_ratio}%)', mininterval=1):
            pass
        measure_time += timer() - measure_time_start

        # save annotated dataset
        if args.save_annotated_datasets:
            dataset_split_annotated.save(
                str((ood_intermediate_dir / experiment_id).with_suffix(f'.{masking_ratio}.tfds'))
            )

        # aggregate p-value statistics as a basic histogram
        summary_time_start = timer()
        p_value_thresholds = [0.001, 0.005, 0.01, 0.05, 0.1]
        p_values_histogram = dataset_split_annotated.reduce(
            (
                tf.zeros(1, dtype=tf.dtypes.int32),  # count
                tf.zeros(len(p_value_thresholds), dtype=tf.dtypes.int32)  # hist
            ),
            lambda state, batch: (  # state = (count, hist), batch = (x, y, ood)
                state[0] + tf.shape(batch[2])[0],
                state[1] + tf.math.reduce_sum(tf.cast(
                    tf.expand_dims(batch[2], 0) < tf.expand_dims(p_value_thresholds, 1),
                    dtype=tf.dtypes.int32), axis=1)
            )
        )
        p_values_histogram_prop = p_values_histogram[1] / p_values_histogram[0]

        # save summarized results
        for threshold, proportion in zip(p_value_thresholds, p_values_histogram_prop.numpy()):
            results.append({
                'masking_ratio': masking_ratio / 100,
                'proportion': proportion.item(),
                'threshold': threshold
            })
        summary_time = timer() - summary_time_start

    durations['measure'] = measure_time
    durations['summary'] = summary_time

    os.makedirs(args.persistent_dir / 'results' / 'ood', exist_ok=True)
    with open(args.persistent_dir / 'results' / 'ood' / f'{experiment_id}.json', "w") as f:
        del args.persistent_dir
        json.dump({
            'args': vars(args),
            'results': results,
            'durations': durations
        }, f)
