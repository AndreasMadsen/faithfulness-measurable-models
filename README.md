# Faithfulness measurable masked language models

This repo contains the code for the paper [_faithfulness measurable masked language models_](https://arxiv.org/abs/2310.07819).

A faithfulness measurable model, is a model that somehow supports that the faithfulness
of an explanation can be validated. In this paper, we propose _masked fine-tuning_ which
makes any permulation of token masking be in-distribution. We apply that method to RoBERTa.

Additionally, we validate that the masking is in-distribution using MaSF, and evaluate
faithfullness on a large array of datasets and explanations.

## Install

This module is not published on PyPi but you can install directly with:

```bash
python -m pip install -e .
```

## API

The module will then be available under `ecoroar`. For example, to use the `MaSF` implementation:

```python
from ecoroar.ood import MaSF
from ecoroar.dataset import BoolQ

tokenizer = HuggingfaceTokenizer(path_to_fine_tuned_model)
model = huggingface_model_from_local(path_to_fine_tuned_model)
ood_detector = MaSF(tokenizer, model)

ood_detector.fit(masked_validation_dataset)
dataset_split_annotated = masked_test_dataset.apply(ood_detector)
```

All parts are documented via their docstring. It is recommended to use the experiment reference scripts
in `experiments/` as examples.

## Experiments

### Tasks

There are scripts for each type of experiment:

1. Model fine-tuning: `python experiments/masking.py`
2. Faithfulness evaluation: `python experiments/faithfulness.py`
3. OOD evaluation: `python experiments/ood.py`

The importance measures used by OOD are calculated in the `python experiments/faithfulness.py` script.

### Parameters

Each of the above scripts takes the same set of CLI arguments. You can learn
about each argument with `--help`. The most important arguments which
will allow you to run the experiments presented in the paper are:

**used in step fine-tuning, faithfulness, and OOD.**

* `--dataset`: The dataset used. Can be `BoolQ`, `CB`, `CoLA`, `IMDB`, `MNLI`, `MRPC`, `QNLI`, `QQP`, `RTE`, `SNLI`, `SST2`, `WNLI`, `bAbI-1`, `bAbI-2`, `bAbI-3`, `MIMIC-d` (Diabetes), or `MIMIC-a` (Anemia).
* `--model`: The model used. Can be `roberta-sl` (RoBERTa-large) or `roberta-sb` (RoBERTa-base).
* `--max-masking-ratio'`: The maximum masking ratio (percentage integer) to apply on the training dataset.
* `--masking-strategy`: The masking strategy to use for masking during fune-tuning.
* `--validation-dataset`: The masking strategy to use for masking during fune-tuning.

Use `--max-masking-ratio 100 --masking-strategy half-det --validation-dataset both` to run the _masked fine-tuning_ experiments.

For example:

```sh
python experiments/masking.py --max-masking-ratio 100 --masking-strategy half-det --validation-dataset both --dataset BoolQ --model roberta-sl
```

**used in step faithfulness and OOD.**

* `--explainer`: The importance measure used. Can be `rand`, `grad-l2`, `grad-l1`, `inp-grad-sign`, `inp-grad-abs`, `int-grad-sign`, `int-grad-abs`, `loo-sign`, `loo-abs`, `beam-sign-50`, `beam-sign-20`, or `beam-sign-10`.

Use `--save-masked-datasets` to save the intermediate masked datasets used for evaluating faithfulness. These can later be reused for checking for OOD issues.

For example:

```sh
python experiments/faithfulness.py --max-masking-ratio 100 --masking-strategy half-det --validation-dataset both --dataset BoolQ --model roberta-sl --explainer int-grad-sign --save-masked-datasets
```

**used in step OOD.**

* `--ood`: The OOD detection method. Should just be `masf` (default).
* `--dist-repeats`: How many interations of the validation dataset should be used. For MaSF, just use `1` (default).

For example:

```sh
python experiments/ood.py --max-masking-ratio 100 --masking-strategy half-det --validation-dataset both --dataset BoolQ --model roberta-sl --explainer int-grad-sign --ood masf
```

## Running on a HPC setup

For downloading the required resources we provide a `experiment/download.py` script.
Additionally, there is a `experiment/preprocess.py` script.

Finally, we provide scripts for submitting all jobs to a Slurm
queue, in `batch_jobs/`. The jobs automatically use `$SCRATCH/ecoroar`
as the persistent dir.

## MIMIC

See https://mimic.physionet.org/gettingstarted/access/ for how to access MIMIC-III.
You will need to download `DIAGNOSES_ICD.csv.gz` and `NOTEEVENTS.csv.gz` and
place them in `mimic/` relative to your presistent dir (e.g. `$SCRATCH/ecoroar/mimic/`).
