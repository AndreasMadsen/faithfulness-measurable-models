
def generate_experiment_id(name: str,
                           model: str = None, dataset: str = None,
                           seed: int = None, max_epochs: int = None,
                           max_masking_ratio: int = None, masking_strategy: str = None,
                           validation_dataset: str = None,
                           explainer: str = None, recursive: bool = None, split: str = None,
                           ood: str = None, dist_repeats: int = None):
    """Creates a standardized experiment name.

    The format is
        {name}_m-{model}_d-{dataset}_s-{seed}_r-{max_masking_ratio}
    Note that parts are only added when not None.

    Args:
        name (str): the name of the experiment.
        model (str, optional): the name of the model.
        dataset (str, optional): the name of the dataset.
        seed (int, optional): the models initialization seed.
        max_epochs (int, optional): the max number of epochs to train.
        max_masking_ratio (int, optional): the max masking ratio used during training in percentage.
        masking_strategy (str, optional): the masking strategy to use for masking during fune-tuning.
        validation_dataset (str, optional): The transformation applied to the validation dataset used for early stopping.
        explainer (str, optional): the explanation algorithm used.
        recursive (boo, optional): are the importance measures computed recursively.
        split (str, optional): the split faithfulness evaluation is performed on.
        ood (str, optional): the ood detection method used.
        dist_repeats (int, optional): the number of repeats used to estimate the distribution for ood

    Returns:
        str: the experiment identifier
    """
    experiment_id = f"{name.lower()}"
    if isinstance(model, str):
        experiment_id += f"_m-{model.lower()}"
    if isinstance(dataset, str):
        experiment_id += f"_d-{dataset.lower()}"
    if isinstance(seed, int):
        experiment_id += f"_s-{seed}"
    if isinstance(max_epochs, int):
        experiment_id += f"_e-{max_epochs}"
    if isinstance(max_masking_ratio, int):
        experiment_id += f"_r-{max_masking_ratio}"
    if isinstance(masking_strategy, str):
        experiment_id += f"_y-{masking_strategy}"
    if isinstance(validation_dataset, str):
        experiment_id += f"_v-{validation_dataset}"
    if isinstance(explainer, str):
        experiment_id += f"_x-{explainer.lower()}"
    if isinstance(recursive, bool):
        experiment_id += f"_re-{int(recursive)}"
    if isinstance(split, str):
        experiment_id += f"_sp-{split.lower()}"
    if isinstance(ood, str):
        experiment_id += f"_o-{ood.lower()}"
    if isinstance(dist_repeats, int):
        experiment_id += f"_dr-{dist_repeats}"

    return experiment_id
