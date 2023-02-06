
def generate_experiment_id(name: str,
                           model: str = None, dataset: str = None,
                           seed: int = None, max_masking_ratio: int = None,
                           max_epochs: int = None):
    """Creates a standardized experiment name.

    The format is
        {name}_m-{model}_d-{dataset}_s-{seed}_r-{max_masking_ratio}
    Note that parts are only added when not None.

    Args:
        name (str): the name of the experiment.
        model (str, optional): the name of the model.
        dataset (str, optional): the name of the dataset.
        seed (int, optional): the models initialization seed.
        max_masking_ratio (int, optional): the max masking ratio used during training in percentage.
        max_epochs (int, optional): the max number of epochs to train.

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
    if isinstance(max_masking_ratio, int):
        experiment_id += f"_r-{max_masking_ratio}"
    if isinstance(max_epochs, int):
        experiment_id += f"_e-{max_epochs}"

    return experiment_id
