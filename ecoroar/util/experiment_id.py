
def generate_experiment_id(name: str, seed: int = None, max_masking_ratio: int = None):
    """Creates a standardized experiment name.

    The format is
        {name}_s-{seed}_m-{max_masking_ratio}
    Note that parts are only added when not None.

    Args:
        name (str): the name of the experiment, this is usually the name of the task.
        seed (int, optional): the models initialization seed.
        max_masking_ratio (int, optional): the max masking ratio used during training in percentage.

    Returns:
        str: the experiment identifier
    """
    experiment_id = f"{name}"
    if isinstance(seed, int):
        experiment_id += f"_s-{seed}"
    if isinstance(max_masking_ratio, int):
        experiment_id += f"_m-{max_masking_ratio}"

    return experiment_id
