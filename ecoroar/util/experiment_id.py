
def generate_experiment_id(name, seed=None, max_masking_ratio=None):
    """Creates a standardized experiment name.

    The format is
        {name}_s-{seed}
    Note that parts are only added when not None.

    Args:
        name: str, the name of the experiment, this is usually the name of the task
        seed: int, the models initialization seed
        max_masking_ratio: int, the max masking ratio used during training in percentage
    Returns:
        string, the experiment identifier
    """
    experiment_id = f"{name}"
    if isinstance(seed, int):
        experiment_id += f"_s-{seed}"
    if isinstance(max_masking_ratio, int):
        experiment_id += f"_m-{max_masking_ratio}"

    return experiment_id
