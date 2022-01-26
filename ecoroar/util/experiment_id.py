
def generate_experiment_id(name, seed=None):
    """Creates a standardized experiment name.

    The format is
        {name}_s-{seed}
    Note that parts are only added when not None.

    Args:
        name: str, the name of the experiment, this is usually the name of the task
        seed: int, the models initialization seed
    Returns:
        string, the experiment identifier
    """
    experiment_id = f"{name}"
    if isinstance(seed, int):
        experiment_id += f"_s-{seed}"

    return experiment_id
