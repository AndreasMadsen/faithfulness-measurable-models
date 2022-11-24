
_CONVERTION_TABLE = {
    'roberta-sb': 'roberta-base',
    'roberta-sl': 'roberta-large',
    'roberta-m15': 'andreasmadsen/efficient_mlm_m0.15',
    'roberta-m20': 'andreasmadsen/efficient_mlm_m0.20',
    'roberta-m30': 'andreasmadsen/efficient_mlm_m0.30',
    'roberta-m40': 'andreasmadsen/efficient_mlm_m0.40',
    'roberta-m50': 'andreasmadsen/efficient_mlm_m0.50',
    'roberta-m60': 'andreasmadsen/efficient_mlm_m0.60',
    'roberta-m70': 'andreasmadsen/efficient_mlm_m0.70',
    'roberta-m80': 'andreasmadsen/efficient_mlm_m0.80',
}

def model_name_to_huggingface_repo(short_model_name: str) -> str:
    """Converts a project specific short model name to a valid huggingface repo

    Args:
        short_model_name (str): The short model name

    Returns:
        str: valid huggingface repo
    """

    return _CONVERTION_TABLE[short_model_name]
