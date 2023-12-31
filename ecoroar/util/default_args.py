
def default_jit_compile(args):
    """Return the default jit_compile setting based on other arguments

    Args:
        args (Namespace): The argsparse namespace

    Returns:
        bool: the jit_compile flag
    """
    if args.jit_compile is not None:
        return args.jit_compile

    if args.dataset is None:
        return None

    # these are quite slow, this makes them 2x faster
    return args.dataset in ['QQP', 'MNLI', 'QNLI', 'SNLI', 'IMDB']


def default_max_epochs(args):
    """Return the default max_epochs setting based on other arguments

    Args:
        args (Namespace): The argsparse namespace

    Returns:
        int: the max_epoch setting
    """
    if args.max_epochs is not None:
        return args.max_epochs

    if args.dataset is None:
        return None

    return ({
        'BoolQ': 15,
        'CB': 50,
        'CoLA': 15,
        'IMDB': 10,
        'MNLI': 10,
        'QQP': 10,
        'RTE': 30,
        'SNLI': 10,
        'SST2': 10,
    }).get(args.dataset, 20)


def default_recursive(args):
    """Return the default recursive setting based on other arguments

    Args:
        args (Namespace): The argsparse namespace

    Returns:
        bool: the recursive setting
    """
    if args.recursive is not None:
        return args.recursive

    if args.explainer is None:
        return None

    return not args.explainer.startswith('beam-')
