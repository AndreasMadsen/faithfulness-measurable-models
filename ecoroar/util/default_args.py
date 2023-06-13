
def default_jit_compile(args):
    if args.jit_compile is not None:
        return args.jit_compile

    if args.dataset is None:
        return None

    # QQP, MNLI, and SNLI are quite slow and raily fails under JIT, this makes them 2x faster
    return args.dataset in ['QQP', 'MNLI', 'SNLI']

def default_max_epochs(args):
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
    if args.recursive is not None:
        return args.recursive

    if args.explainer is None:
        return None

    return not args.explainer.startswith('beam-')
