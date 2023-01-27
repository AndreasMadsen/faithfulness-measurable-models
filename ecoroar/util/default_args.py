
# CoLA, disable JIT compile. Small dataset, too slow to JIT compile.
# BoolQ, disable JIT compile. Memory allocation issue, graph to big?
# SST2, disable JIT compile. Small dataset, slightly slower to JIT compile.
# IMDB, disable JIT compile. Small dataset, slightly slower to JIT compile.

def default_jit_compile(args):
    if args.jit_compile is not None:
        return args.jit_compile

    return args.dataset in ['QQP', 'MNLI']

def default_max_epochs(args):
    if args.max_epochs is not None:
        return args.max_epochs

    return ({
        'BoolQ': 15,
        'COLA': 15,
        'IMDB': 10,
        'MNLI': 10,
        'QQP': 10,
        'SST2': 10,
    }).get(args.max_epochs, 20)
