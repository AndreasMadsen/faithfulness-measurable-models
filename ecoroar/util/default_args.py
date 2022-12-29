
# CoLA, disable JIT compile. Small dataset, too slow to JIT compile.
# BoolQ, disable JIT compile. Memory allocation issue, graph to big?
# SST2, disable JIT compile. Small dataset, slightly slower to JIT compile.
# IMDB, disable JIT compile. Small dataset, slightly slower to JIT compile.

def default_jit_compile(args):
    if args.jit_compile is not None:
        return args.jit_compile

    return args.dataset not in ['CoLA', 'BoolQ', 'SST2', 'IMDB']
