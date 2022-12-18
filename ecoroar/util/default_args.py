
# CoLA, disable JIT compile. Small dataset, too slow to JIT compile.
# BoolQ, disable JIT compile. Memory allocation issue, graph to big?

def default_jit_compile(args):
    if args.jit_compile is not None:
        return args.jit_compile

    return args.dataset not in ['CoLA', 'BoolQ']
