
def get_compiler(run_eagerly, jit_compile):
    """Convert run_eagerly and jit_compile settings into a tf.function decorator

    Args:
        run_eagerly (bool): If true, do not perform any compilation
        jit_compile (bool): If true, perform jit compilations

    Raises:
        ValueError: If both run_eagerly and jit_compile are true

    Returns:
        tf.function(): the compiler function return by tf.function
    """
    # Lazy load, as ecoroar.util is used in experiment_name.py and
    # the startup time for that script is too slow, if tensorflow as to be loaded.
    import tensorflow as tf

    # define compiler
    if run_eagerly:
        if jit_compile:
            raise ValueError('run_eagerly must be false when jit_compile is True')
        else:
            def compiler(x): return x
    else:
        if jit_compile:
            compiler = tf.function(reduce_retracing=True, jit_compile=True)
        else:
            compiler = tf.function(reduce_retracing=True)

    return compiler
