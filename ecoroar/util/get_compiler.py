
import tensorflow as tf

def get_compiler(run_eagerly, jit_compile):
    # define compiler
    if run_eagerly:
        if jit_compile:
            raise ValueError('run_eagerly must be false when jit_compile is True')
        else:
            compiler = lambda x: x
    else:
        if jit_compile:
            compiler = tf.function(reduce_retracing=True, jit_compile=True)
        else:
            compiler = tf.function(reduce_retracing=True)

    return compiler
