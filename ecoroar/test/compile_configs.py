
from dataclasses import dataclass


@dataclass
class CompileConfig:
    name: str
    args: int


compile_configs = [
    CompileConfig('no_compile', {'run_eagerly': True, 'jit_compile': False}),
    CompileConfig('default_compile', {'run_eagerly': False, 'jit_compile': False}),
    CompileConfig('jit_compile', {'run_eagerly': False, 'jit_compile': True})
]
