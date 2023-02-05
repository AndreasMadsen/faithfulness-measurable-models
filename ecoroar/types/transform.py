
from abc import ABC, abstractmethod

from .tokenizer import TokenizedDict


class InputTransform(ABC):
    @abstractmethod
    def __call__(self, x: TokenizedDict) -> TokenizedDict:
        pass
