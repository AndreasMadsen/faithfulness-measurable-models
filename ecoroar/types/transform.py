
from abc import ABC, abstractmethod

from .tokenizer import TokenizedDict


class InputTransform(ABC):
    @abstractmethod
    def __call__(self, x: TokenizedDict) -> TokenizedDict:
        """Transform a TokenizedDict to a TokenizedDict

        An example of this, could be to randomly mask tokens

        Args:
            x (TokenizedDict): the tokenized observation

        Returns:
            TokenizedDict:  a transform of the tokenized observation
        """
        pass
