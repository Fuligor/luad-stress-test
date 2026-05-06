from abc import ABCMeta, abstractmethod

from torch import nn
from timm.utils import freeze, unfreeze


class PostInitMeta(ABCMeta):
    def __call__(cls, *args, **kwargs):
        # here is "before __new__ is called"
        instance = super().__call__(*args, **kwargs)
        # here is "after __new__ and __init__"
        if hasattr(instance, "__post_init__"):
            instance.__post_init__()

        return instance


class BaseModel(nn.Module, metaclass=PostInitMeta):
    def __init__(self, freeze_: bool) -> None:
        super().__init__()

        self._frozen = freeze_

    def __post_init__(self) -> None:
        if self._frozen:
            self.freeze()
        else:
            self.unfreeze()

    @property
    def frozen(self) -> bool:
        return self._frozen

    def freeze(self) -> None:
        freeze(self)
        self.eval()
        self._frozen = True

    def unfreeze(self) -> None:
        unfreeze(self)
        self.train()
        self._frozen = False

    @property
    @abstractmethod
    def num_features(self) -> int:
        raise NotImplementedError()
