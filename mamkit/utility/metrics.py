from torch import Tensor
from torchmetrics.classification.f_beta import MulticlassF1Score
from torchmetrics.functional.classification.f_beta import _fbeta_reduce


class ClassSubsetMulticlassF1Score(MulticlassF1Score):

    def __iter__(self):
        pass

    def __init__(
            self,
            class_subset,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.class_subset = class_subset

    def compute(self) -> Tensor:
        tp, fp, tn, fn = self._final_state()
        f1 = _fbeta_reduce(tp, fp, tn, fn, self.beta, average='none', multidim_average=self.multidim_average)
        return f1[self.class_subset].mean()
