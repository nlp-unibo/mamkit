import torch as th
from mamkit.components.data.processing import Processor


class MAMKitModule(th.nn.Module):

    def setup(
            self,
            processor: Processor
    ):
        pass