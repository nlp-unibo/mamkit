import torch as th
from cinnamon.component import Component

from mamkit.components.data.processing import Processor


class MAMKitModule(th.nn.Module, Component):

    def setup(
            self,
            processor: Processor
    ):
        pass
