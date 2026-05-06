from torch import nn


class AbstractHead(nn.Module):
    def __init__(self, input_features, num_classes):
        super().__init__()

        self.input_features = input_features
        self.num_classes = num_classes
