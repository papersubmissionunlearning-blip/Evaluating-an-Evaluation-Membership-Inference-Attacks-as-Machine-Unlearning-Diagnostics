import torch
import torch.nn as nn
import torch.nn.functional as F
from resnet import ResNet, BasicBlock

class ShadowModel(nn.Module):
    """
    Wrapper around ResNet-18 (ResNet(BasicBlock, [2,2,2,2])).
    - forward(x) -> logits (batch, num_classes)
    - feature(x) -> flattened features before final fc (batch, feat_dim)
    """
    def __init__(self, channel=3, num_classes=10):
        super(ShadowModel, self).__init__()
        # instantiate ResNet-18; the ResNet implementation is expected to
        # follow the conventional API (conv1, bn1, relu, maxpool, layer1..4, avgpool, fc)
        self.net = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)

    def forward(self, x):
        """
        Return logits (before softmax).
        """
        return self.net(x)

    def feature(self, x):
        """
        Return features just before the final fully-connected layer.
        Typical ResNet: apply conv1->bn1->relu->maxpool->layer1..layer4->avgpool, then flatten.
        """
        # If your ResNet implementation exposes these attributes, use them directly:
        out = self.net.conv1(x)
        out = self.net.bn1(out)
        out = self.net.relu(out)
        out = self.net.maxpool(out)

        out = self.net.layer1(out)
        out = self.net.layer2(out)
        out = self.net.layer3(out)
        out = self.net.layer4(out)

        out = self.net.avgpool(out)            # shape (B, C, 1, 1)
        out = torch.flatten(out, 1)            # shape (B, C)
        return out
