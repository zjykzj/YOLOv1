# -*- coding: utf-8 -*-

"""
@date: 2023/3/28 下午3:04
@file: yolov1.py
@author: zj
@description: 
"""
from typing import Type, Union, List, Optional, Callable, Any

import torch
from torch import Tensor, nn
from torchvision.models._api import register_model
from torchvision.models._utils import handle_legacy_interface
from torchvision.models import resnet
from torchvision.models.resnet import Bottleneck, WeightsEnum, ResNet18_Weights, BasicBlock


class ResNet(resnet.ResNet):

    def __init__(self, block: Type[Union[BasicBlock, Bottleneck]], layers: List[int], num_classes: int = 1000,
                 zero_init_residual: bool = False, groups: int = 1, width_per_group: int = 64,
                 replace_stride_with_dilation: Optional[List[bool]] = None,
                 norm_layer: Optional[Callable[..., nn.Module]] = None, B=2, S=7) -> None:
        super().__init__(block, layers, num_classes, zero_init_residual, groups, width_per_group,
                         replace_stride_with_dilation, norm_layer)

        self.B = B
        self.S = S
        self.fc = nn.Linear(512 * block.expansion, (num_classes + 5 * self.B) * self.S * self.S)
        self.num_classes = num_classes

    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int, stride: int = 1,
                    dilate: bool = False) -> nn.Sequential:
        return super()._make_layer(block, planes, blocks, stride, dilate)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # return super()._forward_impl(x)
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x.view(-1, self.S, self.S, self.num_classes + 5 * self.B)

    def forward(self, x: Tensor) -> Tensor:
        # return super().forward(x)
        return self._forward_impl(x)


def _resnet(
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        weights: Optional[WeightsEnum],
        progress: bool,
        **kwargs: Any,
) -> ResNet:
    # if weights is not None:
    #     _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))

    model = ResNet(block, layers, **kwargs)

    if weights is not None:
        ckpt = weights.get_state_dict(progress=progress)
        ckpt.pop('fc.weight')
        ckpt.pop('fc.bias')
        model.load_state_dict(ckpt, strict=False)

    return model


@register_model()
@handle_legacy_interface(weights=("pretrained", ResNet18_Weights.IMAGENET1K_V1))
def yolov1_resnet18(*, weights: Optional[ResNet18_Weights] = None, progress: bool = True, **kwargs: Any) -> ResNet:
    """ResNet-18 from `Deep Residual Learning for Image Recognition <https://arxiv.org/pdf/1512.03385.pdf>`__.

    Args:
        weights (:class:`~torchvision.models.ResNet18_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.ResNet18_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.resnet.ResNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.ResNet18_Weights
        :members:
    """
    weights = ResNet18_Weights.verify(weights)

    return _resnet(BasicBlock, [2, 2, 2, 2], weights, progress, **kwargs)


if __name__ == '__main__':
    # weights = ResNet18_Weights.IMAGENET1K_V1
    # weights = ResNet18_Weights.verify(weights)
    #
    # model = _resnet(BasicBlock, [2, 2, 2, 2], weights, True, num_classes=20, B=2, S=7)
    # print(model)

    model = yolov1_resnet18(B=2, S=7, num_classes=20, weights=ResNet18_Weights.IMAGENET1K_V1)
    print(model)

    data = torch.randn(1, 3, 224, 224)
    print(data.shape)

    res = model(data)
    print(res.shape)
