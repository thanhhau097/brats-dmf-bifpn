import torch

from models.unetpp.DMFNet_pp import DMFNet_pp
from models.unetpp.DMFNet_ppd import DMFNet_ppd
from models.unetpp.DMFNet_fullpp import DMFNet_fullpp
from models.attention_unet.DMFNet_attention import DMFNet_attention
from models import DMFNet_csse
from models import DMFNet_pe, DMFNet_multiattention, DMFNet_attention, DMFNet_singleattention, DMFNet_separate_inputs,\
    DMFNet_pp_double, DMFNet_bifpn, DMFNet_multiscale_weight, DMFNet_interconnect_multiscale_weight, BiFPNNet,\
    DMFNet_multiscale, BiFPNNet_deepvision


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    # x = torch.rand((1, 4, 128, 128, 128))  # [bsize,channels,Height,Width,Depth]
    # for unit in ['concatenate', 'add']:
    #     for layer in [1, 2, 3]:
    #         model = BiFPNNet(n_layers=layer, c=4, n=32, groups=16, channels=64, norm='sync_bn', num_classes=4, bifpn_unit=unit)
    #         # y = model(x)
    #         # print(y.shape)
    #         print(count_parameters(model))

    # model = DMFNet_multiscale(c=4, n=32, groups=16, channels=128, norm='sync_bn', num_classes=4)
    # y = model(x)BiFPNNet_deepvision
    # print(y.shape)
    # print(count_parameters(model))

    x = torch.rand((1, 4, 64, 64, 64))
    model = BiFPNNet_deepvision(n_layers=1, c=4, n=32, groups=16, channels=64, norm='sync_bn', num_classes=4, bifpn_unit='concatenate')
    print(count_parameters(model))
    # y = model(x)
    # print(y.size())
    # loss = torch.sum(y)
    # loss.backward()
