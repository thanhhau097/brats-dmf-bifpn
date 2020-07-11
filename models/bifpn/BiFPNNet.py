import torch
from torch import nn

try:
    from models.sync_batchnorm import SynchronizedBatchNorm3d
except:
    pass

from models.DMFNet_16x import normalization, Conv3d_Block, DilatedConv3DBlock, MFunit, DMFUnit
from models.bifpn.bifpn import BiFPN


class BiFPNNet(nn.Module):
    def __init__(self, n_layers=1, base_unit=MFunit, c=4, n=32, channels=128,
                 groups=16, norm='bn', num_classes=4, bifpn_unit='concatenate'):
        super(BiFPNNet, self).__init__()

        # Entry flow
        self.encoder_block1 = nn.Conv3d(c, n, kernel_size=3, padding=1, stride=2, bias=False)  # H//2
        self.encoder_block2 = nn.Sequential(
            DMFUnit(n, channels, g=groups, stride=2, norm=norm, dilation=[1, 2, 3]),  # H//4 down
            DMFUnit(channels, channels, g=groups, stride=1, norm=norm, dilation=[1, 2, 3]),  # Dilated Conv 3
            DMFUnit(channels, channels, g=groups, stride=1, norm=norm, dilation=[1, 2, 3])
        )

        self.encoder_block3 = nn.Sequential(
            DMFUnit(channels, channels * 2, g=groups, stride=2, norm=norm, dilation=[1, 2, 3]),  # H//8
            DMFUnit(channels * 2, channels * 2, g=groups, stride=1, norm=norm, dilation=[1, 2, 3]),  # Dilated Conv 3
            DMFUnit(channels * 2, channels * 2, g=groups, stride=1, norm=norm, dilation=[1, 2, 3])
        )

        self.encoder_block4 = nn.Sequential(  # H//8,channels*4
            MFunit(channels * 2, channels * 3, g=groups, stride=2, norm=norm),  # H//16
            MFunit(channels * 3, channels * 3, g=groups, stride=1, norm=norm),
            MFunit(channels * 3, channels * 2, g=groups, stride=1, norm=norm),
        )

        # BiFPN
        self.biFPN = BiFPN(n_layers=n_layers, c=c, n=n, channels=channels,
                           groups=groups, norm=norm, base_unit=base_unit, bifpn_unit=bifpn_unit)

        # DECODER
        self.bifpn_unit = bifpn_unit
        if bifpn_unit == 'concatenate':
            channels_list = [n, channels, channels * 2, channels * 2]
        elif bifpn_unit == 'add':
            self.convert_x1 = MFunit(n, channels, g=groups, stride=1, norm=norm)
            self.convert_x2 = MFunit(channels, channels, g=groups, stride=1, norm=norm)
            self.convert_x3 = MFunit(channels * 2, channels, g=groups, stride=1, norm=norm)
            self.convert_x4 = MFunit(channels * 2, channels, g=groups, stride=1, norm=norm)

            channels_list = [channels, channels, channels, channels]
        else:
            raise ValueError('bifpn_unit must be concatenate or add')

        self.upsample1 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)  # H//8
        self.decoder_block1 = MFunit(channels_list[3] + channels_list[2], channels_list[2],
                                     g=groups, stride=1, norm=norm)

        self.upsample2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)  # H//4
        self.decoder_block2 = MFunit(channels_list[2] + channels_list[1], channels_list[1],
                                     g=groups, stride=1, norm=norm)

        self.upsample3 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)  # H//2
        self.decoder_block3 = MFunit(channels_list[1] + channels_list[0], channels_list[0],
                                     g=groups, stride=1, norm=norm)
        self.upsample4 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)  # H
        self.seg = nn.Conv3d(channels_list[0], num_classes, kernel_size=1, padding=0, stride=1, bias=False)

        self.softmax = nn.Softmax(dim=1)

        # Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.nn.init.torch.nn.init.kaiming_normal_(m.weight)  #
            elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.GroupNorm) or isinstance(m, SynchronizedBatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Encoder
        x1 = self.encoder_block1(x)
        x2 = self.encoder_block2(x1)
        x3 = self.encoder_block3(x2)
        x4 = self.encoder_block4(x3)

        if self.bifpn_unit == 'add':
            x1 = self.convert_x1(x1)
            x2 = self.convert_x2(x2)
            x3 = self.convert_x3(x3)
            x4 = self.convert_x4(x4)

        x12, x22, x32, x42 = self.biFPN(x1, x2, x3, x4)

        # decoder
        y1 = self.upsample1(x42)
        y1 = torch.cat([x32, y1], dim=1)
        y1 = self.decoder_block1(y1)

        y2 = self.upsample2(y1)  # H//4
        y2 = torch.cat([x22, y2], dim=1)
        y2 = self.decoder_block2(y2)

        y3 = self.upsample3(y2)  # H//2
        y3 = torch.cat([x12, y3], dim=1)
        y3 = self.decoder_block3(y3)
        y4 = self.upsample4(y3)
        y4 = self.seg(y4)
        if hasattr(self, 'softmax'):
            y4 = self.softmax(y4)
        return y4
