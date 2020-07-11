import torch
from torch import nn

from models.DMFNet_16x import MFunit, DMFUnit


class BiFPNUnit(nn.Module):
    def __init__(self, n=32, channels=128, groups=16, norm='bn', base_unit=MFunit, **kwargs):
        super(BiFPNUnit, self).__init__()

        self.conv21 = base_unit(3 * channels, channels, g=groups, stride=1, norm=norm)
        self.conv31 = base_unit(4 * channels, 2 * channels, g=groups, stride=1, norm=norm)

        self.conv12 = base_unit(n + channels, n, g=groups, stride=1, norm=norm)
        self.conv22 = base_unit(n + 2 * channels, channels, g=groups, stride=1, norm=norm)
        self.conv32 = base_unit(5 * channels, 2 * channels, g=groups, stride=1, norm=norm)
        self.conv42 = base_unit(4 * channels, 2 * channels, g=groups, stride=1, norm=norm)

        self.upsample21 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.upsample32 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.upsample43 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)

        self.downsample12 = nn.MaxPool3d(kernel_size=2)
        self.downsample23 = nn.MaxPool3d(kernel_size=2)
        self.downsample34 = nn.MaxPool3d(kernel_size=2)

    def forward(self, x1, x2, x3, x4):
        x31 = torch.cat([x3, self.upsample43(x4)], dim=1)
        x31 = self.conv31(x31)

        x21 = torch.cat([x2, self.upsample32(x31)], dim=1)
        x21 = self.conv21(x21)

        x12 = torch.cat([x1, self.upsample21(x21)], dim=1)
        x12 = self.conv12(x12)

        x22 = torch.cat([x2, x21, self.downsample12(x12)], dim=1)
        x22 = self.conv22(x22)

        x32 = torch.cat([x3, x31, self.downsample23(x22)], dim=1)
        x32 = self.conv32(x32)

        x42 = torch.cat([x4, self.downsample34(x32)], dim=1)
        x42 = self.conv42(x42)

        return x12, x22, x32, x42


class BiFPNAddUnit(nn.Module):
    def __init__(self, channels=128, groups=16, norm='bn', base_unit=MFunit, **kwargs):
        super(BiFPNAddUnit, self).__init__()

        self.conv21 = base_unit(channels, channels, g=groups, stride=1, norm=norm)
        self.conv31 = base_unit(channels, channels, g=groups, stride=1, norm=norm)

        self.conv12 = base_unit(channels, channels, g=groups, stride=1, norm=norm)
        self.conv22 = base_unit(channels, channels, g=groups, stride=1, norm=norm)
        self.conv32 = base_unit(channels, channels, g=groups, stride=1, norm=norm)
        self.conv42 = base_unit(channels, channels, g=groups, stride=1, norm=norm)

        self.upsample21 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.upsample32 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.upsample43 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)

        self.downsample12 = nn.MaxPool3d(kernel_size=2)
        self.downsample23 = nn.MaxPool3d(kernel_size=2)
        self.downsample34 = nn.MaxPool3d(kernel_size=2)

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.w31_1 = torch.tensor(1.0, dtype=torch.float, requires_grad=True).to(device)
        self.w31_2 = torch.tensor(1.0, dtype=torch.float, requires_grad=True).to(device)

        self.w21_1 = torch.tensor(1.0, dtype=torch.float, requires_grad=True).to(device)
        self.w21_2 = torch.tensor(1.0, dtype=torch.float, requires_grad=True).to(device)

        self.w12_1 = torch.tensor(1.0, dtype=torch.float, requires_grad=True).to(device)
        self.w12_2 = torch.tensor(1.0, dtype=torch.float, requires_grad=True).to(device)

        self.w22_1 = torch.tensor(1.0, dtype=torch.float, requires_grad=True).to(device)
        self.w22_2 = torch.tensor(1.0, dtype=torch.float, requires_grad=True).to(device)
        self.w22_3 = torch.tensor(1.0, dtype=torch.float, requires_grad=True).to(device)

        self.w32_1 = torch.tensor(1.0, dtype=torch.float, requires_grad=True).to(device)
        self.w32_2 = torch.tensor(1.0, dtype=torch.float, requires_grad=True).to(device)
        self.w32_3 = torch.tensor(1.0, dtype=torch.float, requires_grad=True).to(device)

        self.w42_1 = torch.tensor(1.0, dtype=torch.float, requires_grad=True).to(device)
        self.w42_2 = torch.tensor(1.0, dtype=torch.float, requires_grad=True).to(device)

    def forward(self, x1, x2, x3, x4):
        x31 = self.w31_1 * x3 + self.w31_2 * self.upsample43(x4)
        x31 = self.conv31(x31)

        x21 = self.w21_1 * x2 + self.w21_2 * self.upsample32(x31)
        x21 = self.conv21(x21)

        x12 = self.w12_1 * x1 + self.w12_2 * self.upsample21(x21)
        x12 = self.conv12(x12)

        x22 = self.w22_1 * x2 + self.w22_2 * x21 + self.w22_3 * self.downsample12(x12)
        x22 = self.conv22(x22)

        x32 = self.w32_1 * x3 + self.w32_2 * x31 + self.w32_3 * self.downsample23(x22)
        x32 = self.conv32(x32)

        x42 = self.w42_1 * x4 + self.w42_2 * self.downsample34(x32)
        x42 = self.conv42(x42)

        return x12, x22, x32, x42


class BiFPN(nn.Module):
    def __init__(self, n_layers=1, n=32, channels=128, groups=16, norm='bn', base_unit=MFunit,
                 bifpn_unit='concatenate', **kwargs):
        super(BiFPN, self).__init__()
        if bifpn_unit == 'concatenate':
            self.bifpn_unit = BiFPNUnit
        elif bifpn_unit == 'add':
            self.bifpn_unit = BiFPNAddUnit
        else:
            raise ValueError('bifpn_unit must be concatenate or add')

        self.n_layers = n_layers
        self.biFPNs = nn.ModuleList()
        for _ in range(self.n_layers):
            self.biFPNs.append(self.bifpn_unit(n=n, channels=channels, groups=groups, norm=norm, base_unit=base_unit))

    def forward(self, x1, x2, x3, x4):
        for i in range(self.n_layers):
            x1, x2, x3, x4 = self.biFPNs[i](x1, x2, x3, x4)

        return x1, x2, x3, x4
