import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F
from torchvision import models


from GT_table import _E_GT_TRAIN_150
from GT_table import _E_GT_TEST_75
from options import *
from main import gpus_list
from functions import *


# EfficientNet
class Block(nn.Module):
    """expand + depthwise + pointwise + squeeze-excitation"""

    def __init__(self, in_planes, out_planes, expansion, stride):
        super(Block, self).__init__()
        self.stride = stride

        planes = expansion * in_planes
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, groups=planes, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1,
                          stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_planes),
            )

        # SE layers
        self.fc1 = nn.Conv2d(out_planes, out_planes // 16, kernel_size=1)
        self.fc2 = nn.Conv2d(out_planes // 16, out_planes, kernel_size=1)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        shortcut = self.shortcut(x) if self.stride == 1 else out
        # Squeeze-Excitation
        w = F.avg_pool2d(out, out.size(2))
        w = F.relu(self.fc1(w))
        w = self.fc2(w).sigmoid()
        out = out * w + shortcut
        return out


# Transformer
class MDTA(nn.Module):
    def __init__(self, channels, num_heads):
        super(MDTA, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(1, num_heads, 1, 1))

        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=False)
        self.qkv_conv = nn.Conv2d(channels * 3, channels * 3, kernel_size=3, padding=1, groups=channels * 3, bias=False)
        self.project_out = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = self.qkv_conv(self.qkv(x)).chunk(3, dim=1)

        q = q.reshape(b, self.num_heads, -1, h * w)
        k = k.reshape(b, self.num_heads, -1, h * w)
        v = v.reshape(b, self.num_heads, -1, h * w)
        q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)

        attn = torch.softmax(torch.matmul(q, k.transpose(-2, -1).contiguous()) * self.temperature, dim=-1)
        out = self.project_out(torch.matmul(attn, v).reshape(b, -1, h, w))
        return out


class MDTA_cross(nn.Module):
    def __init__(self, channels, num_heads):
        super(MDTA_cross, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(1, num_heads, 1, 1))

        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=False)
        # self.qkv_down = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=False)
        self.qkv_conv = nn.Conv2d(channels * 3, channels * 3, kernel_size=3, padding=1, groups=channels * 3, bias=False)
        # self.qkv_conv_down = nn.Conv2d(channels * 3, channels * 3, kernel_size=3, padding=1, groups=channels * 3, bias=False)
        self.project_out = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

    def forward(self, x, y):
        b, c, h, w = x.shape

        q, k, v = self.qkv_conv(self.qkv(x)).chunk(3, dim=1)
        q1, k1, v1 = self.qkv_conv(self.qkv(y)).chunk(3, dim=1)

        q = q.reshape(b, self.num_heads, -1, h * w)
        k = k.reshape(b, self.num_heads, -1, h * w)
        v = v.reshape(b, self.num_heads, -1, h * w)

        q1 = q1.reshape(b, self.num_heads, -1, h * w)
        k1 = k1.reshape(b, self.num_heads, -1, h * w)
        v1 = v1.reshape(b, self.num_heads, -1, h * w)

        q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
        q1, k1 = F.normalize(q1, dim=-1), F.normalize(k1, dim=-1)

# a, b
        # a:key query value => q,k,v
        # b:key query value => q1,k1,v1
        # For cross-attention, use q, k1 * v1 OR q1, k * v
        # For self-attention, use q, k * v

        # up_feature is the sum of self and cross attention, in that order

        attn_u = torch.softmax(torch.matmul(q, k.transpose(-2, -1).contiguous()) * self.temperature, dim=-1)
        attn_ud = torch.softmax(torch.matmul(q, k1.transpose(-2, -1).contiguous()) * self.temperature, dim=-1)
        attn_d = torch.softmax(torch.matmul(q1, k1.transpose(-2, -1).contiguous()) * self.temperature, dim=-1)
        attn_du = torch.softmax(torch.matmul(q1, k.transpose(-2, -1).contiguous()) * self.temperature, dim=-1)

        cross_feature = (torch.matmul(attn_du, v) + torch.matmul(attn_ud, v1)).reshape(b, -1, h, w)
        self_feature = (torch.matmul(attn_u, v) + torch.matmul(attn_d, v1)).reshape(b, -1, h, w)

        # out = self.project_out(torch.matmul(attn, v).reshape(b, -1, h, w))

        return cross_feature, self_feature


class GDFN(nn.Module):
    def __init__(self, channels, expansion_factor):
        super(GDFN, self).__init__()

        hidden_channels = int(channels * expansion_factor)
        self.project_in = nn.Conv2d(channels, hidden_channels * 2, kernel_size=1, bias=False)
        self.conv = nn.Conv2d(hidden_channels * 2, hidden_channels * 2, kernel_size=3, padding=1,
                              groups=hidden_channels * 2, bias=False)
        self.project_out = nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=False)

    def forward(self, x):
        x1, x2 = self.conv(self.project_in(x)).chunk(2, dim=1)
        x = self.project_out(F.gelu(x1) * x2)
        return x


class TransformerBlock_cross(nn.Module):
    def __init__(self, channels, num_heads, expansion_factor):
        super(TransformerBlock_cross, self).__init__()

        self.norm1_1 = nn.LayerNorm(channels)
        self.norm1_2 = nn.LayerNorm(channels)
        self.attn = MDTA_cross(channels, num_heads)
        self.norm2 = nn.LayerNorm(channels)
        self.ffn = GDFN(channels, expansion_factor)

    def forward(self, x, y):
        b, c, h, w = x.shape
        # print(np.shape(x), np.shape(y))
        out, _ =self.attn(self.norm1_1(x.reshape(b, c, -1).transpose(-2, -1).contiguous()).transpose(-2, -1)
                          .contiguous().reshape(b, c, h, w), self.norm1_2(y.reshape(b, c, -1).transpose(-2, -1).contiguous()).transpose(-2, -1)
                          .contiguous().reshape(b, c, h, w))
        x = x + out
        x = x + self.ffn(self.norm2(x.reshape(b, c, -1).transpose(-2, -1).contiguous()).transpose(-2, -1)
                         .contiguous().reshape(b, c, h, w))
        return x


class TransformerBlock(nn.Module):
    def __init__(self, channels, num_heads, expansion_factor):
        super(TransformerBlock, self).__init__()

        self.norm = nn.LayerNorm(channels)
        self.attn = MDTA(channels, num_heads)
        self.norm2 = nn.LayerNorm(channels)
        self.ffn = GDFN(channels, expansion_factor)

    def forward(self, x):
        b, c, h, w = x.shape
        x = x + self.attn(self.norm(x.reshape(b, c, -1).transpose(-2, -1).contiguous()).transpose(-2, -1)
                          .contiguous().reshape(b, c, h, w))
        x = x + self.ffn(self.norm2(x.reshape(b, c, -1).transpose(-2, -1).contiguous()).transpose(-2, -1)
                         .contiguous().reshape(b, c, h, w))
        return x


class DownSample(nn.Module):
    def __init__(self, channels):
        super(DownSample, self).__init__()
        self.body = nn.Sequential(nn.Conv2d(channels, channels // 2, kernel_size=3, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)


class UpSample(nn.Module):
    def __init__(self, channels):
        super(UpSample, self).__init__()
        self.body = nn.Sequential(nn.Conv2d(channels, channels * 2, kernel_size=3, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)


class EfficientNet(nn.Module):
    def __init__(self, cfg):
        super(EfficientNet, self).__init__()
        self.cfg = cfg
        self.conv1 = nn.Conv2d(3*9, 32, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(in_planes=32)
        # self.linear = nn.Linear(cfg[-1][1], num_classes)

        # self.linear = nn.Linear(15360, num_classes)
        self.conv = nn.Conv2d(192, 4, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def _make_layers(self, in_planes):
        layers = []
        for expansion, out_planes, num_blocks, stride in self.cfg:
            strides = [stride] + [1] * (num_blocks - 1)
            for stride in strides:
                layers.append(Block(in_planes, out_planes, expansion, stride))
                in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        # print(x.shape)
        # x = x.reshape(-1,30,180,240)
        out = F.relu(self.bn1(self.conv1(x)))
        # print(out.shape)
        out = self.layers(out)
        # out = out.view(out.size(0), -1)
        # print(out.shape)
        out = self.conv(out)
        # out = self.relu(out)
        out = torch.abs(out)
        (b, c, h, w) = out.shape
        out1 = torch.zeros(b, c, h, w)
        out1 = out1.cuda(gpus_list[0])

        for bb in range(b):
            for i in range(h):
                for j in range(w):
                    out1[bb, :, i, j] = out[bb, :, i, j] / (torch.norm(out[bb, :, i, j]) + 1e-04)

        return out1


def EfficientNetB0():
    # (expansion, out_planes, num_blocks, stride)
    cfg = [(1, 16, 1, 2),
           (6, 24, 2, 1),
           (6, 40, 2, 2),
           (6, 80, 3, 2),
           (6, 112, 3, 1),
           (6, 192, 4, 2),
           # (6, 320, 1, 2)
           ]
    return EfficientNet(cfg)


###Unet
def convrelu(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        nn.ReLU(inplace=True),
    )

class Unet(nn.Module):
    def __init__(self):
        super().__init__()

        self.base_model = models.resnet18(pretrained=True)
        self.base_layers = list(self.base_model.children())

        self.layer0 = nn.Sequential(*self.base_layers[:3])  # size=(N, 64, x.H/2, x.W/2)
        # self.layer0_1x1 = convrelu(64, 64, 1, 0)
        self.layer1 = nn.Sequential(*self.base_layers[3:5])  # size=(N, 64, x.H/4, x.W/4)
        # self.layer1_1x1 = convrelu(64, 64, 1, 0)
        self.layer2 = self.base_layers[5]  # size=(N, 128, x.H/8, x.W/8)
        # self.layer2_1x1 = convrelu(128, 128, 1, 0)
        self.layer3 = self.base_layers[6]  # size=(N, 256, x.H/16, x.W/16)
        self.layer3_1x1 = convrelu(256, 256, 1, 0)
        self.layer4 = self.base_layers[7]  # size=(N, 512, x.H/32, x.W/32)
        self.layer4_1x1 = convrelu(512, 512, 2, 1)

        # self.upsample = nn.Upsample(size=[23,30], mode='bilinear', align_corners=True)
        self.upsample2 = nn.Upsample(size=[12, 15], mode='bilinear', align_corners=True)

        self.conv_up3 = convrelu(256 + 512, 512, 3, 1)

        self.conv_original_size00 = convrelu(1, 3, 3, 1)

        # self.conv_last = nn.Conv2d(512, 1, 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input):
        input = input.view(-1, 1, 180, 240).float()

        input3 = self.conv_original_size00(input)

        layer0 = self.layer0(input3)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)
        # print('layer4 size :%d', layer4.size())

        layer4 = self.layer4_1x1(layer4)
        x = self.upsample2(layer4)

        layer3 = self.layer3_1x1(layer3)

        x = torch.cat([x, layer3], dim=1)
        x = self.conv_up3(x)

        # out = self.conv_last(x)
        #
        # out = torch.abs(out)
        #
        # (b, c, h, w) = out.shape
        # out1 = torch.zeros(b, c, h, w)
        # out1 = out1.cuda(gpus_list[0])
        # for bb in range(b):
        #     out1[bb, 0, :, :] = out[bb, 0, :, :] / (torch.sum(out[bb, 0, :, :]) + 1e-04)

        return x

# For phase map
class Unet2(nn.Module):
    def __init__(self):
        super().__init__()

        self.base_model = models.resnet18(pretrained=True)
        self.base_layers = list(self.base_model.children())

        self.layer0 = nn.Sequential(*self.base_layers[:3])  # size=(N, 64, x.H/2, x.W/2)
        # self.layer0_1x1 = convrelu(64, 64, 1, 0)
        self.layer1 = nn.Sequential(*self.base_layers[3:5])  # size=(N, 64, x.H/4, x.W/4)
        # self.layer1_1x1 = convrelu(64, 64, 1, 0)
        self.layer2 = self.base_layers[5]  # size=(N, 128, x.H/8, x.W/8)
        # self.layer2_1x1 = convrelu(128, 128, 1, 0)
        self.layer3 = self.base_layers[6]  # size=(N, 256, x.H/16, x.W/16)
        self.layer3_1x1 = convrelu(256, 256, 1, 0)
        self.layer4 = self.base_layers[7]  # size=(N, 512, x.H/32, x.W/32)
        self.layer4_1x1 = convrelu(512, 512, 2, 1)

        # self.upsample = nn.Upsample(size=[23,30], mode='bilinear', align_corners=True)
        self.upsample2 = nn.Upsample(size=[12, 15], mode='bilinear', align_corners=True)

        self.conv_up3 = convrelu(256 + 512, 512, 3, 1)

        self.conv_original_size00 = convrelu(3, 3, 3, 1)

        # self.conv_last = nn.Conv2d(512, 1, 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input):                               # Tensor sizes will be based on output
        input = input.view(-1, 3, 180, 240).float()         # 16, 3, 180, 240

        input3 = self.conv_original_size00(input)           # 16, 3, 180, 240
        # print(input3.size())

        layer0 = self.layer0(input3)                        # 16, 64, 90, 120
        # print(layer0.size())
        layer1 = self.layer1(layer0)                        # 16, 64, 45, 60
        # print(layer1.size())
        layer2 = self.layer2(layer1)                        # 16, 128, 23, 30
        # print(layer2.size())
        layer3 = self.layer3(layer2)                        # 16, 256, 12, 15
        # print(layer3.size())
        layer4 = self.layer4(layer3)                        # 16, 512, 6, 8
        # print(layer4.size())
        # print('layer4 size :%d', layer4.size())

        layer4 = self.layer4_1x1(layer4)                    # 16, 512, 7, 9
        # print(layer4.size())
        x = self.upsample2(layer4)                          # 16, 512, 12, 15
        # print(x.size())

        layer3 = self.layer3_1x1(layer3)                    # 16, 256, 12, 15
        # print(layer3.size())

        x = torch.cat([x, layer3], dim=1)                   # 16, 768, 12, 15
        # print(x.size())
        x = self.conv_up3(x)                                # 16, 512, 12, 15
        # print(x.size())

        # out = self.conv_last(x)
        #
        # out = torch.abs(out)
        #
        # (b, c, h, w) = out.shape
        # out1 = torch.zeros(b, c, h, w)
        # out1 = out1.cuda(gpus_list[0])
        # for bb in range(b):
        #     out1[bb, 0, :, :] = out[bb, 0, :, :] / (torch.sum(out[bb, 0, :, :]) + 1e-04)

        return x


class transformer_layer(nn.Module):
    def __init__(self):
        super().__init__()

        self.channel_cutdown = nn.Conv2d(1024, 512, kernel_size=1, bias=False)
        self.transformblock_cross = TransformerBlock_cross(512, 4, 2.66)
        self.conv_last = nn.Conv2d(512, 1, 1)

    def forward(self, x, y):
        # out = torch.cat([x, y], dim=1)
        # out = self.channel_cutdown(out)
        out = self.transformblock_cross(y, x)  # switch y(RGB), x(amp) so that RGB becomes the base for CA (CA features added to RGB features)
        out = self.conv_last(out)
        out = torch.abs(out)

        (b, c, h, w) = out.shape
        out1 = torch.zeros(b, c, h, w)
        out1 = out1.cuda(gpus_list[0])
        for bb in range(b):
            out1[bb, 0, :, :] = out[bb, 0, :, :] / (torch.sum(out[bb, 0, :, :]) + 1e-04)

        return out1


# 두개 결과 합쳐서 조명
class Result(nn.Module):
    def __init__(self):
        super(Result, self).__init__()
        self.EfficientNet = EfficientNetB0()
        self.UNet = Unet()
        self.UNet2 = Unet2()
        self.transformer_layer = transformer_layer()

    def forward(self, x, y, z):
        out1 = self.EfficientNet(x)
        out2 = self.UNet(y)
        out3 = self.UNet2(z)
        out = self.transformer_layer(out2, out3)
        # print('gradient',y[:,100,100])

        (b, c, h, w) = out1.shape
        pred = torch.zeros(b, 3)
        pred1 = torch.zeros(b, 3)
        pred = pred.cuda(gpus_list[0])
        pred1 = pred1.cuda(gpus_list[0])

        # print(out2)
        for i in range(h):
            for j in range(w):
                pred[:, 0] = pred[:, 0] + (out1[:, 0, i, j] * out[:, 0, i, j])
                pred[:, 1] = pred[:, 1] + (out1[:, 1, i, j] * out[:, 0, i, j])
                pred[:, 2] = pred[:, 2] + (out1[:, 2, i, j] * out[:, 0, i, j])

        for bb in range(b):
            pred1[bb, :] = pred[bb, :] / (torch.norm(pred[bb, :]) + 1e-04)

        return pred1, out, out1


class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir, image_int_dir):
        super(DatasetFromFolder, self).__init__()
        self.img_int_dir_name = image_int_dir

    def __getitem__(self, index):
        gt = _E_GT_TRAIN_150[index]
        input1 = input11[index]
        input2 = load_img(opt.data_int_dir + '/' + str(index+1) + '.png')           # import amplitude map
        # input3 = load_img(opt.phase_train_dir + '/' + str(index+1) + '.png')        # import phase map
        input3 = input11[index, 12:15, :, :]    # passes the 5th frame

        gt = torch.from_numpy(gt)

        return input1, input2, input3, gt

    def __len__(self):
        return 150


class DatasetFromFolderEval(data.Dataset):
    def __init__(self, lr_dir, lr_int_dir):
        super(DatasetFromFolderEval, self).__init__()
        self.lr_int_dir_name = lr_int_dir

    def __getitem__(self, index):
        gt = _E_GT_TEST_75[index]
        input1 = test11[index]
        input2 = load_img(opt.input_int_dir + '/' + str(index+1) + '.png')          # import amplitude map
        # input3 = load_img(opt.phase_eval_dir + '/' + str(index+1) + '.png')         # import phase map
        input3 = test11[index, 12:15, :, :]     # passes the 5th frame

        gt = torch.from_numpy(gt)

        return input1, input2, input3, gt, index

    def __len__(self):
        return 75


class DatasetFromFolderOutlier(data.Dataset):
    def __init__(self, lr_dir, lr_int_dir, outlier_list):
        super(DatasetFromFolderOutlier, self).__init__()
        self.lr_int_dir_name = lr_int_dir
        self.outliers = outlier_list

    def __getitem__(self, index):
        gt = _E_GT_TEST_75[opt.outlier_list[index]]     # if this doesn't work, try using just outlier_list, or use self
        input1 = test11[opt.outlier_list[index]]
        input2 = load_img(opt.input_int_dir + '/' + str(opt.outlier_list[index]+1) + '.png')          # import amplitude map
        input3 = test11[opt.outlier_list[index], 12:15, :, :]     # passes the 5th frame

        gt = torch.from_numpy(gt)

        return input1, input2, input3, gt, opt.outlier_list[index]

    def __len__(self):
        return len(opt.outlier_list)


def get_training_set(data_dir, data_int_dir):
    return DatasetFromFolder(data_dir, data_int_dir)

def get_eval_set(lr_dir, lr_int_dir):
    return DatasetFromFolderEval(lr_dir, lr_int_dir)

def get_outlier_set(lr_dir, lr_int_dir, outlier_list):
    return DatasetFromFolderOutlier(lr_dir, lr_int_dir, outlier_list)
