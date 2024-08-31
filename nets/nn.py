import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
from torchsummary import summary

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
    
class Conv(nn.Module):
    def __init__(self, in_ch, out_ch, k=1, p=0, s=1, d=1, g=1, act=True):
        super(Conv, self).__init__()
        if act:
            self.convs = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, k, stride=s, padding=p, dilation=d, groups=g),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(0.1, inplace=True)
            )
        else:
            self.convs = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, k, stride=s, padding=p, dilation=d, groups=g),
                nn.BatchNorm2d(out_ch)
            )

    def forward(self, x):
        return self.convs(x)

class reorg_layer(nn.Module):
    def __init__(self, stride):
        super(reorg_layer, self).__init__()
        self.stride = stride

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        _height, _width = height // self.stride, width // self.stride
        
        x = x.view(batch_size, channels, _height, self.stride, _width, self.stride).transpose(3, 4).contiguous()
        x = x.view(batch_size, channels, _height * _width, self.stride * self.stride).transpose(2, 3).contiguous()
        x = x.view(batch_size, channels, self.stride * self.stride, _height, _width).transpose(1, 2).contiguous()
        x = x.view(batch_size, -1, _height, _width)

        return x
    

# class DetNet(nn.Module):
#     # no expansion
#     # dilation = 2
#     # type B use 1x1 conv
#     expansion = 1

#     def __init__(self, in_planes, planes, stride=1, block_type='A'):
#         super(DetNet, self).__init__()
#         self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=2, bias=False, dilation=2)
#         self.bn2 = nn.BatchNorm2d(planes)
#         self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
#         self.bn3 = nn.BatchNorm2d(self.expansion * planes)

#         self.downsample = nn.Sequential()
#         if stride != 1 or in_planes != self.expansion * planes or block_type == 'B':
#             self.downsample = nn.Sequential(
#                 nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(self.expansion * planes)
#             )

#     def forward(self, x):
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = F.relu(self.bn2(self.conv2(out)))
#         out = self.bn3(self.conv3(out))
#         out += self.downsample(x)
#         out = F.relu(out)
#         return out

import torchvision.models as models

class ResNet(nn.Module):
    # zero_init_residual 파라미터 추가
    def __init__(self, block, layers, zero_init_residual=False):
        super(ResNet, self).__init__()
        self.in_planes = 64
        

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        # self.layer5 = self._make_layer(block, 512, layers[3], stride=2)
        #self.layer5 = self._make_detnet_layer(in_channels=2048)
        # self.avgpool = nn.AvgPool2d(14) #fit 448 input size
        # self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.conv_end1 = nn.Conv2d(1536, 1024, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_end1 = nn.BatchNorm2d(1024)
        self.conv_end2 = nn.Conv2d(1024, 30, kernel_size=1, stride=1, bias=False)
        self.bn_end2 = nn.BatchNorm2d(30)
        #````````````````````````````````````````````````````````
        # head
        self.convsets_1 = nn.Sequential(
            Conv(2048, 1024, k=1),
            Conv(1024, 1024, k=3, p=1),
            Conv(1024, 1024, k=3, p=1)
        )

        # reorg
        self.route_layer = Conv(1024, 128, k=1)
        self.reorg = reorg_layer(stride=2)

        # head
        self.convsets_2 = Conv(1024+128*4, 1024, k=3, p=1)
        
        # pred
        #self.pred = nn.Conv2d(1024, self.num_anchors*(1 + 4 + self.num_classes), 1)
        #1``````````````````````````````````````````````````````````````````

        # 변경
        # He 초기화 방법
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # 변경
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        #[3, 4, 6, 3]
        layers = [block(self.in_planes, planes, stride, downsample)]
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    # def _make_detnet_layer(self, in_channels):
    #     layers = [
    #         DetNet(in_planes=in_channels, planes=256, block_type='B'),
    #         DetNet(in_planes=256, planes=256, block_type='A'),
    #         DetNet(in_planes=256, planes=256, block_type='A')
    #     ]
    #     return nn.Sequential(*layers)

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x1 = self.maxpool(x)

        x2 = self.layer1(x1)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)
        #x = self.layer5(x)
        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        # x = self.fc(x)
        #x = self.conv_end(x)
        #x = self.bn_end(x)
        #x = torch.sigmoid(x)
        # x = x.view(-1,14,14,30)
        #x = x.permute(0, 2, 3, 1)  # (-1,14,14,30)

        # reorg layer
        p5 = self.convsets_1(x5)            
        p4 = self.reorg(self.route_layer(x4))
        final = torch.cat([p4, p5], dim=1) # (1024+512, 14, 14)
        
        x = self.conv_end1(final)
        x = self.bn_end1(x)
        x = self.conv_end2(x)
        x = self.bn_end2(x)
        x = torch.sigmoid(x)
        x = x.permute(0, 2, 3, 1)
        # output = {
        #     'layer1' : x3, # 아직 쓰이지는 않지만 FPN 구현시 사용할 예정
        #     'layer2' : x4,
        #     'layer3' : x5
        # }
        return x

def resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2, pretrained=True, **kwargs):
    model_ = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        state_dict = model_zoo.load_url(weights.url)
        #print("Downloaded keys:")
        #print(state_dict.keys())  # Print the keys to verify the downloaded weights
        model_.load_state_dict(state_dict, strict=False)
    return model_
# # resnet50
# def resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2, pretrained=True, **kwargs):
#     model_ = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
#     if pretrained:
#         model_.load_state_dict(model_zoo.load_url("https://download.pytorch.org/models/resnet50-11ad3fa6.pth"), strict=False)
#     return model_

def resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V2, pretrained=True, **kwargs):
    model_ = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        state_dict = model_zoo.load_url(weights.url)
        #print("Downloaded keys:")
        #print(state_dict.keys())  # Print the keys to verify the downloaded weights
        model_.load_state_dict(state_dict, strict=False)
    return model_
#  def resnet101(pretrained=True, **kwargs):
    # model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    # if pretrained:
    #     model.load_state_dict(model_zoo.load_url("https://download.pytorch.org/models/resnet101-cd907fc2.pth"))
    # return model
def resnet152(weights=models.ResNet152_Weights.IMAGENET1K_V2, pretrained=True, **kwargs):
    model_ = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        state_dict = model_zoo.load_url(weights.url)
        #print("Downloaded keys:")
        #print(state_dict.keys())  # Print the keys to verify the downloaded weights
        model_.load_state_dict(state_dict, strict=False)
    return model_
# def resnet152(pretrained=False, **kwargs):
#     model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
#     if pretrained:
#         model.load_state_dict(model_zoo.load_url("https://download.pytorch.org/models/resnet152-f82ba261.pth"))
#     return model




    
if __name__ == '__main__':
    a = torch.randn((1, 3, 448, 448))
    model = resnet101()
    output = model(a)
    print(output.shape)
    # for k in output.keys():
    #     print('{} : {}'.format(k, output[k].shape))









# if __name__ == '__main__':
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model = resnet50().to(device)
#     summary(model, (3, 448, 448), device=str(device))
