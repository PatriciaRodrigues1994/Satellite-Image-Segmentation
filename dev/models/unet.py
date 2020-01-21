import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

# class Interpolate(nn.Module):
#     def __init__(self, size, mode):
#         super(Interpolate, self).__init__()
#         self.interp = nn.functional.interpolate
#         self.size = size
#         self.mode = mode
        
#     def forward(self, x):
#         x = self.interp(x, size=self.size, mode=self.mode, align_corners=False)
#         return x
    
    
# class ConvBnRelu(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, padding, stride):
#         super(ConvBnRelu, self).__init__()
#         self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride)
#         self.bn = nn.BatchNorm2d(out_channels)
#         self.relu = nn.ReLU()

#     def forward(self, x):
#         x = self.conv(x)
#         x = self.bn(x)
#         x = self.relu(x)
#         return x


# class StackEncoder(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(StackEncoder, self).__init__()
#         self.convr1 = ConvBnRelu(in_channels, out_channels, kernel_size=(3, 3), stride=1, padding=0)
#         self.convr2 = ConvBnRelu(out_channels, out_channels, kernel_size=(3, 3), stride=1, padding=0)
#         self.maxPool = nn.MaxPool2d(kernel_size=(2, 2), stride=2)

#     def forward(self, x):
#         x = self.convr1(x)
#         x = self.convr2(x)
#         x_trace = x
#         x = self.maxPool(x)
#         return x, x_trace


# class StackDecoder(nn.Module):
#     def __init__(self, in_channels, out_channels, upsample_size):
#         super(StackDecoder, self).__init__()


#         self.upSample = Interpolate(size=upsample_size, mode="bilinear")
#         self.convr1 = ConvBnRelu(in_channels, out_channels, kernel_size=(3, 3), stride=1, padding=0)
#         # Crop + concat step between these 2
#         self.convr2 = ConvBnRelu(in_channels, out_channels, kernel_size=(3, 3), stride=1, padding=0)

#     def _crop_concat(self, upsampled, bypass):
#         """
#          Crop y to the (h, w) of x and concat them.
#          Used for the expansive path.
#         Returns:
#             The concatenated tensor
#         """
#         c = (bypass.size()[2] - upsampled.size()[2]) // 2
#         bypass = F.pad(bypass, (-c, -c, -c, -c))

#         return torch.cat((upsampled, bypass), 1)

#     def forward(self, x, down_tensor):
#         x = self.upSample(x)
#         x = self.convr1(x)
#         x = self._crop_concat(x, down_tensor)
#         x = self.convr2(x)
#         return x


# class UNetOriginal(nn.Module):
#     def __init__(self, in_shape):
#         super(UNetOriginal, self).__init__()
#         channels, height, width = in_shape

#         self.down1 = StackEncoder(channels, 64)
#         self.down2 = StackEncoder(64, 128)
#         self.down3 = StackEncoder(128, 256)
#         self.down4 = StackEncoder(256, 512)

#         self.center = nn.Sequential(
#             ConvBnRelu(512, 1024, kernel_size=(3, 3), stride=1, padding=0),
#             ConvBnRelu(1024, 1024, kernel_size=(3, 3), stride=1, padding=0)
#         )

#         self.up1 = StackDecoder(in_channels=1024, out_channels=512, upsample_size=(56, 56))
#         self.up2 = StackDecoder(in_channels=512, out_channels=256, upsample_size=(104, 104))
#         self.up3 = StackDecoder(in_channels=256, out_channels=128, upsample_size=(200, 200))
#         self.up4 = StackDecoder(in_channels=128, out_channels=64, upsample_size=(304, 304))

#         # 1x1 convolution at the last layer
#         # Different from the paper is the output size here
#         self.output_seg_map = nn.Conv2d(64, 1, kernel_size=(1, 1), padding=0, stride=1)

#     def forward(self, x):
#         print("input shape forward : ", x.shape)
#         x, x_trace1 = self.down1(x)  # Calls the forward() method of each layer
        
#         x, x_trace2 = self.down2(x)
#         x, x_trace3 = self.down3(x)
#         x, x_trace4 = self.down4(x)
#         print("down4 forward : ", x.shape)
#         x = self.center(x)
#         print("centre forward : ", x.shape)
#         x = self.up1(x, x_trace4)
#         x = self.up2(x, x_trace3)
#         x = self.up3(x, x_trace2)
#         x = self.up4(x, x_trace1)
#         print("up4 forward : ", x.shape)
#         out = self.output_seg_map(x)
#         print("output_seg_map forward : ",out.shape)
#         out = torch.squeeze(out, dim=1)
#         print("squeeze forward : ",out.shape)
#         return out

def freezing_pretrained_layers(model = None, freeze = True):
    """
    Freeze or Unfreeze the pretrained Portion of Unet
    """
    ### Method 1
#     for each in [model.conv1, model.conv2]:
#         for i in [0,2]:
#             each[i].weight.requires_grad = freeze
#             each[i].bias.requires_grad = freeze

#     for each in [model.conv3, model.conv4, model.conv5]:
#         for i in [0,2,4]:
#             each[i].weight.requires_grad = freeze
#             each[i].bias.requires_grad = freeze
     ### Method 2
    for each in model.encoder:
        if each._get_name() == 'Conv2d':
            each.weight.requires_grad = freeze
            each.bias.requires_grad = freeze
            
    return model


def conv3x3(in_, out):
    return nn.Conv2d(in_, out, 3, padding=1)


class ConvRelu(nn.Module):
    def __init__(self, in_, out):
        super().__init__()
        self.conv = conv3x3(in_, out)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x

class Interpolate(nn.Module):
    def __init__(self, size=None, scale_factor=None, mode='nearest', align_corners=False):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.size = size
        self.mode = mode
        self.scale_factor = scale_factor
        self.align_corners = align_corners
        
    def forward(self, x):
        x = self.interp(x, size=self.size, scale_factor=self.scale_factor, 
                        mode=self.mode, align_corners=self.align_corners)
        return x

class DecoderBlockV2(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, is_deconv=True):
        super(DecoderBlockV2, self).__init__()
        self.in_channels = in_channels

        if is_deconv:
            """
                Paramaters for Deconvolution were chosen to avoid artifacts, following
                link https://distill.pub/2016/deconv-checkerboard/
            """

            self.block = nn.Sequential(
                ConvRelu(in_channels, middle_channels),
                nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=4, stride=2,
                                   padding=1),
                nn.ReLU(inplace=True)
            )
        else:
            self.block = nn.Sequential(
                Interpolate(scale_factor=2, mode='bilinear'),
                ConvRelu(in_channels, middle_channels),
                ConvRelu(middle_channels, out_channels),
            )

    def forward(self, x):
        return self.block(x)



class UNet16(nn.Module):
    def __init__(self, num_classes=1, num_filters=32, pretrained=False, is_deconv=False):
        """
        :param num_classes:
        :param num_filters:
        :param pretrained:
            False - no pre-trained network used
            True - encoder pre-trained with VGG16
        :is_deconv:
            False: bilinear interpolation is used in decoder
            True: deconvolution is used in decoder
        """
        super().__init__()
        self.num_classes = num_classes

        self.pool = nn.MaxPool2d(2, 2)

        self.encoder = torchvision.models.vgg16(pretrained=pretrained).features

        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Sequential(self.encoder[0],
                                   self.relu,
                                   self.encoder[2],
                                   self.relu)

        self.conv2 = nn.Sequential(self.encoder[5],
                                   self.relu,
                                   self.encoder[7],
                                   self.relu)

        self.conv3 = nn.Sequential(self.encoder[10],
                                   self.relu,
                                   self.encoder[12],
                                   self.relu,
                                   self.encoder[14],
                                   self.relu)

        self.conv4 = nn.Sequential(self.encoder[17],
                                   self.relu,
                                   self.encoder[19],
                                   self.relu,
                                   self.encoder[21],
                                   self.relu)

        self.conv5 = nn.Sequential(self.encoder[24],
                                   self.relu,
                                   self.encoder[26],
                                   self.relu,
                                   self.encoder[28],
                                   self.relu)

        self.center = DecoderBlockV2(512, num_filters * 8 * 2, num_filters * 8, is_deconv)

        self.dec5 = DecoderBlockV2(512 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.dec4 = DecoderBlockV2(512 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.dec3 = DecoderBlockV2(256 + num_filters * 8, num_filters * 4 * 2, num_filters * 2, is_deconv)
        self.dec2 = DecoderBlockV2(128 + num_filters * 2, num_filters * 2 * 2, num_filters, is_deconv)
        self.dec1 = ConvRelu(64 + num_filters, num_filters)
        self.final = nn.Conv2d(num_filters, num_classes, kernel_size=1)

    def forward(self, x):
        
        conv1 = self.conv1(x)
        conv2 = self.conv2(self.pool(conv1))
        conv3 = self.conv3(self.pool(conv2))
        conv4 = self.conv4(self.pool(conv3))
        conv5 = self.conv5(self.pool(conv4))
        
        center = self.center(self.pool(conv5))
        
        dec5 = self.dec5(torch.cat([center, conv5], 1))

        dec4 = self.dec4(torch.cat([dec5, conv4], 1))
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        dec1 = self.dec1(torch.cat([dec2, conv1], 1))
       
        if self.num_classes > 1:
            x_out = F.log_softmax(self.final(dec1), dim=1)
        else:
            x_out = self.final(dec1)
        x_out = torch.squeeze(x_out, dim=1)
        
        return x_out
