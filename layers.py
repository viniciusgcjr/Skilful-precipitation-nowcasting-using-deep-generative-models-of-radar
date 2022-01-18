import torch
import torch.nn as nn
import torch.nn.utils.spectral_norm as SN



def Conv2D(in_channels,out_channels,kernel_size,padding):
    # 3x3x3 convolution with padding
    return nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=1,
        padding=padding)

def Conv3D(in_channels ,out_channels ):
    return nn.Conv3d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        stride=1,
        padding=1)


def SNConv3D(in_channels ,out_channels,kernel_size,padding):
    return SN(nn.Conv3d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=1,
        padding=padding
    ),eps=1e-4)


def SNConv2D(in_channels,out_channels,kernel_size,padding):
    return SN(nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=1,
        padding=padding
    ),eps=1e-4)


def downsample_avg_pool3d():

    return nn.AvgPool3d((2,2,2), stride=(2,2,2))



def downsample_avg_pool():
    pass
    return nn.AvgPool2d((2,2), stride=(2,2))


def upsample_nearest_neighbor(upsample_size):
    return nn.Upsample(
        scale_factor=upsample_size,
        mode='nearest')


def BatchNorm(num_features):
    return torch.nn.BatchNorm1d(num_features)



def Linear(in_features,out_features):
    return torch.nn.Linear(in_features, out_features)




if __name__ == "__main__":
 input=torch.randn(16,8,1,128,128)
 output=SpatialDiscriminator()(input)
 print('output.shape',output.shape)
 # pool of square window of size=3, stride=2
 m = nn.AvgPool2d(3, stride=2)
 # pool of non-square window
 m = nn.AvgPool3d((2,2,2), stride=(2,2,2))

 input = torch.randn(20, 22, 4, 128, 128)
 output = m(input)
 print(output.shape)
