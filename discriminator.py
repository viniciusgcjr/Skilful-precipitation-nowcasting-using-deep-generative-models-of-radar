import torch
import torch.nn as nn
import layers
from torch.nn.modules.pixelshuffle import PixelUnshuffle
import torch.nn.functional as F
import random

class DBlock(nn.Module):

  def __init__(self, input_channels,output_channels, kernel_size=3, downsample=True,
               pre_activation=True, conv=layers.Conv2D,
               pooling=layers.downsample_avg_pool, activation=F.relu):
    super(DBlock, self).__init__()

    self._input_channels = input_channels
    self._output_channels = output_channels
    self._kernel_size = kernel_size
    self._downsample = downsample
    self._pre_activation = pre_activation

    self._conv1x1 = conv(in_channels=input_channels,
                    out_channels=output_channels,
                    kernel_size=1,
                    padding=0)

    self.first_conv3x3 = conv(in_channels=input_channels,
                      out_channels=output_channels,
                      kernel_size=self._kernel_size,
                      padding=1)

    self.last_conv3x3 = conv(in_channels=output_channels,
                      out_channels=output_channels,
                      kernel_size=self._kernel_size,
                      padding=1)

    self._pooling = pooling()
    self._activation = activation



    

  def forward(self, inputs):
    h0 = inputs

    # Pre-activation.
    if self._pre_activation:
      h0 = self._activation(h0)

    # First convolution.
    if len(h0.size()) == 4:
      input_channels = h0.shape[-3]
    if len(h0.size()) == 5:
      input_channels = h0.shape[-4]

    h1 = self.first_conv3x3(h0)

    h1 = self._activation(h1)

    # Second convolution.
    h2 = self.last_conv3x3(h1)

    # Downsampling.
    if self._downsample:
      h2 = self._pooling(h2)

    # The residual connection, make sure it has the same dimensionality
    # with additional 1x1 convolution and downsampling if needed.
    if input_channels != self._output_channels or self._downsample:
      sc = self._conv1x1(inputs)
      if self._downsample:
        sc = self._pooling(sc)
    else:
      sc = inputs

    # Residual connection.
    return h2 + sc



class TemporalDiscriminator(nn.Module):
  """Spatial Discriminator."""

  def __init__(self):
    super(TemporalDiscriminator, self).__init__()
    self.space_to_depth = PixelUnshuffle(downscale_factor=2)
    self.relu = nn.ReLU(inplace=True)

    self.DBlock1_3D = DBlock(input_channels=4,output_channels=48, conv=layers.SNConv3D, 
                  pooling=layers.downsample_avg_pool3d,
                  pre_activation=False)

    self.DBlock2_3D = DBlock(input_channels=48,output_channels=96, conv=layers.SNConv3D,
                             pooling=layers.downsample_avg_pool3d)

    self.DBlock1_2D = DBlock(input_channels=96,output_channels=192)
    self.DBlock2_2D = DBlock(input_channels=192,output_channels=384)
    self.DBlock3_2D = DBlock(input_channels=384,output_channels=768)
    

  def forward(self, frames):
    """Build the temporal discriminator.

    Args:
      frames: a tensor with a complete observation [b, ts, 128, 128, 1]

    Returns:
      A tensor with discriminator loss scalars [b].
    """
    b, ts, cs, hs, ws  = frames.shape

    new_hs = int(hs/2)
    new_ws = int(ws/2)
    new_cs = cs * 4

    # Process each of the ti inputs independently.
    frames = frames.reshape( b*ts, cs, hs, ws)

    # Space-to-depth stacking from 128x128x1 to 64x64x4.

    frames = self.space_to_depth(frames)

    # Stack back to sequences of length ti.
    frames = frames.reshape( b, new_cs,ts, new_hs, new_ws)



    # Two residual 3D Blocks to halve the resolution of the image, double
    # the number of channels, and reduce the number of time steps.
    y = self.DBlock1_3D(frames)

    y = self.DBlock2_3D(y)



    # Get t < ts, h, w, and c, as we have downsampled in 3D.
    _, c, t, h, w  = y.shape


    # Process each of the t images independently.
    # b t h w c -> (b x t) h w c

    y = y.reshape( -1, c, h, w)

    # Three residual D Blocks to halve the resolution of the image and double
    # the number of channels.

    y = self.DBlock1_2D(y)
    y = self.DBlock2_2D(y)
    y = self.DBlock3_2D(y)

    # One more D Block without downsampling or increase in number of channels.
    y = DBlock(input_channels=768,output_channels=768, downsample=False)(y)

    # Sum-pool the representations and feed to spectrally normalized lin. layer.
    y = self.relu(y)
    y = torch.sum(y, dim=2)
    y = torch.sum(y, dim=2)

    # y = tf.reduce_sum(nn.relu(y), axis=[1, 2])
    y = layers.BatchNorm(num_features=y.shape[1])(y)
    in_features=y.shape[1]
    output_layer = layers.Linear(in_features=in_features,out_features=1)
    output = output_layer(y)

    # Take the sum across the t samples. Note: we apply the ReLU to
    # (1 - score_real) and (1 + score_generated) in the loss.
    output = output.reshape(b, t, 1)
    scores  = torch.sum(output , dim=1, keepdim=True)
    return scores


class SpatialDiscriminator(nn.Module):
  """Spatial Discriminator."""

  def __init__(self):
    super(SpatialDiscriminator, self).__init__()
    self.space2depth = PixelUnshuffle(downscale_factor=2)
    self.relu = nn.ReLU(inplace=True)

    self.DBlock1 = DBlock(input_channels=4,output_channels=48, pre_activation=False)
    self.DBlock2 = DBlock(input_channels=48,output_channels=96)
    self.DBlock3 = DBlock(input_channels=96,output_channels=192)
    self.DBlock4 = DBlock(input_channels=192,output_channels=384)
    self.DBlock5 = DBlock(input_channels=384,output_channels=768)
    self.DBlock6 = DBlock(input_channels=768,output_channels=768, downsample=False)


  def forward(self, frames):
    """Build the spatial discriminator.

    Args:
      frames: a tensor with a complete observation [b, n, 128, 128, 1].

    Returns:
      A tensor with discriminator loss scalars [b].
    """
    b, n, c, h, w = frames.shape

    # Process each of the n inputs independently.
    frames = frames.reshape( b * n, c, h, w )

    # Space-to-depth stacking from 128x128x1 to 64x64x4.
    frames = self.space2depth(frames)
    print(torch.is_tensor(frames.is_cuda),'判断是否是tensor')

    # Five residual D Blocks to halve the resolution of the image and double
    # the number of channels.
    y = self.DBlock1(frames)
    y = self.DBlock2(y)
    y = self.DBlock3(y)
    y = self.DBlock4(y)
    y = self.DBlock5(y)

    # One more D Block without downsampling or increase in number of channels.
    y = self.DBlock6(y)
    y = self.relu(y)
    y = torch.sum(y, dim=2)
    y = torch.sum(y, dim=2)
    # Sum-pool the representations and feed to spectrally normalized lin. layer.
    # y = torch.sum(nn.ReLU(y), dim=(2, 3))

    y = layers.BatchNorm(y.shape[1])(y)
    in_features=y.shape[1]
    output_layer = layers.Linear(in_features=in_features,out_features=1)
    output = output_layer(y)

    # Take the sum across the t samples. Note: we apply the ReLU to
    # (1 - score_real) and (1 + score_generated) in the loss.
    output = output.reshape(b, n, 1)
    output = torch.sum(output,dim=1,keepdim=True)
    return output



class Discriminator(nn.Module):
  """Discriminator."""

  def __init__(self):
    super(Discriminator, self).__init__()
    """Constructor."""
    # Number of random time steps for the spatial discriminator.
    self._num_spatial_frames = 8
    # Input size ratio with respect to crop size for the temporal discriminator.
    self._temporal_crop_ratio = 2
    # As the input is the whole sequence of the event (including conditioning
    # frames), the spatial discriminator needs to pick only the t > T+0.
    self._num_conditioning_frames = 4
    self._spatial_discriminator = SpatialDiscriminator()
    self._temporal_discriminator = TemporalDiscriminator()

  def forward(self, frames):
    """Build the discriminator.

    Args:
      frames: a tensor with a complete observation [b, 22, 256, 256, 1].

    Returns:
      A tensor with discriminator loss scalars [b, 2].
    """
    b, t, c, h, w = frames.shape
    print(b,t,c,h,w,'btchw')

    # Prepare the frames for spatial discriminator: pick 8 random time steps out
    # of 18 lead time steps, and downsample from 256x256 to 128x128.
    # target_frames_sel = tf.range(self._num_conditioning_frames, t)
    # target_frames_sel = torch.arange(self._num_conditioning_frames, t)
    # print('target_frames_sel',target_frames_sel)
    permutation = torch.stack([
        (torch.randperm(18) + 4)[:self._num_spatial_frames]
        for _ in range(b*2)
    ], 0)
    print(permutation,'permutation')
    print(b,'b')
    # 有问题有问题有问题有问题有问题有问题有问题有问题有问题有问题有问题有问题有问题有问题有问题有问题
    # 有问题有问题有问题有问题有问题有问题有问题有问题有问题有问题有问题有问题有问题有问题有问题有问题
    frames_for_sd = torch.randn(b, self._num_spatial_frames, c, h, w).cuda()
    for i in range(b):
      frames_for_sd[i] = frames[i,permutation[i]]

    #有问题有问题有问题有问题有问题有问题有问题有问题有问题有问题有问题有问题有问题有问题有问题有问题
    # 有问题有问题有问题有问题有问题有问题有问题有问题有问题有问题有问题有问题有问题有问题有问题有问题
    # 有问题有问题有问题有问题有问题有问题有问题有问题有问题有问题有问题有问题有问题有问题有问题有问题
    # frames_for_sd = tf.layers.average_pooling3d(
    #     frames_for_sd, [1, 2, 2], [1, 2, 2], data_format='channels_last')
    frames_for_sd = nn.AvgPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))(frames_for_sd)
    # Compute the average spatial discriminator score for each of 8 picked time
    # steps.
    sd_out = self._spatial_discriminator(frames_for_sd)

    # Prepare the frames for temporal discriminator: choose the offset of a
    # random crop of size 128x128 out of 256x256 and pick full sequence samples.
    cr = self._temporal_crop_ratio

    h_offset = torch.randint(0, (cr - 1) * (h // cr),([]))
    w_offset = torch.randint(0, (cr - 1) * (w // cr),([]))
    frames_for_td=frames[:,:,:,h_offset:h_offset+h//cr,w_offset:w_offset+w//cr]




    # Compute the average temporal discriminator score over length 5 sequences.
    td_out = self._temporal_discriminator(frames_for_td)

    return torch.cat((sd_out, td_out), 1)




if __name__ == "__main__":
 input=torch.randn(16,22,1,128,128)
 output=TemporalDiscriminator()(input)
 print('output.shape',output.shape)
 #
 # input = torch.randn(16, 8, 1, 128, 128)
 # output = SpatialDiscriminator()(input)
 # print('output.shape', output.shape)

 # input = torch.randn(16, 22, 1, 128, 128)
 # output = Discriminator()(input)
 # parameters=Discriminator().parameters()
 # print('output.shape', output.shape)

 # frames_for_sd = torch.randn(16, 8, 1, 128, 128)
 # frames_for_sd = nn.AvgPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))(frames_for_sd)
 # print(frames_for_sd.shape)

 input = torch.randn(20, 22, 11, 64, 64)
 y = DBlock(input_channels=22,output_channels=48, conv=layers.SNConv3D,
               pooling=layers.downsample_avg_pool3d,
               pre_activation=False)(input)
 print(y.shape)

