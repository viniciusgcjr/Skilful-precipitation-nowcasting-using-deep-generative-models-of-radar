"""Generator implementation."""

import functools
import torch.nn.functional as F
import numpy as np
# from torch.nn.modules.pixelshuffle import PixelShuffle,PixelUnshuffle
import einops
import discriminator
import latent_stack
import layers
import torch
import torch.nn as nn
from  convGRU import ConvGRU






class Generator(nn.Module):
  """Generator for the proposed model."""

  def __init__(self, lead_time=90, time_delta=5):
    """Constructor.

    Args:
      lead_time: last lead time for the generator to predict. Default: 90 min.
      time_delta: time step between predictions. Default: 5 min.
    """
    super(Generator, self).__init__()
    self._cond_stack = ContextConditioningStack()
    self._sampler = Sampler(lead_time, time_delta)

  def forward(self, inputs):
    """Connect to a graph.

    Args:
      inputs: a batch of inputs on the shape [batch_size, time, h, w, 1].
    Returns:
      predictions: a batch of predictions in the form
        [batch_size, num_lead_times, h, w, 1].
    """
    _, _, _,height, width = inputs.shape
    initial_states = self._cond_stack(inputs)
    predictions = self._sampler(initial_states, [height, width])
    return predictions




class ContextConditioningStack(nn.Module):
  """Conditioning Stack for the Generator."""

  def __init__(self):
    super(ContextConditioningStack, self).__init__()
    self._block1 = discriminator.DBlock(input_channels=4,output_channels=48, downsample=True)
    # self._conv_mix1 = layers.SNConv2D(output_channels=48, kernel_size=3)
    self._conv_mix1 = layers.SNConv2D(in_channels=192,out_channels=48,kernel_size=3,padding=1)
    self._block2 = discriminator.DBlock(input_channels=48,output_channels=96, downsample=True)
    # self._conv_mix2 = layers.SNConv2D(output_channels=96, kernel_size=3)
    self._conv_mix2 = layers.SNConv2D(in_channels=384,out_channels=96,kernel_size=3,padding=1)
    self._block3 = discriminator.DBlock(input_channels=96,output_channels=192, downsample=True)
    # self._conv_mix3 = layers.SNConv2D(output_channels=192, kernel_size=3)
    self._conv_mix3 = layers.SNConv2D(in_channels=768,out_channels=192,kernel_size=3,padding=1)
    self._block4 = discriminator.DBlock(input_channels=192,output_channels=384, downsample=True)
    # self._conv_mix4 = layers.SNConv2D(output_channels=384, kernel_size=3)
    self._conv_mix4 = layers.SNConv2D(in_channels=1536,out_channels=384,kernel_size=3,padding=1)
    self._space_to_depth=torch.nn.PixelUnshuffle(downscale_factor=2)



    # self._mixing_layer=_mixing_layer()


  def forward(self, inputs):
    dataList=[]
    # Space to depth conversion of 256x256x1 radar to 128x128x4 hiddens.

    h0 = self._space_to_depth(inputs)
    steps = h0.size(1)  # Number of timesteps

    h1 = []
    h2 = []
    h3 = []
    h4 = []
    for i in range(steps):
      print(h0[:, i, :, :, :].shape,'tttttttt')
      s1 = self._block1(h0[:, i, :, :, :])
      s2 = self._block2(s1)
      s3 = self._block3(s2)
      s4 = self._block4(s3)
      h1.append(s1)
      h2.append(s2)
      h3.append(s3)
      h4.append(s4)
    h1 = torch.stack(h1, dim=1)  # B, T, C, H, W and want along C dimension
    h2 = torch.stack(h2, dim=1)  # B, T, C, H, W and want along C dimension
    h3 = torch.stack(h3, dim=1)  # B, T, C, H, W and want along C dimension
    h4 = torch.stack(h4, dim=1)


    # input_channel1=h1.shape[1]*h1.shape[2]
    # input_channel2 =h2.shape[1]*h2.shape[2]
    # input_channel3 =h3.shape[1]*h3.shape[2]
    # input_channel4 =h4.shape[1]*h4.shape[2]

    # Spectrally normalized convolutions, followed by rectified linear units.
    init_state_1 = self._mixing_layer(h1, self._conv_mix1)
    init_state_2 = self._mixing_layer(h2, self._conv_mix2)
    init_state_3 = self._mixing_layer(h3, self._conv_mix3)
    init_state_4 = self._mixing_layer(h4, self._conv_mix4)

    # Return a stack of conditioning representations of size 64x64x48, 32x32x96,
    # 16x16x192 and 8x8x384.
    return init_state_1, init_state_2, init_state_3, init_state_4


  def _mixing_layer(self, inputs, conv_block):
        # Convert from [batch_size, time, h, w, c] -> [batch_size, h, w, c * time]
        # then perform convolution on the output while preserving number of c.
        stacked_inputs = einops.rearrange(inputs, "b t c h w -> b (c t) h w")
        return F.relu(conv_block(stacked_inputs))




class Sampler(nn.Module):
  """Sampler for the Generator."""

  def __init__(self, lead_time=90, time_delta=5,latent_channels: int = 768, context_channels: int = 384, output_channels: int = 1):
    super(Sampler, self).__init__()
    self._num_predictions = lead_time // time_delta
    self._latent_stack = latent_stack.LatentConditioningStack()

    self._conv_gru4 = ConvGRU(
      input_channels=latent_channels + context_channels,
      output_channels=context_channels,
      kernel_size=3,

    )
    self._conv4 = layers.SNConv2D(in_channels=context_channels,out_channels=latent_channels, kernel_size=1,padding=0)
    self._gblock4 = GBlock(input_channels=latent_channels,output_channels=latent_channels)
    self._g_up_block4 = UpsampleGBlock(input_channels=latent_channels,output_channels=latent_channels//2)



    self._conv_gru3 = ConvGRU(
      input_channels=latent_channels // 2 + context_channels // 2,
      output_channels=context_channels // 2,
      kernel_size=3,
    )
    self._conv3 = layers.SNConv2D(in_channels=context_channels//2,out_channels=latent_channels//2, kernel_size=1,padding=0)
    self._gblock3 = GBlock(input_channels=latent_channels//2,output_channels=latent_channels//2)
    self._g_up_block3 = UpsampleGBlock(input_channels=latent_channels//2,output_channels=latent_channels//4)



    self._conv_gru2 = ConvGRU(
      input_channels=latent_channels // 4 + context_channels // 4,
      output_channels=context_channels // 4,
      kernel_size=3,
    )
    self._conv2 = layers.SNConv2D(in_channels=context_channels//4,out_channels=latent_channels//4, kernel_size=1,padding=0)
    self._gblock2 = GBlock(input_channels=latent_channels//4,output_channels=latent_channels//4)
    self._g_up_block2 = UpsampleGBlock(input_channels=latent_channels//4,output_channels=latent_channels//8)



    self._conv_gru1 = ConvGRU(
      input_channels=latent_channels // 8 + context_channels // 8,
      output_channels=context_channels // 8,
      kernel_size=3,
    )
    self._conv1 = layers.SNConv2D(in_channels=context_channels//8,out_channels=latent_channels//8, kernel_size=1,padding=0)
    self._gblock1 = GBlock(input_channels=latent_channels//8,output_channels=latent_channels//8)
    self._g_up_block1 = UpsampleGBlock(input_channels=latent_channels//8,output_channels=latent_channels//16)

    self._bn = torch.nn.BatchNorm2d(latent_channels // 16)
    self._output_conv = layers.SNConv2D(in_channels=latent_channels // 16,out_channels=4* output_channels,kernel_size=1,padding=0)
    self.depth2space = torch.nn.PixelShuffle(upscale_factor=2)
    self.relu = torch.nn.ReLU()

  def forward(self, initial_states, resolution):
    init_state_1, init_state_2, init_state_3, init_state_4 = initial_states
    batch_size = init_state_1.shape[0]

    # Latent conditioning stack.
    z = self._latent_stack(batch_size, resolution)
    hs = [z] * self._num_predictions

    # Layer 4 (bottom-most).
    # print(hs.shape,'hs')
    # print(init_state_4.shape, 'init_state_4')
    hs = self._conv_gru4(hs, init_state_4)
    hs = [self._conv4(h) for h in hs]
    hs = [self._gblock4(h) for h in hs]
    hs = [self._g_up_block4(h) for h in hs]

    # Layer 3.
    hs = self._conv_gru3(hs, init_state_3)
    hs = [self._conv3(h) for h in hs]
    hs = [self._gblock3(h) for h in hs]
    hs = [self._g_up_block3(h) for h in hs]

    # Layer 2.
    hs = self._conv_gru2(hs, init_state_2)
    hs = [self._conv2(h) for h in hs]
    hs = [self._gblock2(h) for h in hs]
    hs = [self._g_up_block2(h) for h in hs]

    # Layer 1 (top-most).
    hs = self._conv_gru1(hs, init_state_1)
    hs = [self._conv1(h) for h in hs]
    hs = [self._gblock1(h) for h in hs]
    hs = [self._g_up_block1(h) for h in hs]

    # Output layer.
    hs = [F.relu(self._bn(h)) for h in hs]
    hs = [self._output_conv(h) for h in hs]
    hs = [self.depth2space(h) for h in hs]

    return torch.stack(hs, dim=1)


class GBlock(nn.Module):
  """Residual generator block without upsampling."""

  def __init__(self,input_channels,output_channels):
    super(GBlock, self).__init__()
    self._conv1_3x3 = layers.SNConv2D(
        in_channels=input_channels,out_channels=output_channels, kernel_size=3,padding=1)
    self._bn1 = torch.nn.BatchNorm2d(output_channels)
    self._conv2_3x3 = layers.SNConv2D(
        in_channels=output_channels,out_channels=output_channels, kernel_size=3,padding=1)
    self._bn2 = torch.nn.BatchNorm2d(output_channels)
    self._output_channels = output_channels
    self.input_channels = input_channels
    self.relu = torch.nn.ReLU()



  def forward(self, inputs):
    input_channels = self.input_channels
    output_channels = self._output_channels
    # Optional spectrally normalized 1x1 convolution.
    if input_channels != self._output_channels:
      conv_1x1 = layers.SNConv2D(
        input_channels,output_channels, kernel_size=1,padding=0)
      sc = conv_1x1(inputs)
    else:
      sc = inputs

    # Two-layer residual connection, with batch normalization, nonlinearity and
    # 3x3 spectrally normalized convolution in each layer.
    h = self._bn1(inputs)
    h = self.relu(h)
    h = self._conv1_3x3(h)
    h = self._bn2(h)
    h = self.relu(h)
    h = self._conv2_3x3(h)

    # Residual connection.
    return h + sc


class UpsampleGBlock(nn.Module):
  """Upsampling residual generator block."""

  def __init__(self,input_channels, output_channels):
    super(UpsampleGBlock, self).__init__()
    self._conv_1x1 = layers.SNConv2D(
       in_channels = input_channels,out_channels=output_channels, kernel_size=1, padding=0)
    self._conv1_3x3 = layers.SNConv2D(
      in_channels=input_channels, out_channels=output_channels, kernel_size=3, padding=1)
    self._bn1 = torch.nn.BatchNorm2d(input_channels)
    self._conv2_3x3 = layers.SNConv2D(
      in_channels=output_channels, out_channels=output_channels, kernel_size=3, padding=1)
    self._bn2 = torch.nn.BatchNorm2d(output_channels)
    self._output_channels = output_channels
    self.relu = torch.nn.ReLU()



  def forward(self, inputs):
    # x2 upsampling and spectrally normalized 1x1 convolution.
    sc = layers.upsample_nearest_neighbor(upsample_size=2)(inputs)
    sc = self._conv_1x1(sc)

    # Two-layer residual connection, with batch normalization, nonlinearity and
    # 3x3 spectrally normalized convolution in each layer, and x2 upsampling in
    # the first layer.
    h = self._bn1(inputs)
    h = self.relu(h)
    h = layers.upsample_nearest_neighbor(upsample_size=2)(h)
    h = self._conv1_3x3(h)
    h = self._bn2(h)
    h = self.relu(h)
    h = self._conv2_3x3(h)

    # Residual connection.
    return h + sc




if __name__=="__main__":
  # CSnet=ConditioningStack()
  # input=torch.randn((16,4,1,128,128))
  # init_state_1, init_state_2, init_state_3, init_state_4=CSnet(input)
  # print(init_state_1.shape)
  # print(init_state_2.shape)
  # print(init_state_3.shape)
  # print(init_state_4.shape)
  # print('end')
  input = torch.randn((2, 4, 1, 256, 256))
  print('sssssss')
  generator=Generator()
  print(generator.parameters())
  predictions=generator(input)
  print(predictions.shape)