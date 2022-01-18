

"""Latent Conditioning Stack."""

import torch
import torch.nn as nn
import layers
import einops
from torch.nn import functional as F
# from torch.nn.modules.pixelshuffle import PixelUnshuffle



class LatentConditioningStack(nn.Module):
  """Latent Conditioning Stack for the Sampler."""

  def __init__(self,):
    super(LatentConditioningStack, self).__init__()
    self._lblock1 = LBlock(input_channels=8,output_channels=24)
    self._lblock2 = LBlock(input_channels=24,output_channels=48)
    self._lblock3 = LBlock(input_channels=48,output_channels=192)
    self._mini_atten_block = Attention(input_channels=192,output_channels=192)
    self._lblock4 = LBlock(input_channels=192,output_channels=768)
    self.SNConv2D = layers.SNConv2D(in_channels=8,out_channels=8, kernel_size=3,padding=1)

  def forward(self, batch_size, resolution=(256, 256)):

    # Independent draws from a Normal distribution.
    h, w = resolution[0] // 32, resolution[1] // 32
    # z = tf.random.normal([batch_size, h, w, 8])
    z = torch.randn((batch_size, 8, h, w)).cuda()

    # 3x3 convolution.
    in_channels = z.shape[1]
    print(in_channels,'ppppppppppp')

    z = self.SNConv2D(z)

    # Three L Blocks to increase the number of channels to 24, 48, 192.
    z = self._lblock1(z)
    z = self._lblock2(z)
    z = self._lblock3(z)

    # Spatial attention module.
    z = self._mini_atten_block(z)


    # L Block to increase the number of channels to 768.
    z = self._lblock4(z)

    return z


class LBlock(nn.Module):
  """Residual block for the Latent Stack."""

  def __init__(self,input_channels,output_channels , kernel_size=3, conv=layers.Conv2D,
               activation=F.relu):
    """Constructor for the D blocks of the DVD-GAN.

    Args:
      output_channels: Integer number of channels in convolution operations in
        the main branch, and number of channels in the output of the block.
      kernel_size: Integer kernel size of the convolutions. Default: 3.
      conv: TF module. Default: layers.Conv2D.
      activation: Activation before the conv. layers. Default: tf.nn.relu.
    """
    super(LBlock, self).__init__()
    self._output_channels = output_channels
    self._kernel_size = kernel_size
    self._conv_1x1= conv(in_channels=input_channels,
                    out_channels=output_channels-input_channels,
                    kernel_size=1,
                    padding=0)
    self.first_conv_3x3 = conv(in_channels=input_channels,
                          out_channels=output_channels,
                          kernel_size=3,
                          padding=1)
    self.last_conv_3x3 = conv(in_channels=output_channels,
                    out_channels=output_channels,
                    kernel_size=3,
                    padding=1)
    self._activation = activation

  def forward(self, inputs):
    """Build the LBlock.

    Args:
      inputs: a tensor with a complete observation [N 256 256 1]

    Returns:
      A tensor with discriminator loss scalars [B].
    """

    # Stack of two conv. layers and nonlinearities that increase the number of
    # channels.
    h0 = self._activation(inputs)
    # h1 = self._conv(num_channels=self.output_channels,
    #                 kernel_size=self._kernel_size)(h0)

    input_channels = h0.shape[1]
    h1 = self.first_conv_3x3(h0)

    h1 = self._activation(h1)

    h2 = self.last_conv_3x3(h1)

    # h2 = self._conv(num_channels=self._output_channels,
    #                 kernel_size=self._kernel_size)(h1)

    # Prepare the residual connection branch.
    if input_channels < self._output_channels:

      sc = self._conv_1x1(inputs)



      sc = torch.cat((inputs, sc), dim=1)
    else:
      sc = inputs

    # Residual connection.
    print(h2.shape,sc.shape,'oooooooo')
    return h2 + sc

def attention_einsum(q, k, v):
    """Apply the attention operator to tensors of shape [h, w, c]."""

    # Reshape 3D tensors to 2D tensor with first dimension L = h x w.
    k = einops.rearrange(k, "h w c -> (h w) c")  # [h, w, c] -> [L, c]
    v = einops.rearrange(v, "h w c -> (h w) c")  # [h, w, c] -> [L, c]

    # Einstein summation corresponding to the query * key operation.
    beta = F.softmax(torch.einsum("hwc, Lc->hwL", q, k), dim=-1)

    # Einstein summation corresponding to the attention * value operation.
    out = torch.einsum("hwL, Lc->hwc", beta, v)
    return out


class Attention(torch.nn.Module):
  """Attention Module"""

  def __init__(self, input_channels: int, output_channels: int, ratio_kq=8, ratio_v=8):
    super(Attention, self).__init__()

    self.ratio_kq = ratio_kq
    self.ratio_v = ratio_v
    self.output_channels = output_channels
    self.input_channels = input_channels

    # Compute query, key and value using 1x1 convolutions.
    self.query = torch.nn.Conv2d(
      in_channels=input_channels,
      out_channels=self.output_channels // self.ratio_kq,
      kernel_size=(1, 1),
      padding="valid",
      bias=False,
    )
    self.key = torch.nn.Conv2d(
      in_channels=input_channels,
      out_channels=self.output_channels // self.ratio_kq,
      kernel_size=(1, 1),
      padding="valid",
      bias=False,
    )
    self.value = torch.nn.Conv2d(
      in_channels=input_channels,
      out_channels=self.output_channels // self.ratio_v,
      kernel_size=(1, 1),
      padding="valid",
      bias=False,
    )

    self.last_conv = torch.nn.Conv2d(
      in_channels=self.output_channels // 8,
      out_channels=self.output_channels,
      kernel_size=(1, 1),
      padding="valid",
      bias=False,
    )

    # Learnable gain parameter
    self.gamma = nn.Parameter(torch.zeros(1))

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    # Compute query, key and value using 1x1 convolutions.
    query = self.query(x)
    key = self.key(x)
    value = self.value(x)
    # Apply the attention operation.
    # TODO See can speed this up, ApplyAlongAxis isn't defined in the pseudocode
    out = []
    for b in range(x.shape[0]):
      # Apply to each in batch
      out.append(attention_einsum(query[b], key[b], value[b]))
    out = torch.stack(out, dim=0)
    out = self.gamma * self.last_conv(out)
    # Residual connection.
    return out + x

if __name__=='__main__':
    lcsnet = LatentConditioningStack()
    output=lcsnet(16)
    print(output.shape)