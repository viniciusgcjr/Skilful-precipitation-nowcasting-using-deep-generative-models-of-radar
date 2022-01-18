
import functools
import torch.nn.functional as F
import numpy as np
import einops
import discriminator
import latent_stack
import layers
import torch
import generator
import torch.nn as nn
from  convGRU import ConvGRU
from torch.nn.modules.pixelshuffle import PixelShuffle,PixelUnshuffle
from loss import loss_hinge_gen,loss_hinge_disc,grid_cell_regularizer
from latent_stack import LatentConditioningStack
from generator import Sampler, Generator,ContextConditioningStack
from discriminator import Discriminator
import torch.utils.data as Data


import torch
flag = torch.cuda.is_available()
print(flag)

ngpu= 1
# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
print(device)





class DGMR(torch.nn.Module):

    def __init__(
            self,
            forecast_steps: int = 18,
            input_channels: int = 1,
            output_shape: int = 256,
            gen_lr: float = 5e-5,
            disc_lr: float = 2e-4,
            visualize: bool = False,
            pretrained: bool = False,
            conv_type: str = "standard",
            num_samples: int = 6,
            grid_lambda: float = 20.0,
            beta1: float = 0.0,
            beta2: float = 0.999,
            latent_channels: int = 768,
            context_channels: int = 384,
    ):
        super(DGMR, self).__init__()
        self.gen_lr = gen_lr
        self.disc_lr = disc_lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.grid_lambda = grid_lambda
        self.num_samples = num_samples
        self.visualize = visualize
        self.latent_channels = latent_channels
        self.context_channels = context_channels
        self.input_channels = input_channels
        self.conditioning_stack = ContextConditioningStack()
        self.latent_stack = LatentConditioningStack()
        self.sampler = Sampler()
        self.generator = Generator()
        self.discriminator = Discriminator()



        # self.save_hyperparameters()

        self.global_iteration = 0

        # Important: This property activates manual optimization.
        self.automatic_optimization = False
        torch.autograd.set_detect_anomaly(True)

    def train_step(self,batch_inputs,batch_targets):
      # batch_inputs, batch_targets = get_data_batch(batch_size)
      batch_inputs  = batch_inputs
      batch_targets = batch_targets
      g_opt = torch.optim.Adam(self.generator.parameters(), lr=self.gen_lr, betas=(0.0, 0.999))
      # d_opt = torch.optim.Adam(self.discriminator.parameters(), lr=self.disc_lr, betas=(0.0, 0.999))



      ##########################
      # Optimize Discriminator #
      ##########################


      for _ in range(2):
        predictions = self.generator(batch_inputs)
        # Cat along time dimension [B, C, T, H, W]
        generated_sequence = torch.cat([batch_inputs, predictions], dim=1)
        real_sequence = torch.cat([batch_inputs, batch_targets], dim=1)
        # Cat long batch for the real+generated
        concatenated_inputs = torch.cat([real_sequence, generated_sequence], dim=0)

        concatenated_outputs = self.discriminator(concatenated_inputs)

        score_real, score_generated = torch.split(concatenated_outputs, 2, dim=0)
        discriminator_loss = loss_hinge_disc(score_generated, score_real)
        # d_opt.zero_grad()
        # self.manual_backward(discriminator_loss)
        # d_opt.step()

      ######################
      # Optimize Generator #
      ######################

      predictions = [self.generator(batch_inputs) for _ in range(6)]
      grid_cell_reg = grid_cell_regularizer(torch.stack(predictions, dim=0), batch_targets)
      # Concat along time dimension
      generated_sequence = [torch.cat([batch_inputs, x], dim=1) for x in predictions]
      real_sequence = torch.cat([batch_inputs, batch_targets], dim=1)
      # Cat long batch for the real+generated, for each example in the range
      # For each of the 6 examples
      generated_scores = []
      for g_seq in generated_sequence:
        concatenated_inputs = torch.cat([real_sequence, g_seq], dim=0)
        concatenated_outputs = self.discriminator(concatenated_inputs)
        score_real, score_generated = torch.split(concatenated_outputs, 2, dim=0)
        generated_scores.append(score_generated)
      generator_disc_loss = loss_hinge_gen(torch.cat(generated_scores, dim=0))
      generator_loss = generator_disc_loss + self.grid_lambda * grid_cell_reg
      g_opt.zero_grad()
      generator_loss.backward()
      g_opt.step()





    # def configure_optimizers(self):
    #     b1 = self.beta1
    #     b2 = self.beta2
    #     opt_g = torch.optim.Adam(self.generator.parameters(), lr=self.gen_lr, betas=(b1, b2))
    #     opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=self.disc_lr, betas=(b1, b2))
    #     return [opt_g, opt_d], []





if __name__=='__main__':
    model = DGMR(
        forecast_steps=18,
        input_channels=1,
        output_shape=128,
        latent_channels=768,
        context_channels=384,
        num_samples=3,
    )
    model=model.to("cuda")
    print('medol',next(model.parameters()).is_cuda)
    data = np.load(r'../data/radarFrame.npy')
    data = data[:100]
    print(data.shape)

    data_train_x = data[:, 0:4]
    data_train_y = data[:, 4:22]
    print(data_train_x.shape)
    print(data_train_y.shape)

    data_train_x = torch.from_numpy(data_train_x)
    data_train_y = torch.from_numpy(data_train_y)

    train_data = Data.TensorDataset(data_train_x, data_train_y)

    BATCH_SIZE = 2
    EPOCH = 5

    train_loader = Data.DataLoader(dataset=train_data, batch_size=16, shuffle=True)
  
    for epoch in range(EPOCH):
        for step, (b_x, b_y) in enumerate(train_loader):  # gives batch data, normalize x when iterate train_loader

            b_x=b_x.cuda()
            b_y=b_y.cuda()
            model.train_step(b_x,b_y)
        print('epoch')

    torch.save(model, 'model.pkl')
    print('end')





















