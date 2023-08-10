

from typing import Any
import torch
import torch.nn as nn 
import torch.nn.functional as F
from torch.optim import Adam

import pytorch_lightning as pl

from torch.utils.data import DataLoader, TensorDataset

from transformers import (
    AutoTokenizer,
)

class Vanilla_LSTM_Data_Module(pl.LightningModule):

    def __init__(
    self,
    mean = torch.tensor(0.0),
    std = torch.tensor(1.0)
    ):
        super().__init__()
        self.mean = mean
        self.std = std

        # Define the parameters for the LSTM units
        self.wlr1 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)
        self.wlr2 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)
        self.blr1 = nn.Parameter(torch.tensor(0.), requires_grad=True)

        self.wpr1 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)
        self.wpr2 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)
        self.bpr1 = nn.Parameter(torch.tensor(0.), requires_grad=True)

        self.wp1 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)
        self.wp2 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)
        self.bp1 = nn.Parameter(torch.tensor(0.), requires_grad=True)

        self.wo1 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)
        self.wo2 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)
        self.bo1 = nn.Parameter(torch.tensor(0.), requires_grad=True)


    # This function is the core of the LSTM unit, containing the equations for the long and short-term memory
    def lstm_unit(self, input_size, long_memory, short_memory):

        # The percentage of long memory that will be remembered
        long_remember_percent = torch.sigmoid((self.wlr1 * short_memory) +
                                              (self.wlr2 * input_size) +
                                              self.blr1)
        
        # Create a new potential long memory and decide how much of it will be remembered
        potential_remember_percent = torch.sigmoid((self.wèr1 * short_memory) +
                                              (self.wèr2 * input_size) +
                                              self.bpr1)
        potential_memory = torch.tanh((self.wp1 * short_memory) +
                                      (self.wp2 * input_size) +
                                      self.bp1)
        
        # Update the long memory
        updated_long_memory = (long_memory * long_remember_percent) + (potential_remember_percent * potential_memory)
        
        # Create a new short-term memory and determine what percentage to remember
        output_percent = torch.sigmoid((self.wo1 * short_memory) +
                                        (self.wo2 * input_size) +
                                        self.bo1)
        updated_short_memory = torch.tanh(updated_long_memory) * output_percent

        # Update the long and short-term memory
        return ([updated_long_memory, updated_short_memory])
    

    def forward(self, input):
        long_memory = 0
        short_memory = 0
        # TODO: Qui inserisce il numero di giorni fissato, non so come muovermi
        for i in input:
            long_memory, short_memory = self.lstm_unit(i, long_memory, short_memory)
        return short_memory
    
        day1 = input[0]
        day2 = input[1]

        long_memory, short_memory = self.lstm_unit(day1, long_memory, short_memory)
        long_memory, short_memory = self.lstm_unit(day2, long_memory, short_memory)

    def configure_optimizers(self):
        return Adam(self.parameters())
        return super().configure_optimizers()
    
    def training_step(self, batch, batch_idx):
        # Calculate loss and log training process

        input_i, label_i = batch
        output_i = self.forward(input_i[0])
        #Loss is Sum of the squared residuals
        loss = (output_i - label_i[0]) ** 2

        self.log('train_loss', loss)    

        if label_i == 0:
            self.log("out_0", output_i)
        elif label_i == 1:
            self.log("out_1", output_i)

        return loss





def evaluate_vanilla_LSTM():
    print("Using {torch.cuda.get_device_name(DEVICE)}")

    # why lose time using keras or tensorflow ?
    # when we can use pytorch (pytorch lightning I mean, but also pytorch is ok)

