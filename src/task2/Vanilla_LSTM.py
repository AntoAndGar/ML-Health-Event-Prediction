

from typing import Any
import torch
import torch.nn as nn 
import torch.nn.functional as F
from torch.optim import Adam

import numpy as np

import pytorch_lightning as pl

from torch.utils.data import DataLoader, TensorDataset

from transformers import (
    AutoTokenizer,
)
import pandas as pd

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


class LightingVanillaLSTM(pl.LightningModule):
    def __init__(self, input_size, hidden_size = 1):
        super().__init__()
        #input_size = number of features as input
        #hidden_size = number of features in hidden state, so the output size
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size)
        #self.lstm = nn.LSTM(input_size=1, hidden_size=1, num_layers=1, batch_first=True)

    def forward(self, input):
        input_trans = input.view(len(input), 1)
        lstm_out, temp = self.lstm(input_trans)

        prediction = lstm_out[-1]
        return prediction
    
    def configure_optimizers(self) -> Any:
        return Adam(self.parameters(), lr=0.001)

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
    
    
def evaluate_vanilla_LSTM(model, dataloader):
    print("Using {torch.cuda.get_device_name(DEVICE)}")

    trainer = pl.Trainer(max_epochs=2000)
    trainer.fit(model, train_dataloaders=dataloader)

    # why lose time using keras or tensorflow ?
    # when we can use pytorch (pytorch lightning I mean, but also pytorch is ok)

def create_dataset(df_anagrafica, df_diagnosi, df_esami_par, df_esami_par_cal, df_esami_stru, df_pre_diab_farm, df_pre_diab_no_farm, df_pre_no_diab):
    df_esami_par['tipo'] = 'esame'
    df_esami_par_cal['tipo'] = 'esame'
    df_esami_stru['tipo'] = 'esame'
    df_pre_diab_farm['tipo'] = 'prescrizione'
    df_pre_diab_no_farm['tipo'] = 'prescrizione'
    df_pre_no_diab['tipo'] = 'prescrizione'
    df_diagnosi['tipo'] = 'diagnosi'

    df_esami_par['extra'] = 'parametri'
    df_esami_par_cal['extra'] = 'parametri calcolati'
    df_esami_stru['extra'] = 'strumentali'
    df_pre_diab_farm['extra'] = 'farmaco diabate'
    df_pre_diab_no_farm['extra'] = 'non-farmaco diabete'
    df_pre_no_diab['extra'] = 'non-diabete'
    df_diagnosi['extra'] = ''
    final_df = pd.concat([df_esami_par, df_esami_par_cal, df_esami_stru, df_pre_diab_farm, df_pre_diab_no_farm, df_pre_no_diab, df_diagnosi])
    final_df = final_df.merge(df_anagrafica, on=['idana', 'idcentro'], how='inner')
    final_df.sort_values(by=['data'])
    final_df['sesso'] = final_df['sesso'].replace('M', 0)
    final_df['sesso'] = final_df['sesso'].replace('F', 1)

    final_df['valore'] = final_df['valore'].replace('N', 0)
    final_df['valore'] = final_df['valore'].replace('P', 1)
    final_df['valore'] = final_df['valore'].replace('S', 2)
    mapping = {k: v for v, k in enumerate(final_df.codiceamd.unique())}
    final_df['codiceamd'] = final_df['codiceamd'].map(mapping)

    mapping = {k: v for v, k in enumerate(final_df.codiceatc.unique())}
    final_df['codiceatc'] = final_df['codiceatc'].map(mapping)

    mapping = {k: v for v, k in enumerate(final_df.codicestitch.unique())}
    final_df['codicestitch'] = final_df['codicestitch'].map(mapping)

    mapping = {k: v for v, k in enumerate(final_df.descrizionefarmaco.unique())}
    final_df['descrizionefarmaco'] = final_df['descrizionefarmaco'].map(mapping)
    return final_df
    mapping = {k: v for v, k in enumerate(final_df.tipo.unique())}
    final_df['tipo'] = final_df['tipo'].map(mapping)

    mapping = {k: v for v, k in enumerate(final_df.extra.unique())}
    final_df['extra'] = final_df['extra'].map(mapping)

    #print("Valore: \t", final_df['valore'].unique())   
    final_df['valore'] = pd.to_numeric(final_df['valore'], errors='coerce')
    aaa = final_df['valore'].unique()   
    for i in aaa:
        if type(i) == str:
            raise("Value not converted to numeric. Impossible to convert to numeric, and use this data for LSTM")
    #print("Dtypes: \t", final_df.dtypes)
    return final_df

def create_array():
    return

'''
def create_dataset(df_anagrafica, list_of_df):
    create_dataset(
        df_anagrafica=df_anagrafica,
        df_diagnosi=list_of_df[0],
        df_esami_par=list_of_df[1],
        df_esami_par_cal=list_of_df[2],
        df_esami_stru=list_of_df[3],
        df_pre_diab_farm=list_of_df[4],
        df_pre_diab_no_farm=list_of_df[5],
        df_pre_no_diab=list_of_df[6])
'''
