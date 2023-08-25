

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

class LightingVanillaLSTM(pl.LightningModule):
    def __init__(self, input_size, hidden_size = 1):
        super().__init__()
        self.input_size = input_size
        self.loss = nn.CrossEntropyLoss()
        #input_size = number of features as input
        #hidden_size = number of features in hidden state, so the output size
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=2)
    def forward(self, input):
        #print("Input shape: ", input.shape)
        #print("Dypes: ", input.dtype)
        #print("Input: ", input)
        #input_trans = input.view(len(input), 1)
        #TODO: tensore di Hidden state e tensore di LSTM
        #h0 = torch.randn(2, 32) #TODO: fix the shape
        #c0 = torch.randn(2, 32) #TODO: fix the shape
        #tuple_tensor = (torch.tensor)
        #lstm_out, (hn, cn) = self.lstm(input,(h0, c0))
        lstm_out, (hn, cn) = self.lstm(input)
        #print("LSTM out shape: ", lstm_out.shape)
        #print("LSTM out: ", lstm_out)
        #prediction = self.linear(lstm_out)
        prediction = lstm_out[-1] #FIXME: fix the prediction
        return prediction
    
    def configure_optimizers(self) -> Any:
        return Adam(self.parameters(), lr=0.01)

    def training_step(self, batch, batch_idx):
        # Calculate loss and log training process

        input_i, label_i = batch
        output_i = self.forward(input_i)
        #Loss is Sum of the squared residuals

        temp_loss = self.loss(output_i.view(1), label_i.view(1))

        #loss = (output_i - label_i[0]) ** 2

        self.log('train_loss', temp_loss)    
        '''
        if label_i == 0:
            self.log("out_0", output_i) 
        elif label_i == 1:
            self.log("out_1", output_i)
        '''
        return temp_loss
        
def evaluate_vanilla_LSTM(model, train, test, max_epochs=10):
    print("Using {torch.cuda.get_device_name(DEVICE)}")
    trainer = pl.Trainer(max_epochs=max_epochs)
    trainer.fit(model, train_dataloaders=train) #, val_dataloaders=val)
    #result = model.evaluate(test_dataloaders=test)
    # Test the model
    # In test phase, we don't need to compute gradients (for memory efficiency)
    with torch.no_grad():
        correct = 0
        total = 0
        min = 10
        max = -10
        for input_i, label_i in test:
            output = model(input_i)
            if output >= max:
                max = output
            if output <= min:
                min = output
            if output >= 0.5:
                output = 1
            else:
                output = 0
            if output == label_i:
                correct += 1
            total += 1
        print("Max: ", max)
        print("Min: ", min)
        print('Accuracy of the network on the test data: {} %'.format(100 * correct / total))
    return 

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

    mapping = {k: v for v, k in enumerate(final_df.tipo.unique())}
    final_df['tipo'] = final_df['tipo'].map(mapping)

    mapping = {k: v for v, k in enumerate(final_df.extra.unique())}
    final_df['extra'] = final_df['extra'].map(mapping)

    #print("Valore: \t", final_df['valore'].unique())   
    final_df['valore'] = pd.to_numeric(final_df['valore'], errors='coerce')
    final_df = final_df.fillna(-100)
    aaa = final_df['valore'].unique()   
    for i in aaa:
        if type(i) == str:
            raise("Value not converted to numeric. Impossible to convert to numeric, and use this data for LSTM")
    #print("Dtypes: \t", final_df.dtypes)
    return final_df

