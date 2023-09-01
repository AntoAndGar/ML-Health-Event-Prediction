

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
    def __init__(self, input_size, hidden_size = 32):
        super().__init__()
        self.input_size = input_size
        self.loss = nn.MSELoss()
        #self.linear = nn.Linear(hidden_size, 1)

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=2)
    def forward(self, input):
        #print(input.shape)dis
        #print(input.shape)
        #if len(input.shape) == 2:
        mask = torch.all(input != -200, dim=1)
        #Use the mask to select non-empty rows
        input = input[mask]
        #print(input.shape)
        lstm_out, (hn, cn) = self.lstm(input)
        return lstm_out.mean()
        print(input.shape)
        mask = torch.all(input != -200, dim=1)
        #Use the mask to select non-empty rows
        input = input[mask]
        print(input.shape)
        lstm_out, (hn, cn) = self.lstm(input)
        #prediction = self.linear(lstm_out[:,-1])
        #prediction = lstm_out[-1] #FIXME: fix the prediction
        #prediction = lstm_out.mean()
        return lstm_out.mean()
        return prediction

    def configure_optimizers(self) -> Any:
        return Adam(self.parameters(), lr=0.0001)

    def training_step(self, batch, batch_idx):
        # Calculate loss and log training process

        x, y = batch
        y_hat = self(x)

        temp_loss = self.loss(y_hat, y)
        #result = pl.EvalResult(loss)
        #result.log('train_loss', loss, prog_bar=True)
        self.log('train_loss', temp_loss, prog_bar=True)   
        return temp_loss
    
    def validation_step(self, batch, batch_idx):
        # Calculate loss and log training process
        #print("Validation step")
        x, y = batch
        tot_loss = 0
        number_samples = y.shape[0]
        for i in range(y.shape[0]):
            y_hat = self(x[i])
            temp_loss = self.loss(y_hat, y[i])
            self.log('val_loss', temp_loss, prog_bar=True)
            tot_loss += temp_loss
        return tot_loss/number_samples
        print(x.shape)
        print(y.shape)
        y_hat = self(x)
        print(y_hat.shape)
        print(y.shape)
        temp_loss = self.loss(y_hat, y)
        print(temp_loss.shape)
        #result = pl.EvalResult(loss)
        #result.log('train_loss', loss, prog_bar=True)
        self.log('val_loss', temp_loss, prog_bar=True)   
        return temp_loss

def evaluate_vanilla_LSTM(model, train, test, val, max_epochs=5):
    print("Using {torch.cuda.get_device_name(DEVICE)}")
    trainer = pl.Trainer(max_epochs=max_epochs)
    trainer.fit(model, train_dataloaders=train, val_dataloaders=val)
    #result = model.evaluate(test_dataloaders=test)
    # Test the model
    # In test phase, we don't need to compute gradients (for memory efficiency)
    with torch.no_grad():
        correct = 0
        total = 0
        wrong = 0
        min = 10
        max = -10
        for input_i, label_i in test:
            output = model(input_i)#[-1]
            #print(output, label_i)
            if output >= max:
                max = output
            if output <= min:
                min = output
            if output > 1:
                output = 1
            if output < 0:
                output = 0
            if output >= 0.5:
                output = 1
            else:
                output = 0
            if output == label_i:
                correct += 1
            else:
                wrong += 1
            total += 1
        print("Max: ", max)
        print("Min: ", min)
        print("Correct: ", correct)
        print("Total: ", total)
        print("Wrong: ", wrong)
        print('Accuracy of the network on the test data: {} %'.format(100 * correct / total))
    return

def create_dataset(df_anagrafica, df_diagnosi, df_esami_par, df_esami_par_cal, df_esami_stru, df_pre_diab_farm, df_pre_diab_no_farm, df_pre_no_diab, delta=False):
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
    if delta:
        final_df.sort_values(by=['delta_data'])
    else:
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

    mapping = {False: 0, True: 1, 'False': 0, 'True': 1}
    final_df['label'] = final_df['label'].map(mapping)

    #print("Valore: \t", final_df['valore'].unique())   
    final_df['valore'] = pd.to_numeric(final_df['valore'], errors='coerce')

    # Convert datatime of year to float or int [it's the same]
    #final_df.drop(columns=['annonascita', 'annodecesso', 'annodiagnosidiabete', 'annoprimoaccesso'], inplace=True)
    '''
    final_df['annonascita'] = final_df['annonascita'].dt.year  
    if not delta:
        final_df['annodecesso'] = final_df['annodecesso'].dt.year
        final_df['annodiagnosidiabete'] = final_df['annodiagnosidiabete'].dt.year
        final_df['annoprimoaccesso'] = final_df['annoprimoaccesso'].dt.year
    '''
    final_df = final_df.fillna(-100)
    print("Dtypes: \n", final_df.dtypes)

    return final_df

