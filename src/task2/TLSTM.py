import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from pytorch_lightning import (
    LightningModule,
    Trainer,
    seed_everything,
    LightningDataModule,
)
from pytorch_lightning.callbacks import ModelCheckpoint

SEED = 0
seed_everything(SEED, workers=True)
GEN_SEED = torch.Generator().manual_seed(SEED)


def create_dataset(
    df_anagrafica,
    df_diagnosi,
    df_esami_par,
    df_esami_par_cal,
    df_esami_stru,
    df_pre_diab_farm,
    df_pre_diab_no_farm,
    df_pre_no_diab,
):
    # Adding 'tipo' column directly to each DataFrame
    df_esami_par["tipo"] = "esame parametri"
    df_esami_par_cal["tipo"] = "esame parametri calcolati"
    df_esami_stru["tipo"] = "esame strumentale"
    df_pre_diab_farm["tipo"] = "prescrizione farmaco diabate"
    df_pre_diab_no_farm["tipo"] = "prescrizione non-farmaco diabete"
    df_pre_no_diab["tipo"] = "prescrizione non-diabete"
    df_diagnosi["tipo"] = "diagnosi"

    # Concatenate dataframes
    dfs = [
        df_esami_par,
        df_esami_par_cal,
        df_esami_stru,
        df_pre_diab_farm,
        df_pre_diab_no_farm,
        df_pre_no_diab,
        df_diagnosi,
    ]
    final_df = pd.concat(dfs, ignore_index=True)

    # Drop unnecessary columns because the df is too big
    df_anagrafica = df_anagrafica.drop(
        columns=[
            "annonascita",
            "annoprimoaccesso",
            "annodecesso",
            "annodiagnosidiabete",
        ]
    )
    # Merge with df_anagrafica
    final_df = final_df.merge(df_anagrafica, on=["idana", "idcentro"], how="inner")

    # Sort by date (inplace)
    final_df.sort_values(by=["data"], inplace=True)

    # Mapping for 'sesso' column (use astype to save memory)
    final_df["sesso"] = (
        final_df["sesso"].map({"M": 0, "F": 1}, na_action="ignore").astype("Int8")
    )

    # Mapping for 'valore' column (use astype to save memory)
    valore_mapping = {"N": 0, "P": 1, "S": 2}
    final_df["valore"] = (
        final_df["valore"].map(valore_mapping, na_action="ignore").astype("Int8")
    )

    # Mapping for other columns
    columns_to_map = [
        "codiceamd",
        "codiceatc",
        "codicestitch",
        "descrizionefarmaco",
        "tipo",
    ]
    for col in columns_to_map:
        mapping = {k: v for v, k in enumerate(final_df[col].unique())}
        final_df[col] = final_df[col].map(mapping, na_action="ignore").astype("Int16")

    # Convert 'valore' to numeric
    final_df["valore"] = pd.to_numeric(final_df["valore"], errors="coerce")

    # Fill NaNs with -100
    final_df = final_df.fillna(-100)

    # Check for non-numeric 'valore' values
    non_numeric_valores = final_df["valore"].unique()
    if any(isinstance(i, str) for i in non_numeric_valores):
        raise ValueError("Value not converted to numeric.")

    return final_df


class TLSTM(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, fc_dim, train):
        super(TLSTM, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.train_mode = train

        # Create PyTorch tensors instead of TensorFlow placeholders
        self.input = nn.Parameter(torch.Tensor(), requires_grad=True)
        self.labels = nn.Parameter(torch.Tensor(), requires_grad=True)
        self.time = nn.Parameter(torch.Tensor(), requires_grad=True)
        self.keep_prob = nn.Parameter(torch.Tensor(), requires_grad=True)

        if train == 1:
            self.Wi = self.init_weights(
                self.input_dim, self.hidden_dim, name="Input_Hidden_weight", reg=None
            )
            self.Ui = self.init_weights(
                self.hidden_dim, self.hidden_dim, name="Input_State_weight", reg=None
            )
            self.bi = self.init_bias(self.hidden_dim, name="Input_Hidden_bias")

            self.Wf = self.init_weights(
                self.input_dim, self.hidden_dim, name="Forget_Hidden_weight", reg=None
            )
            self.Uf = self.init_weights(
                self.hidden_dim, self.hidden_dim, name="Forget_State_weight", reg=None
            )
            self.bf = self.init_bias(self.hidden_dim, name="Forget_Hidden_bias")

            self.Wog = self.init_weights(
                self.input_dim, self.hidden_dim, name="Output_Hidden_weight", reg=None
            )
            self.Uog = self.init_weights(
                self.hidden_dim, self.hidden_dim, name="Output_State_weight", reg=None
            )
            self.bog = self.init_bias(self.hidden_dim, name="Output_Hidden_bias")

            self.Wc = self.init_weights(
                self.input_dim, self.hidden_dim, name="Cell_Hidden_weight", reg=None
            )
            self.Uc = self.init_weights(
                self.hidden_dim, self.hidden_dim, name="Cell_State_weight", reg=None
            )
            self.bc = self.init_bias(self.hidden_dim, name="Cell_Hidden_bias")

            self.W_decomp = self.init_weights(
                self.hidden_dim,
                self.hidden_dim,
                name="Decomposition_Hidden_weight",
                reg=None,
            )
            self.b_decomp = self.init_bias(
                self.hidden_dim, name="Decomposition_Hidden_bias_enc"
            )

            self.Wo = self.init_weights(
                self.hidden_dim, fc_dim, name="Fc_Layer_weight", reg=None
            )
            self.bo = self.init_bias(fc_dim, name="Fc_Layer_bias")

            self.W_softmax = self.init_weights(
                fc_dim, output_dim, name="Output_Layer_weight", reg=None
            )
            self.b_softmax = self.init_bias(output_dim, name="Output_Layer_bias")

        else:
            self.Wi = self.no_init_weights(
                self.input_dim, self.hidden_dim, name="Input_Hidden_weight"
            )
            self.Ui = self.no_init_weights(
                self.hidden_dim, self.hidden_dim, name="Input_State_weight"
            )
            self.bi = self.no_init_bias(self.hidden_dim, name="Input_Hidden_bias")

            self.Wf = self.no_init_weights(
                self.input_dim, self.hidden_dim, name="Forget_Hidden_weight"
            )
            self.Uf = self.no_init_weights(
                self.hidden_dim, self.hidden_dim, name="Forget_State_weight"
            )
            self.bf = self.no_init_bias(self.hidden_dim, name="Forget_Hidden_bias")

            self.Wog = self.no_init_weights(
                self.input_dim, self.hidden_dim, name="Output_Hidden_weight"
            )
            self.Uog = self.no_init_weights(
                self.hidden_dim, self.hidden_dim, name="Output_State_weight"
            )
            self.bog = self.no_init_bias(self.hidden_dim, name="Output_Hidden_bias")

            self.Wc = self.no_init_weights(
                self.input_dim, self.hidden_dim, name="Cell_Hidden_weight"
            )
            self.Uc = self.no_init_weights(
                self.hidden_dim, self.hidden_dim, name="Cell_State_weight"
            )
            self.bc = self.no_init_bias(self.hidden_dim, name="Cell_Hidden_bias")

            self.W_decomp = self.no_init_weights(
                self.hidden_dim, self.hidden_dim, name="Decomposition_Hidden_weight"
            )
            self.b_decomp = self.no_init_bias(
                self.hidden_dim, name="Decomposition_Hidden_bias_enc"
            )

            self.Wo = self.no_init_weights(
                self.hidden_dim, fc_dim, name="Fc_Layer_weight"
            )
            self.bo = self.no_init_bias(fc_dim, name="Fc_Layer_bias")

            self.W_softmax = self.no_init_weights(
                fc_dim, output_dim, name="Output_Layer_weight"
            )
            self.b_softmax = self.no_init_bias(output_dim, name="Output_Layer_bias")

    def init_weights(self, input_dim, output_dim, name, std=0.1, reg=None):
        return nn.Parameter(torch.randn(input_dim, output_dim) * std)

    def init_bias(self, output_dim, name):
        return nn.Parameter(torch.ones(output_dim))

    def no_init_weights(self, input_dim, output_dim, name):
        return nn.Parameter(torch.randn(input_dim, output_dim))

    def no_init_bias(self, output_dim, name):
        return nn.Parameter(torch.ones(output_dim))

    def forward(self, input):
        return self.get_outputs()

    def TLSTM_Unit(self, prev_hidden_memory, concat_input):
        prev_hidden_state, prev_cell = prev_hidden_memory

        batch_size = concat_input.size(0)
        x = concat_input[:, :, 1:]
        t = concat_input[:, :, 0:1]

        # Dealing with time irregularity
        T = self.map_elapse_time(t)

        C_ST = torch.tanh(torch.matmul(prev_cell, self.W_decomp) + self.b_decomp)
        C_ST_dis = T * C_ST
        prev_cell = prev_cell - C_ST + C_ST_dis

        i = torch.sigmoid(
            torch.matmul(x, self.Wi)
            + torch.matmul(prev_hidden_state, self.Ui)
            + self.bi
        )
        f = torch.sigmoid(
            torch.matmul(x, self.Wf)
            + torch.matmul(prev_hidden_state, self.Uf)
            + self.bf
        )
        o = torch.sigmoid(
            torch.matmul(x, self.Wog)
            + torch.matmul(prev_hidden_state, self.Uog)
            + self.bog
        )
        C = torch.tanh(
            torch.matmul(x, self.Wc)
            + torch.matmul(prev_hidden_state, self.Uc)
            + self.bc
        )
        Ct = f * prev_cell + i * C
        current_hidden_state = o * torch.tanh(Ct)

        return torch.stack([current_hidden_state, Ct])

    def get_states(self):
        batch_size = self.input.size(0)
        print("batch_size: ", batch_size)
        scan_input_ = self.input.permute(2, 0, 1)
        print("scan_input: ", scan_input_)
        scan_input = scan_input_.permute(2, 1, 0)
        scan_time = self.time.permute(1, 0)
        initial_hidden = torch.zeros(batch_size, self.hidden_dim, dtype=torch.float32)
        ini_state_cell = torch.stack([initial_hidden, initial_hidden])
        scan_time = scan_time.unsqueeze(2)
        concat_input = torch.cat([scan_time, scan_input], 2)
        packed_hidden_states = torch.scan(self.TLSTM_Unit, concat_input, ini_state_cell)
        all_states = packed_hidden_states[:, 0, :, :]
        return all_states

    def get_output(self, state):
        output = F.relu(torch.matmul(state, self.Wo) + self.bo)
        output = F.dropout(output, self.keep_prob)
        output = torch.matmul(output, self.W_softmax) + self.b_softmax
        return output

    def get_outputs(self):
        all_states = self.get_states()
        all_outputs = torch.stack([self.get_output(state) for state in all_states])
        output = all_outputs[-1, :, :]  # Take the last output
        return output

    def get_cost_acc(self):
        logits = self.get_outputs()
        cross_entropy = nn.CrossEntropyLoss()(logits, torch.argmax(self.labels, dim=1))
        y_pred = torch.argmax(logits, dim=1)
        y = torch.argmax(self.labels, dim=1)
        return cross_entropy, y_pred, y, logits, self.labels

    def map_elapse_time(self, t):
        c1 = torch.tensor(1, dtype=torch.float32)
        c2 = torch.tensor(2.7183, dtype=torch.float32)
        T = c1 / torch.log(t + c2)
        Ones = torch.ones(1, self.hidden_dim, dtype=torch.float32)
        T = torch.matmul(T, Ones)
        return T


class TLSTMDataModule(LightningDataModule):
    def __init__(
        self,
        input_df,
        train_batch_size: int = 16,
        eval_batch_size: int = 16,
        **kwargs,
    ):
        super().__init__()
        self.input_df = input_df
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size

    def setup(self, stage=None):
        # Calculate the input length directly
        len_input = (
            len(self.input_df.columns) - 4
        )  # Assuming the number of columns is constant

        # Group by "idana" and "idcentro" and sort each group by "data"
        grouped = self.input_df.groupby(["idana", "idcentro"], group_keys=True)

        inputs = []
        labels = []
        max_history_len = 0

        # Limit the number of iterations to a reasonable value
        max_iterations = min(len(grouped), 9999999)
        for count, (_, group) in enumerate(grouped):
            patient_history = group.sort_values(by=["data"])
            labels.append(
                patient_history["label"].iloc[0]
            )  # Use iloc to access the first item
            patient_history = patient_history.drop(
                columns=["idana", "idcentro", "label", "data"]
            )
            nested_list = patient_history.to_numpy(dtype=np.float32)
            inputs.append(nested_list)

            max_history_len = max(
                max_history_len, nested_list.shape[0]
            )  # Update max_history_len

            if count + 1 >= max_iterations:
                break

        # Pad sequences using pad_sequence
        tensor_list = [
            torch.cat(
                (
                    torch.zeros(max_history_len - len(sublist), len_input),
                    torch.tensor(sublist),
                )
            )
            for sublist in inputs
        ]
        padded_tensor = pad_sequence(tensor_list, batch_first=True)
        padded_tensor = padded_tensor.to(torch.float32)

        bool_tensor = torch.tensor(labels, dtype=torch.float32)

        # Printing unique values in bool_tensor
        unique_values, value_counts = torch.unique(bool_tensor, return_counts=True)
        print("Unique values in bool_tensor:")
        print(unique_values)
        print("Value counts:")
        print(value_counts)

        # Create a dataset
        dataset = TensorDataset(padded_tensor, bool_tensor)

        # split dataset into train and validation sampling randomly
        # use 20% of training data for validation
        train_set_size = int(len(dataset) * 0.8)
        valid_set_size = len(dataset) - train_set_size

        # split the dataset randomly into two
        self.train_data, self.valid_data = torch.utils.data.random_split(
            dataset, [train_set_size, valid_set_size], generator=GEN_SEED
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=8,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_data,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=4,
        )

    def test_dataloader(self):
        # placeholder
        return DataLoader(
            self.valid_data,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=4,
        )


# adapted from https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#child-modules
class LitTLSTM(LightningModule):
    def __init__(
        self,
        tlstm,
        learning_rate: float = 1e-4,
        adam_epsilon: float = 1e-8,
        warmup_steps: int = 0,
        weight_decay: float = 0.0,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["tlstm"])

        self.model = tlstm
        self.metric = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model.forward(x)

    def training_step(self, batch, batch_idx):
        x, _ = batch
        x_hat = self.model(x)
        loss = self.metric(x, x_hat)
        return loss

    def validation_step(self, batch, batch_idx):
        self._shared_eval(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        self._shared_eval(batch, batch_idx, "test")

    def _shared_eval(self, batch, batch_idx, prefix):
        x, _ = batch
        x_hat = self.model(x)
        loss = self.metric(x, x_hat)
        self.log(f"{prefix}_loss", loss)

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.hparams.learning_rate,
            eps=self.hparams.adam_epsilon,
        )

        # scheduler = get_linear_schedule_with_warmup(
        #     optimizer,
        #     num_warmup_steps=self.hparams.warmup_steps,
        #     num_training_steps=self.trainer.estimated_stepping_batches,
        # )
        # scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer]  # , [scheduler]
