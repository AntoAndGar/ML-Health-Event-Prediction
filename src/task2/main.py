import concurrent.futures as futures
import datasets
import multiprocessing
import numpy as np
import pandas as pd
import pickle
import os
import re
import torch
import random
import sys

import Vanilla_LSTM

from datetime import datetime
from typing import Optional

from pytorch_lightning import (
    LightningDataModule,
    LightningModule,
    Trainer,
    seed_everything,
)
from pytorch_lightning.callbacks import ModelCheckpoint

from torch.utils.data import DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence

from torchmetrics.classification import BinaryAccuracy, BinaryF1Score

from transformers import (
    # AdamW,  # this does not work
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

import TLSTM
import tensorflow as tf
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score


SEED = 0
rng = np.random.default_rng(SEED)
GEN_SEED = torch.Generator().manual_seed(SEED)
seed_everything(SEED, workers=True)
torch.manual_seed(SEED)
MODEL_NAME = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"

os.environ["TOKENIZERS_PARALLELISM"] = "true"

READ_DATA_PATH = "clean_data"
PRESCRIZIONI: bool = True
BERT_DATASET: bool = True
CREATE_BERT_DATASET: bool = True
PARALLEL_LOAD_DATASET: bool = True
WRITE_DATASET: bool = False
DATASET_NAME = "dataset_def.pkl"
EVALUATE_BERT: bool = True

TRAIN_TLSTM: bool = True
EVALUATE_TLSTM: bool = True

# VANILLA LSTM PARAMETERS
VANILLA_LSTM: bool = True
LOAD_VANILLA_DF: bool = False
SAVE_VANILLA_DF: bool = False
DROP_ANNI: bool = True
LSTM_DF = "lstm_df"

# DELTA_ETA PARAMETERS
DELTA_ETA: bool = True
WRITE_DELTA_ETA_DF: bool = False
DELTA_ETA_PATH = "delta_eta_df"

# DELTA VANILLA LSTM PARAMETERS
DELTA_VANILLA_LSTM: bool = True
LOAD_DELTA_VANILLA_DF: bool = False
SAVE_DELTA_VANILLA_DF: bool = False
DELTA_LSTM_DF = "lstm_df"

WRITE_CSV: bool = True
WRITE_DATA_PATH = "balanced_data"

BALANCING = "standard"

if PRESCRIZIONI:
    file_names = [
        "anagraficapazientiattivi_c_pres",
        "diagnosi_c_pres",
        "esamilaboratorioparametri_c_pres",
        "esamilaboratorioparametricalcolati_c_pres",
        "esamistrumentali_c_pres",
        "prescrizionidiabetefarmaci_c_pres",
        "prescrizionidiabetenonfarmaci_c_pres",
        "prescrizioninondiabete_c_pres",
    ]
else:
    file_names = [
        "anagraficapazientiattivi_c",
        "diagnosi_c",
        "esamilaboratorioparametri_c",
        "esamilaboratorioparametricalcolati_c",
        "esamistrumentali_c",
        "prescrizionidiabetefarmaci_c",
        "prescrizionidiabetenonfarmaci_c",
        "prescrizioninondiabete_c",
    ]

AMD_OF_CARDIOVASCULAR_EVENT = [
    "AMD047",
    "AMD048",
    "AMD049",
    "AMD071",
    "AMD081",
    "AMD082",
    "AMD208",
    "AMD303",
]


def read_csv(filename):
    return pd.read_csv(filename, header=0)


print("Generating Futures...")
# Read all the dataset concurrently and store them in a dictionary with the name of the file as key
with futures.ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
    df_list = dict()
    for name in file_names:
        df_list[str(name)] = executor.submit(read_csv, f"{READ_DATA_PATH}/{name}.csv")

print("Loading data...")

if PRESCRIZIONI:
    df_anagrafica = df_list["anagraficapazientiattivi_c_pres"].result()
    df_diagnosi = df_list["diagnosi_c_pres"].result()
    df_esami_lab_par = df_list["esamilaboratorioparametri_c_pres"].result()
    df_esami_lab_par_cal = df_list["esamilaboratorioparametricalcolati_c_pres"].result()
    df_esami_stru = df_list["esamistrumentali_c_pres"].result()
    df_pres_diab_farm = df_list["prescrizionidiabetefarmaci_c_pres"].result()
    df_pres_diab_no_farm = df_list["prescrizionidiabetenonfarmaci_c_pres"].result()
    df_pres_no_diab = df_list["prescrizioninondiabete_c_pres"].result()
else:
    df_anagrafica = df_list["anagraficapazientiattivi_c"].result()
    df_diagnosi = df_list["diagnosi_c"].result()
    df_esami_lab_par = df_list["esamilaboratorioparametri_c"].result()
    df_esami_lab_par_cal = df_list["esamilaboratorioparametricalcolati_c"].result()
    df_esami_stru = df_list["esamistrumentali_c"].result()
    df_pres_diab_farm = df_list["prescrizionidiabetefarmaci_c"].result()
    df_pres_diab_no_farm = df_list["prescrizionidiabetenonfarmaci_c"].result()
    df_pres_no_diab = df_list["prescrizioninondiabete_c"].result()

list_of_df = [
    df_diagnosi,
    df_esami_lab_par,
    df_esami_lab_par_cal,
    df_esami_stru,
    df_pres_diab_farm,
    df_pres_diab_no_farm,
    df_pres_no_diab,
]


# Casting "data" features to datetime in all tables except "anagrafica"
def cast_to_datetime(df, col, format="%Y-%m-%d"):
    df[col] = pd.to_datetime(df[col], format=format)
    return df[col]


for df in list_of_df:
    df["data"] = cast_to_datetime(df, "data", format="%Y-%m-%d")

# Casting also "anagrafica" dates
for col in ["annonascita", "annoprimoaccesso", "annodecesso", "annodiagnosidiabete"]:
    df_anagrafica[col] = cast_to_datetime(df_anagrafica, col, format="%Y-%m-%d")

del file_names, df_list

#######################################
############### STEP 1 ################
#######################################

print(df_anagrafica.label.value_counts())
if df_anagrafica.label.unique().size > 2:
    # print("Error: more than 2 different labels")
    raise ("Error: more than 2 different labels")


df_anagrafica_label_0 = df_anagrafica[df_anagrafica.label == False]
df_anagrafica_label_1 = df_anagrafica[df_anagrafica.label == True]

print(
    f"Number of records in anagrafica that have label equal to 0: {len(df_anagrafica_label_0)}"
)
print(
    f"Number of records in anagrafica that have label equal to 1: {len(df_anagrafica_label_1)}"
)

esami_and_prescrizioni_concat = pd.concat(
    objs=(
        idf.set_index(["idana", "idcentro"])
        for idf in [
            df_diagnosi[["idana", "idcentro", "data"]],
            df_esami_lab_par[["idana", "idcentro", "data"]],
            df_esami_lab_par_cal[["idana", "idcentro", "data"]],
            df_esami_stru[["idana", "idcentro", "data"]],
        ]
    ),
    join="inner",
).reset_index()

if PRESCRIZIONI:
    esami_and_prescrizioni_concat = pd.concat(
        objs=(
            idf.set_index(["idana", "idcentro"])
            for idf in [
                esami_and_prescrizioni_concat[["idana", "idcentro", "data"]],
                df_pres_diab_farm[["idana", "idcentro", "data"]],
                df_pres_diab_no_farm[["idana", "idcentro", "data"]],
                df_pres_no_diab[["idana", "idcentro", "data"]],
            ]
        ),
        join="inner",
    ).reset_index()

last_event = esami_and_prescrizioni_concat.groupby(["idana", "idcentro"]).max()

last_event_label_0_keys = df_anagrafica_label_0[["idana", "idcentro"]].merge(
    last_event, on=["idana", "idcentro"]
)


# for each given a dataset, drop all the rows for negative patient
# that have a date in the last 6 months
def drop_last_six_months(df: pd.DataFrame) -> pd.DataFrame:
    df_label_0_last_event = df.merge(
        last_event_label_0_keys,
        on=["idana", "idcentro"],
        how="left",
        suffixes=("_left", "_right"),
    )

    temp = df_label_0_last_event["data_left"] >= (
        df_label_0_last_event["data_right"] - np.timedelta64(6, "M")
    )
    df = df.drop(temp[temp].index)
    return df


print("Before: ", len(df_diagnosi))
df_diagnosi = drop_last_six_months(df_diagnosi)
print("After: ", len(df_diagnosi))

print("Before: ", len(df_esami_lab_par))
df_esami_lab_par = drop_last_six_months(df_esami_lab_par)
print("After: ", len(df_esami_lab_par))

print("Before: ", len(df_esami_lab_par_cal))
df_esami_lab_par_cal = drop_last_six_months(df_esami_lab_par_cal)
print("After: ", len(df_esami_lab_par_cal))

print("Before: ", len(df_esami_stru))
df_esami_stru = drop_last_six_months(df_esami_stru)
print("After: ", len(df_esami_stru))

if PRESCRIZIONI:
    print("Before: ", len(df_pres_diab_farm))
    df_pres_diab_farm = drop_last_six_months(df_pres_diab_farm)
    print("After: ", len(df_pres_diab_farm))

    print("Before: ", len(df_pres_diab_no_farm))
    df_pres_diab_no_farm = drop_last_six_months(df_pres_diab_no_farm)
    print("After: ", len(df_pres_diab_no_farm))

    print("Before: ", len(df_pres_no_diab))
    df_pres_no_diab = drop_last_six_months(df_pres_no_diab)
    print("After: ", len(df_pres_no_diab))

# for positive patient we must drop only the cardiovascular event
# in the last 6 month insead of all history, this is a request from specifics
print("Diagnosis before: ", len(df_diagnosi))

df_diagnosi = df_diagnosi.merge(
    last_event, on=["idana", "idcentro"], how="left", suffixes=("_left", "_right")
)

df_diagnosi = df_diagnosi.merge(
    df_anagrafica[["idana", "idcentro", "label"]], on=["idana", "idcentro"], how="left"
)

df_diagnosi["delete"] = (
    (df_diagnosi["codiceamd"].isin(AMD_OF_CARDIOVASCULAR_EVENT))
    & (df_diagnosi["label"] == True)
    & (df_diagnosi["data_left"] >= (df_diagnosi["data_right"] - pd.Timedelta(days=186)))
)

df_diagnosi = (
    df_diagnosi[(df_diagnosi["delete"] == False)]
    .drop(["delete", "label", "data_right"], axis=1)
    .rename({"data_left": "data"}, axis=1)
)

print("Diagnosis after: ", len(df_diagnosi))

del last_event, last_event_label_0_keys

if BALANCING == "lossy":
    temp_balanced_aa = df_anagrafica_label_1.sample(
        n=len(df_anagrafica_label_0), random_state=rng
    )
    balanced_aa = pd.concat([temp_balanced_aa, df_anagrafica_label_0])
    print(balanced_aa.label.value_counts())
    balanced_aa_keys = balanced_aa[["idana", "idcentro"]].drop_duplicates()
    df_diagnosi = df_diagnosi.merge(
        balanced_aa_keys, on=["idcentro", "idana"], how="inner"
    )
    df_esami_lab_par = df_esami_lab_par.merge(
        balanced_aa_keys, on=["idcentro", "idana"], how="inner"
    )
    df_esami_lab_par_cal = df_esami_lab_par_cal.merge(
        balanced_aa_keys, on=["idcentro", "idana"], how="inner"
    )
    df_esami_stru = df_esami_stru.merge(
        balanced_aa_keys, on=["idcentro", "idana"], how="inner"
    )
    df_pres_diab_farm = df_pres_diab_farm.merge(
        balanced_aa_keys, on=["idcentro", "idana"], how="inner"
    )
    df_pres_diab_no_farm = df_pres_diab_no_farm.merge(
        balanced_aa_keys, on=["idcentro", "idana"], how="inner"
    )
    df_pres_no_diab = df_pres_no_diab.merge(
        balanced_aa_keys, on=["idcentro", "idana"], how="inner"
    )

elif BALANCING == "standard":
    duplication_factor = int(len(df_anagrafica_label_0) / len(df_anagrafica_label_1))

    duplicated_df_anagrafica_label_1 = pd.concat(
        [df_anagrafica_label_1] * duplication_factor,
        ignore_index=True,
    )
    duplicated_df_anagrafica_label_1[
        "duplicated"
    ] = duplicated_df_anagrafica_label_1.duplicated()
    # assign a counter to each duplicated row
    duplicated_df_anagrafica_label_1["duplicate_identifier"] = (
        duplicated_df_anagrafica_label_1[duplicated_df_anagrafica_label_1["duplicated"]]
        .groupby(["idana", "idcentro"])
        .cumcount()
        .add(1)
    )
    # print(
    #     duplicated_df_anagrafica_label_1.sort_values(by=["idana", "idcentro"]).head(30)
    # )

    def balance(df, fraction):
        # fraction is the quantity of events to keep
        # merge with other df
        new_dup_record = df.merge(
            duplicated_df_anagrafica_label_1[
                ["idana", "idcentro", "duplicated", "duplicate_identifier"]
            ],
            on=["idana", "idcentro"],
            how="inner",
        )
        # delete events at random
        new_dup_record = new_dup_record[new_dup_record["duplicated"]].sample(
            frac=fraction, random_state=rng
        )

        # shuffle events data
        noise = pd.to_timedelta(
            rng.normal(0, 5, len(new_dup_record)).astype("int"), unit="d"
        )
        new_dup_record["data"] = new_dup_record["data"] + noise
        # TODO: while here we are adding noise to the date, we should ensure that the new date is not
        # in the 6 months after last event, to ensure we are not creating other False.
        # Maybe do also here a drop_last_six_months? nah this is improbable to happen because the
        # noise is small and also the probability of having a date in the 6 months after the last event
        # new_dup_record = drop_last_six_months(new_dup_record)

        # the idana is negative for th duplicate to easly distinguish it from the original
        # the 10000 is a number greather than the max value of number of patient in each idcentro
        new_dup_record["idana"] = -(
            new_dup_record["idana"].astype("int")
            + 100000 * new_dup_record["duplicate_identifier"].astype("int")
        )
        # remove the duplicate identifier and duplicated columns
        new_dup_record = new_dup_record.drop(
            ["duplicate_identifier", "duplicated"], axis=1
        )
        # append the new df to the original one
        df = pd.concat([df, new_dup_record], ignore_index=True)
        return df

    print("Before balance: ", len(df_anagrafica))
    new_dup_record = duplicated_df_anagrafica_label_1[
        duplicated_df_anagrafica_label_1["duplicated"]
    ]

    new_dup_record["idana"] = -(
        new_dup_record["idana"].astype("int")
        + 100000 * new_dup_record["duplicate_identifier"].astype("int")
    )

    new_dup_record = new_dup_record.drop(["duplicate_identifier", "duplicated"], axis=1)
    df_anagrafica = pd.concat([df_anagrafica, new_dup_record], ignore_index=True)
    print("After balance: ", len(df_anagrafica))
    print(df_anagrafica.label.value_counts())
    print(df_anagrafica.head())
    print(df_anagrafica.tail(10))

    print("Before balance: ", len(df_diagnosi))
    df_diagnosi = balance(df_diagnosi, 0.50)
    print("After balance: ", len(df_diagnosi))

    print("Before balance: ", len(df_esami_lab_par))
    df_esami_lab_par = balance(df_esami_lab_par, 0.50)
    print("After balance: ", len(df_esami_lab_par))

    print("Before balance: ", len(df_esami_lab_par_cal))
    df_esami_lab_par_cal = balance(df_esami_lab_par_cal, 0.50)
    print("After balance: ", len(df_esami_lab_par_cal))

    print("Before balance: ", len(df_esami_stru))
    df_esami_stru = balance(df_esami_stru, 0.50)
    print("After balance: ", len(df_esami_stru))

    if PRESCRIZIONI:
        print("Before balance: ", len(df_pres_diab_farm))
        df_pres_diab_farm = balance(df_pres_diab_farm, 0.50)
        print("After balance: ", len(df_pres_diab_farm))

        print("Before balance: ", len(df_pres_diab_no_farm))
        df_pres_diab_no_farm = balance(df_pres_diab_no_farm, 0.50)
        print("After balance: ", len(df_pres_diab_no_farm))

        print("Before balance: ", len(df_pres_no_diab))
        df_pres_no_diab = balance(df_pres_no_diab, 0.50)
        print("After balance: ", len(df_pres_no_diab))
    if WRITE_CSV:
        print("Exporting the cleaned datasets...")
        dict_file_names = {
            f"anagraficapazientiattivi_b{'_pres' if PRESCRIZIONI else ''}": df_anagrafica,
            f"diagnosi_b{'_pres' if PRESCRIZIONI else ''}": df_diagnosi,
            f"esamilaboratorioparametri_b{'_pres' if PRESCRIZIONI else ''}": df_esami_lab_par,
            f"esamilaboratorioparametricalcolati_b{'_pres' if PRESCRIZIONI else ''}": df_esami_lab_par_cal,
            f"esamistrumentali_b{'_pres' if PRESCRIZIONI else ''}": df_esami_stru,
            f"prescrizionidiabetefarmaci_b{'_pres' if PRESCRIZIONI else ''}": df_pres_diab_farm,
            f"prescrizionidiabetenonfarmaci_b{'_pres' if PRESCRIZIONI else ''}": df_pres_diab_no_farm,
            f"prescrizioninondiabete_b{'_pres' if PRESCRIZIONI else ''}": df_pres_no_diab,
        }

        for i, (df_name, df) in enumerate(dict_file_names.items()):
            df.to_csv(f"{WRITE_DATA_PATH}/{df_name}.csv", index=False)
            print(f"{df_name}.csv exported ({i+1}/{len(dict_file_names)})")
        print("Exporting completed!")


def progressBar(count_value, total, suffix=""):
    bar_length = 100
    filled_up_Length = int(round(bar_length * count_value / float(total)))
    percentage = round(100.0 * count_value / float(total), 1)
    bar = "=" * filled_up_Length + "-" * (bar_length - filled_up_Length)
    sys.stdout.write("[%s] %s%s ...%s\r" % (bar, percentage, "%", suffix))
    sys.stdout.flush()


van_val = 0.1
van_test = 0.3
van_train = 1 - van_test - van_val
if VANILLA_LSTM:
    if not LOAD_VANILLA_DF:
        vanilla_df = Vanilla_LSTM.create_dataset(
            df_anagrafica,
            df_diagnosi,
            df_esami_lab_par,
            df_esami_lab_par_cal,
            df_esami_stru,
            df_pres_diab_farm,
            df_pres_diab_no_farm,
            df_pres_no_diab,
        )
        if SAVE_VANILLA_DF:
            vanilla_df.to_csv(f"{LSTM_DF}/vanilla_df.csv", index=False)
            print(f"vanilla_df.csv exported")
    else:
        print("loading vanilla data")

        vanilla_df = read_csv("{LSTM_DF}/vanilla_df.csv")

        vanilla_df = vanilla_df.fillna(-100)

    if DROP_ANNI:
        vanilla_df = vanilla_df.drop(
            columns=[
                "annonascita",
                "annoprimoaccesso",
                "annodecesso",
                "annodiagnosidiabete",
            ]
        )

    len_input = len(vanilla_df.columns) - 4  # 13
    vanilla_model = Vanilla_LSTM.LightingVanillaLSTM(
        input_size=len_input, hidden_size=512
    )
    grouped_vanilla = vanilla_df.groupby(["idana", "idcentro"], group_keys=True)
    inputs = []
    labels = []
    max_history_len = 0
    count = 0

    for name, group in grouped_vanilla:
        if group.values.shape[0] > max_history_len:
            max_history_len = group.values.shape[0]

    k = 2
    while k * 2 < max_history_len:
        k = k * 2
    altrocount = 0
    print("k max history len: ", k)
    for name, group in grouped_vanilla:
        vanilla_patient_hystory = group.sort_values(by=["data"])
        labels.append(vanilla_patient_hystory["label"].values[0])
        vanilla_patient_hystory = vanilla_patient_hystory.drop(
            columns=["idana", "idcentro", "label", "data"]
        )

        if vanilla_patient_hystory.values.shape[0] > k:
            altrocount += 1
            inputs.append(vanilla_patient_hystory.values[k:])
        else:
            inputs.append(vanilla_patient_hystory.values)
        count += 1
        progressBar(count, len(grouped_vanilla))

    print("altrocount: ", altrocount)

    tensor_list = [
        torch.cat(
            (
                torch.zeros(max_history_len - len(sublist), len_input) - 200.0,
                torch.tensor(sublist),
            )
        )
        for sublist in inputs
    ]
    padded_tensor = pad_sequence(tensor_list, batch_first=True)  # batch_first=True
    # padded_tensor = padded_tensor.to(torch.long)
    padded_tensor = padded_tensor.to(torch.float32)
    bool_tensor = torch.tensor(labels, dtype=torch.bool)
    bool_tensor = torch.tensor(labels, dtype=torch.float32)
    print("Valori unici in bool_tensor:")
    print(torch.unique(bool_tensor, return_counts=True))

    # Now you can use train_loader, val_loader, and test_loader for training, validation, and testing.
    vanilla_dataset = Vanilla_LSTM.TensorDataset(padded_tensor, bool_tensor)

    # Define the sizes for train, validation, and test sets
    train_size = int(van_train * len(vanilla_dataset))
    val_size = int(van_val * len(vanilla_dataset))
    test_size = len(vanilla_dataset) - train_size - val_size

    # Split the dataset into train, validation, and test sets
    vanilla_train_dataset, vanilla_test_dataset, vanilla_val_dataset = random_split(
        vanilla_dataset, [train_size, test_size, val_size]
    )

    # Create DataLoader instances for train, validation, and test sets
    batch_size = 16  # Adjust as needed
    train_loader = DataLoader(
        vanilla_train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(vanilla_val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(vanilla_test_dataset, batch_size=batch_size, shuffle=True)

    vanilla_train_loader = Vanilla_LSTM.DataLoader(
        vanilla_train_dataset, batch_size=batch_size, shuffle=True
    )
    vanilla_val_loader = Vanilla_LSTM.DataLoader(
        vanilla_val_dataset, batch_size=batch_size, shuffle=True
    )
    vanilla_test_loader = Vanilla_LSTM.DataLoader(
        vanilla_test_dataset, batch_size=batch_size, shuffle=True
    )
    Vanilla_LSTM.evaluate_vanilla_LSTM(
        vanilla_model,
        train=vanilla_train_dataset,
        test=vanilla_test_dataset,
        val=vanilla_val_loader,
    )
    torch.save(vanilla_model.state_dict(), "vanilla_lstm")

#####################
# PubMedBERT
#####################


if BERT_DATASET:
    tuple_dataset = []

    if CREATE_BERT_DATASET:
        amd = pd.read_csv("amd_codes_for_bert.csv").rename(
            {"codice": "codiceamd"}, axis=1
        )
        atc = pd.read_csv("atc_info_nodup.csv")
        # Converting Dataset for Deep Learning purposes

        # NOTE: Utilizzare sempre gli stessi nomi dal primo all'ultimo task non è più semplice?
        #       Sia per noi nello scrivere il codice che per chi lo legge
        # Non per BERT che deve leggere i nomi italiani delle colonne e il contenuto in inglese
        # e lui è addestrato su dataset in inglese quindi non ha senso che gli diamo i nomi in italiano
        df_anagrafica = (
            df_anagrafica[
                [
                    "idcentro",
                    "idana",
                    "sesso",
                    "annodiagnosidiabete",
                    "scolarita",
                    "statocivile",
                    "professione",
                    "annonascita",
                    "annoprimoaccesso",
                    "annodecesso",
                    "label",
                ]
            ]
            .rename(
                {
                    "idcentro": "idcenter",
                    "idana": "idpatient",
                    "sesso": "sex",
                    "annodiagnosidiabete": "yeardiagnosisdiabetes",
                    "scolarita": "levelofeducation",
                    "statocivile": "maritalstatus",
                    "professione": "profession",
                    "annonascita": "yearofbirth",
                    "annoprimoaccesso": "yearfirstaccess",
                    "annodecesso": "yearofdeath",
                    "label": "label",
                },
                axis=1,
            )
            .fillna("[UNK]")
        )

        df_diagnosi = df_diagnosi.merge(amd, on="codiceamd", how="left")
        # reindexing the columns so we have in the string codiceamd: x meaning: y valore: z
        df_diagnosi = (
            df_diagnosi[["idcentro", "idana", "data", "codiceamd", "meaning", "valore"]]
            .rename(
                {
                    "idcentro": "idcenter",
                    "idana": "idpatient",
                    "data": "date",
                    "codiceamd": "amdcode",
                    "meaning": "meaning",
                    "valore": "value",
                },
                axis=1,
            )
            .fillna("[UNK]")
        )

        df_esami_lab_par = df_esami_lab_par.merge(amd, on="codiceamd", how="left")
        df_esami_lab_par = (
            df_esami_lab_par[
                ["idcentro", "idana", "data", "codiceamd", "meaning", "valore"]
            ]
            .rename(
                {
                    "idcentro": "idcenter",
                    "idana": "idpatient",
                    "data": "date",
                    "codiceamd": "amdcode",
                    "meaning": "meaning",
                    "valore": "value",
                },
                axis=1,
            )
            .fillna("[UNK]")
        )

        df_esami_lab_par_cal = df_esami_lab_par_cal.merge(
            amd, on="codiceamd", how="left"
        )
        df_esami_lab_par_cal = (
            df_esami_lab_par_cal[
                [
                    "idcentro",
                    "idana",
                    "data",
                    "codiceamd",
                    "codicestitch",
                    "meaning",
                    "valore",
                ]
            ]
            .rename(
                {
                    "idcentro": "idcenter",
                    "idana": "idpatient",
                    "data": "date",
                    "codiceamd": "amdcode",
                    "codicestitch": "stitchcode",
                    "meaning": "meaning",
                    "valore": "value",
                },
                axis=1,
            )
            .fillna("[UNK]")
        )

        df_esami_stru = df_esami_stru.merge(amd, on="codiceamd", how="left")
        df_esami_stru = (
            df_esami_stru[
                ["idcentro", "idana", "data", "codiceamd", "meaning", "valore"]
            ]
            .rename(
                {
                    "idcentro": "idcenter",
                    "idana": "idpatient",
                    "data": "date",
                    "codiceamd": "amdcode",
                    "meaning": "meaning",
                    "valore": "value",
                },
                axis=1,
            )
            .fillna("[UNK]")
        )

        # this is the only one that has the codiceatc column and no codice amd
        df_pres_diab_farm = (
            df_pres_diab_farm.merge(
                atc[["codiceatc", "atc_nome"]], on="codiceatc", how="left"
            )[
                [
                    "idcentro",
                    "idana",
                    "data",
                    "codiceatc",
                    "atc_nome",
                    "quantita",
                    "idpasto",
                    "descrizionefarmaco",
                ]
            ]
            .rename(
                {
                    "idcentro": "idcenter",
                    "idana": "idpatient",
                    "data": "date",
                    "codiceatc": "atccode",
                    "atc_nome": "meaning",
                    "quantita": "quantity",
                    "idpasto": "idmeal",
                    "descrizionefarmaco": "drugdescription",
                },
                axis=1,
            )
            .fillna("[UNK]")
        )

        df_pres_diab_no_farm = df_pres_diab_no_farm.merge(
            amd, on="codiceamd", how="left"
        )
        df_pres_diab_no_farm = (
            df_pres_diab_no_farm[
                ["idcentro", "idana", "data", "codiceamd", "meaning", "valore"]
            ]
            .rename(
                {
                    "idcentro": "idcenter",
                    "idana": "idpatient",
                    "data": "date",
                    "codiceamd": "amdcode",
                    "meaning": "meaning",
                    "valore": "value",
                },
                axis=1,
            )
            .fillna("[UNK]")
        )

        df_pres_no_diab = df_pres_no_diab.merge(amd, on="codiceamd", how="left")
        df_pres_no_diab = (
            df_pres_no_diab[
                ["idcentro", "idana", "data", "codiceamd", "meaning", "valore"]
            ]
            .rename(
                {
                    "idcentro": "idcenter",
                    "idana": "idpatient",
                    "data": "date",
                    "codiceamd": "amdcode",
                    "meaning": "meaning",
                    "valore": "value",
                },
                axis=1,
            )
            .fillna("[UNK]")
        )

        print("end rebuilding dataframes")

        dict_list_of_df = {
            "diagnosis": df_diagnosi,
            "exam parameter": df_esami_lab_par,
            "exam parameter calculated": df_esami_lab_par_cal,
            "exam strumental": df_esami_stru,
            "prescription diabete drugs": df_pres_diab_farm,
            "prescription diabete not drugs": df_pres_diab_no_farm,
            "prescription not diabete": df_pres_no_diab,
        }

        df_anagrafica_no_label = df_anagrafica[
            [
                "idcenter",
                "idpatient",
                "sex",
                "yeardiagnosisdiabetes",
                "levelofeducation",
                "maritalstatus",
                "profession",
                "yearofbirth",
                "yearfirstaccess",
                "yearofdeath",
            ]
        ]

        # import time
        # start_time = time.time()

        def create_history_string(patient):
            df_anagrafica_filtered = df_anagrafica_no_label.loc[
                (df_anagrafica_no_label["idcenter"] == patient[0])
                & (df_anagrafica_no_label["idpatient"] == patient[1])
            ]

            history = "".join(
                [
                    re.sub(
                        r"\s*idpatient\s*-*\d+,*|\s*00:00:00",
                        "",
                        f"{row}, ".replace("(", " ", 1)
                        .replace("Timestamp(", "")
                        .replace(")", "")
                        .replace("'", "")
                        .replace(",", "")
                        .replace("=", " "),
                    )
                    for row in df_anagrafica_filtered.itertuples(
                        index=False, name="patientregistry"
                    )
                ]
            )

            temp = []
            # Iterate over each DataFrame in the dictionary dict_list_of_df
            for name, df in dict_list_of_df.items():
                # Filter the rows based on patient's ID and sort by date
                df_filtered = df.loc[
                    (df["idcenter"] == patient[0]) & (df["idpatient"] == patient[1])
                ].sort_values(by="date", ascending=False)

                # Extract relevant information
                info = [
                    re.sub(
                        r"\s*idpatient\s*-*\d+,*\s*|\s*idcenter\s*-*\d+,*\s*|\s*00:00:00|Pandas|Timestamp\(|\(|\)|'|,",
                        "",
                        f"{row},".replace("=", " "),
                    )
                    for row in df_filtered.itertuples(index=False)
                ]

                # Add the formatted information to the temp list
                temp.extend([f"{name}"] + info)

            # Combine the elements in the temp list into a single string
            history += " ".join(temp)
            return history

        if not (PARALLEL_LOAD_DATASET):
            # sequential version
            for i, patient in enumerate(
                df_anagrafica[["idcenter", "idpatient"]].drop_duplicates()[:500].values
            ):
                # print(i)
                # Get patient history as a string from df_anagrafica and other DataFrames
                history_of_patient = create_history_string(patient)

                # Get label
                label = int(
                    df_anagrafica.loc[
                        (df_anagrafica["idcenter"] == patient[0])
                        & (df_anagrafica["idpatient"] == patient[1])
                    ]["label"].item()
                )

                tuple_dataset.append((history_of_patient, label))
        elif PARALLEL_LOAD_DATASET:
            # parallel version
            # Okay this code with a pc of 12 core is resonable fast, 1000 patient in 11 seconds
            # so I estimate a total of 15 minutes for all the patient
            def process_patient(patient):
                history_of_patient = create_history_string(patient)
                label = int(
                    df_anagrafica.loc[
                        (df_anagrafica["idcenter"] == patient[0])
                        & (df_anagrafica["idpatient"] == patient[1])
                    ]["label"].item()
                )
                return (history_of_patient, label)

            patients = df_anagrafica[["idcenter", "idpatient"]].drop_duplicates().values

            with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
                tuple_dataset = pool.map(process_patient, patients)

        print("dataset: ", len(tuple_dataset))
        # dataset now contains a list of tuples, each containing the patient history string and their label
        print(tuple_dataset[:1])

        # end_time = time.time()
        # execution_time = end_time - start_time
        # print(f"Execution Time: {execution_time:.6f} seconds")
        if WRITE_DATASET:
            with open(DATASET_NAME, "wb") as f:
                pickle.dump(tuple_dataset, f)
            print("stored dataset")
    else:
        with open(DATASET_NAME, "rb") as f:
            tuple_dataset = pickle.load(f)

        print("loaded dataset")
        print("dataset: ", len(tuple_dataset))
        # print(tuple_dataset[:1])


def convert_to_huggingfaceDataset(tuple_dataset):
    # here data is a list of tuples,
    # each containing the patient history string and their label
    # we need to convert it to a hugginface dataset
    dict_list = [{"label": data[1], "text": data[0]} for data in tuple_dataset]
    dataset = datasets.Dataset.from_list(dict_list)
    return dataset


class PubMedBERTDataModule(LightningDataModule):
    def __init__(
        self,
        tuple_dataset,
        model_name_with_path: str,
        max_seq_length: int = 512,  # 512 is the max length of BERT and PubMedBERT but I need 32768
        train_batch_size: int = 8,
        eval_batch_size: int = 8,
        **kwargs,
    ):
        super().__init__()
        self.model_name_with_path = model_name_with_path
        self.max_seq_length = max_seq_length
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_with_path, use_fast=True
        )

    def setup(self, stage=None):
        dataset = convert_to_huggingfaceDataset(tuple_dataset)
        tokenized_dataset = dataset.map(
            self.convert_to_features,
            batched=True,
            remove_columns=["text", "label"],
        )
        tokenized_dataset.set_format(type="torch")

        # split dataset into train and validation sampling randomly
        # use 20% of training data for validation
        train_set_size = int(len(tokenized_dataset) * 0.8)
        valid_set_size = len(tokenized_dataset) - train_set_size

        # split the dataset randomly into two
        self.train_data, self.valid_data = torch.utils.data.random_split(
            tokenized_dataset, [train_set_size, valid_set_size], generator=GEN_SEED
        )

    def prepare_data(self):
        AutoTokenizer.from_pretrained(
            self.model_name_with_path,
            use_fast=True,
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

    def convert_to_features(self, example_batch, indices=None):
        # Tokenize the patient history
        features = self.tokenizer(
            text=example_batch["text"],
            max_length=self.max_seq_length,
            padding="longest",
            truncation=True,
            return_tensors="pt",
        )
        # Rename label to labels to make it easier to pass to model forward
        features["labels"] = example_batch["label"]

        return features


class PubMedBERTTransformer(LightningModule):
    def __init__(
        self,
        model_name_or_path: str,
        num_labels: int = 2,  # It will create a PubMedBERT model (in our case) instance with encoder weights copied from the PubMedBERT model and a randomly initialized sequence classification head on top of the encoder with an output size of 2
        learning_rate: float = 2e-5,
        adam_epsilon: float = 1e-8,
        warmup_steps: int = 0,
        weight_decay: float = 0.0,
        **kwargs,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.config = AutoConfig.from_pretrained(
            model_name_or_path, num_labels=num_labels
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path, config=self.config
        )
        self.train_acc_metric = BinaryAccuracy()
        self.val_acc_metric = BinaryAccuracy()
        self.train_f1_metric = BinaryF1Score()
        self.val_f1_metric = BinaryF1Score()

    def forward(self, **inputs):
        return self.model(**inputs)

    def step(self, batch):
        outputs = self(**batch)
        loss, logits = outputs[:2]
        if self.hparams.num_labels > 1:
            preds = logits.argmax(axis=1)
        elif self.hparams.num_labels == 1:
            preds = logits.squeeze()
        labels = batch["labels"]
        return {"loss": loss, "logits": logits, "preds": preds, "labels": labels}

    def training_step(self, batch, batch_idx):
        outputs = self.step(batch)
        self.train_acc_metric(outputs["preds"], outputs["labels"])
        self.train_f1_metric(outputs["preds"], outputs["labels"])
        self.log(
            "train_acc",
            self.train_acc_metric,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
        )
        self.log(
            "train_f1", self.train_f1_metric, on_step=True, on_epoch=True, prog_bar=True
        )
        self.log(
            "train_loss", outputs["loss"], on_step=True, on_epoch=True, prog_bar=True
        )
        return outputs["loss"]

    def validation_step(self, batch, batch_idx):
        outputs = self.step(batch)
        self.val_acc_metric(outputs["preds"], outputs["labels"])
        self.val_f1_metric(outputs["preds"], outputs["labels"])
        self.log("val_acc", self.val_acc_metric, on_epoch=True, prog_bar=True)
        self.log("val_f1", self.val_f1_metric, on_epoch=True, prog_bar=True)
        self.log("val_loss", outputs["loss"], on_epoch=True, prog_bar=True)
        return {
            "loss": outputs["loss"],
            "preds": outputs["preds"],
            "labels": outputs["labels"],
        }

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

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]


def evaluate_PubMedBERT():
    dm = PubMedBERTDataModule(tuple_dataset, MODEL_NAME)

    model = PubMedBERTTransformer(
        model_name_or_path=MODEL_NAME,
    )

    checkpoint_callback = ModelCheckpoint(monitor="val_f1", mode="max")

    trainer = Trainer(
        max_epochs=3,
        accelerator="auto",
        devices="auto",
        benchmark=True,
        precision="16-mixed",
        callbacks=[checkpoint_callback],
    )
    trainer.fit(model=model, datamodule=dm)

    return


def training_tlstm(
    data_train_batches,
    labels_train_batches,
    elapsed_train_batches,
    number_train_batches,
    learning_rate,
    training_epochs,
    train_dropout_prob,
    hidden_dim,
    fc_dim,
    key,
    model_path="./tlstm_dir/tlstm_model",
):
    print("Training TLSTM")

    input_dim = data_train_batches[0].shape[2]
    output_dim = labels_train_batches[0].shape[1]

    lstm = TLSTM.TLSTM(input_dim, output_dim, hidden_dim, fc_dim, key)

    cross_entropy, y_pred, y, logits, labels = lstm.get_cost_acc()
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate).minimize(
        cross_entropy
    )

    init = tf.compat.v1.global_variables_initializer()
    saver = tf.compat.v1.train.Saver()

    with tf.compat.v1.Session() as sess:
        sess.run(init)
        for epoch in range(training_epochs):  #
            # Loop over all batches
            total_cost = 0
            for i in range(number_train_batches):  #
                # batch_xs is [number of patients x sequence length x input dimensionality]
                batch_xs, batch_ys, batch_ts = (
                    data_train_batches[i],
                    labels_train_batches[i],
                    elapsed_train_batches[i],
                )
                batch_ts = np.reshape(batch_ts, [batch_ts.shape[0], batch_ts.shape[1]])
                sess.run(
                    optimizer,
                    feed_dict={
                        lstm.input: batch_xs,
                        lstm.labels: batch_ys,
                        lstm.keep_prob: train_dropout_prob,
                        lstm.time: batch_ts,
                    },
                )
                print(f"Epoch: {epoch + 1} batch: {i}")

        print("Training is over!")
        saver.save(sess, model_path)


def testing_tlstm(
    data_train_batches,
    labels_train_batches,
    elapsed_train_batches,
    number_train_batches,
    train_dropout_prob,
    hidden_dim,
    fc_dim,
    key,
    model_path="./tlstm_dir/tlstm_model",
):
    print("Testing TLSTM")
    input_dim = data_train_batches[0].shape[2]
    output_dim = labels_train_batches[0].shape[1]

    lstm = TLSTM.TLSTM(input_dim, output_dim, hidden_dim, fc_dim, key)

    tf.compat.v1.disable_eager_execution()
    sav = tf.compat.v1.train.import_meta_graph(f"{model_path}.meta")

    with tf.compat.v1.Session() as sess:
        sav.restore(
            sess, tf.compat.v1.train.latest_checkpoint(f"./{model_path.split('/')[1]}/")
        )

        Y_pred = []
        Y_true = []
        Labels = []
        Logits = []

        for i in range(number_train_batches):  #
            batch_xs, batch_ys, batch_ts = (
                data_train_batches[i],
                labels_train_batches[i],
                elapsed_train_batches[i],
            )
            batch_ts = np.reshape(batch_ts, [batch_ts.shape[0], batch_ts.shape[1]])
            c_train, y_pred_train, y_train, logits_train, labels_train = sess.run(
                lstm.get_cost_acc(),
                feed_dict={
                    lstm.input: batch_xs,
                    lstm.labels: batch_ys,
                    lstm.keep_prob: train_dropout_prob,
                    lstm.time: batch_ts,
                },
            )
            print(f"Validation batch: {i}")

            if i > 0:
                Y_true = np.concatenate([Y_true, y_train], 0)
                Y_pred = np.concatenate([Y_pred, y_pred_train], 0)
                Labels = np.concatenate([Labels, labels_train], 0)
                Logits = np.concatenate([Logits, logits_train], 0)
            else:
                Y_true = y_train
                Y_pred = y_pred_train
                Labels = labels_train
                Logits = logits_train

        total_acc = accuracy_score(Y_true, Y_pred)
        print("Train Accuracy = {:.3f}".format(total_acc))
        total_auc = roc_auc_score(Labels, Logits, average="micro")
        print("Train AUC = {:.3f}".format(total_auc))
        total_auc_macro = roc_auc_score(Labels, Logits, average="macro")
        print("Train AUC Macro = {:.3f}".format(total_auc_macro))


def evaluate_T_LSTM():
    df = TLSTM.create_dataset(
        df_anagrafica,
        df_diagnosi,
        df_esami_lab_par,
        df_esami_lab_par_cal,
        df_esami_stru,
        df_pres_diab_farm,
        df_pres_diab_no_farm,
        df_pres_no_diab,
    )
    print(df.head(2))

    feature, labels, elapsed_time = TLSTM.create_tensor_dataset(df)

    len_batch = 7
    num_batch = len(feature) // len_batch
    print("num_batch: ", num_batch)

    def split_padded(a, n):
        print(a.shape)
        padding = (-len(a)) % n
        if len(a.shape) == 1:
            return np.array(np.array_split(np.concatenate((a, np.zeros(padding))), n))
        elif len(a.shape) == 2:
            return np.array(
                np.array_split(np.concatenate((a, np.zeros((padding, a.shape[1])))), n)
            )
        elif len(a.shape) == 3:
            return np.array(
                np.array_split(
                    np.concatenate((a, np.zeros((padding, a.shape[1], a.shape[2])))), n
                )
            )
        else:
            raise ValueError("The input array must be 1D, 2D or 3D")

    train_data_batches = split_padded(feature, num_batch)
    labels_train_batches = split_padded(labels, num_batch)
    elapsed_train_batches = split_padded(elapsed_time, num_batch)
    print("train_data_batches: ", train_data_batches.shape)

    learning_rate = 1e-3
    training_epochs = 1
    dropout_prob = 1.0
    hidden_dim = 128
    fc_dim = 64
    training_mode = 1

    if TRAIN_TLSTM:
        training_tlstm(
            train_data_batches,
            labels_train_batches,
            elapsed_train_batches,
            num_batch,
            learning_rate,
            training_epochs,
            dropout_prob,
            hidden_dim,
            fc_dim,
            training_mode,
        )

    len_val_batch = 63
    num_val_batch = len(feature) // len_val_batch
    num_batch_to_select = 300
    val_data_batches = split_padded(feature, num_val_batch)[:num_batch_to_select]
    labels_val_batches = split_padded(labels, num_val_batch)[:num_batch_to_select]
    elapsed_val_batches = split_padded(elapsed_time, num_val_batch)[
        :num_batch_to_select
    ]
    print("val_data_batches: ", val_data_batches.shape)

    testing_tlstm(
        val_data_batches,
        labels_val_batches,
        elapsed_val_batches,
        num_batch_to_select,
        dropout_prob,
        hidden_dim,
        fc_dim,
        training_mode,
    )

    return


if EVALUATE_TLSTM:
    evaluate_T_LSTM()
if EVALUATE_BERT:
    evaluate_PubMedBERT()

#####################
# Delta-Eta
#####################

if DELTA_ETA:
    df_anagrafica["eta"] = (
        df_anagrafica["annodecesso"].fillna(pd.Timestamp.now())
        - df_anagrafica["annonascita"]
    )  # /np.timedelta64(1, 'Y')
    print(df_anagrafica["eta"].max())
    # TODO: PARAMETRICE THIS
    MAXIMUM_AGE_FACTOR = 1.05
    maximum_age_days = (
        df_anagrafica["eta"].max() / np.timedelta64(1, "D")
    ) * MAXIMUM_AGE_FACTOR
    maximum_age_years = (
        df_anagrafica["eta"].max() / np.timedelta64(1, "Y")
    ) * MAXIMUM_AGE_FACTOR
    df_anagrafica["delta_decesso"] = (
        df_anagrafica["eta"] / np.timedelta64(1, "Y") / maximum_age_years
    )
    df_anagrafica.loc[df_anagrafica["annodecesso"].isnull(), "delta_decesso"] = np.nan
    df_anagrafica["delta_annoprimoaccesso"] = (
        (df_anagrafica["annoprimoaccesso"] - df_anagrafica["annonascita"])
        / np.timedelta64(1, "Y")
        / maximum_age_years
    )
    df_anagrafica["delta_annodiagnosidiabete"] = (
        (df_anagrafica["annodiagnosidiabete"] - df_anagrafica["annonascita"])
        / np.timedelta64(1, "Y")
        / maximum_age_years
    )

    df_anagrafica.drop(
        columns=["eta", "annoprimoaccesso", "annodiagnosidiabete", "annodecesso"],
        inplace=True,
    )

    df_diagnosi = df_diagnosi.merge(
        df_anagrafica[["idcentro", "idana", "annonascita"]],
        on=["idcentro", "idana"],
        how="left",
    )
    df_diagnosi["delta_data"] = (
        (df_diagnosi["data"] - df_diagnosi["annonascita"])
        / np.timedelta64(1, "Y")
        / maximum_age_years
    )

    df_esami_lab_par = df_esami_lab_par = df_esami_lab_par.merge(
        df_anagrafica[["idcentro", "idana", "annonascita"]],
        on=["idcentro", "idana"],
        how="left",
    )
    df_esami_lab_par["delta_data"] = (
        (df_esami_lab_par["data"] - df_esami_lab_par["annonascita"])
        / np.timedelta64(1, "Y")
        / maximum_age_years
    )

    df_esami_lab_par_cal = df_esami_lab_par_cal.merge(
        df_anagrafica[["idcentro", "idana", "annonascita"]],
        on=["idcentro", "idana"],
        how="left",
    )
    df_esami_lab_par_cal["delta_data"] = (
        (df_esami_lab_par_cal["data"] - df_esami_lab_par_cal["annonascita"])
        / np.timedelta64(1, "Y")
        / maximum_age_years
    )

    df_esami_stru = df_esami_stru.merge(
        df_anagrafica[["idcentro", "idana", "annonascita"]],
        on=["idcentro", "idana"],
        how="left",
    )
    df_esami_stru["delta_data"] = (
        (df_esami_stru["data"] - df_esami_stru["annonascita"])
        / np.timedelta64(1, "Y")
        / maximum_age_years
    )

    df_pres_diab_farm = df_pres_diab_farm.merge(
        df_anagrafica[["idcentro", "idana", "annonascita"]],
        on=["idcentro", "idana"],
        how="left",
    )
    df_pres_diab_farm["delta_data"] = (
        (df_pres_diab_farm["data"] - df_pres_diab_farm["annonascita"])
        / np.timedelta64(1, "Y")
        / maximum_age_years
    )

    df_pres_diab_no_farm = df_pres_diab_no_farm.merge(
        df_anagrafica[["idcentro", "idana", "annonascita"]],
        on=["idcentro", "idana"],
        how="left",
    )
    df_pres_diab_no_farm["delta_data"] = (
        (df_pres_diab_no_farm["data"] - df_pres_diab_no_farm["annonascita"])
        / np.timedelta64(1, "Y")
        / maximum_age_years
    )

    df_pres_no_diab = df_pres_no_diab.merge(
        df_anagrafica[["idcentro", "idana", "annonascita"]],
        on=["idcentro", "idana"],
        how="left",
    )
    df_pres_no_diab["delta_data"] = (
        (df_pres_no_diab["data"] - df_pres_no_diab["annonascita"])
        / np.timedelta64(1, "Y")
        / maximum_age_years
    )

    # df_anagrafica.drop(columns=['annonascita'], inplace=True)
    df_diagnosi.drop(columns=["annonascita", "data"], inplace=True)
    df_esami_lab_par.drop(columns=["annonascita", "data"], inplace=True)
    df_esami_lab_par_cal.drop(columns=["annonascita", "data"], inplace=True)
    df_esami_stru.drop(columns=["annonascita", "data"], inplace=True)
    df_pres_diab_farm.drop(columns=["annonascita", "data"], inplace=True)
    df_pres_diab_no_farm.drop(columns=["annonascita", "data"], inplace=True)
    df_pres_no_diab.drop(columns=["annonascita", "data"], inplace=True)

    if WRITE_DELTA_ETA_DF:
        df_anagrafica.to_csv(f"{DELTA_ETA_PATH}/df_anagrafica_delta.csv", index=False)
        print(f"{DELTA_ETA_PATH}/df_anagrafica_delta.csv exported")

        df_diagnosi.to_csv(f"{DELTA_ETA_PATH}/df_diagnosi_delta.csv", index=False)
        print(f"{DELTA_ETA_PATH}/df_diagnosi_delta.csv exported")

        df_esami_lab_par.to_csv(
            f"{DELTA_ETA_PATH}/df_esami_lab_par_delta.csv", index=False
        )
        print(f"{DELTA_ETA_PATH}/df_esami_lab_par_delta.csv exported")

        df_esami_lab_par_cal.to_csv(
            f"{DELTA_ETA_PATH}/df_esami_lab_par_cal_delta.csv", index=False
        )
        print(f"{DELTA_ETA_PATH}/df_esami_lab_par_cal_delta.csv exported")

        df_esami_stru.to_csv(f"{DELTA_ETA_PATH}/df_esami_stru_delta.csv", index=False)
        print(f"{DELTA_ETA_PATH}/df_esami_stru_delta.csv exported")

        df_pres_diab_farm.to_csv(
            f"{DELTA_ETA_PATH}/df_pres_diab_farm_delta.csv", index=False
        )
        print(f"{DELTA_ETA_PATH}/df_pres_diab_farm_delta.csv exported")

        df_pres_diab_no_farm.to_csv(
            f"{DELTA_ETA_PATH}/df_pres_diab_no_farm_delta.csv", index=False
        )
        print(f"{DELTA_ETA_PATH}/df_pres_diab_no_farm_delta.csv exported")

        df_pres_no_diab.to_csv(
            f"{DELTA_ETA_PATH}/df_pres_no_diab_delta.csv", index=False
        )
        print(f"{DELTA_ETA_PATH}/df_pres_no_diab_delta.csv exported")

    if DELTA_VANILLA_LSTM:
        if not LOAD_DELTA_VANILLA_DF:
            vanilla_df = Vanilla_LSTM.create_dataset(
                df_anagrafica,
                df_diagnosi,
                df_esami_lab_par,
                df_esami_lab_par_cal,
                df_esami_stru,
                df_pres_diab_farm,
                df_pres_diab_no_farm,
                df_pres_no_diab,
                delta=True,
            )
            if SAVE_DELTA_VANILLA_DF:
                vanilla_df.to_csv(f"{DELTA_LSTM_DF}/vanilla_df_d.csv", index=False)
                print(f"vanilla_df.csv exported")
        else:
            print("loading vanilla data")
            vanilla_df = read_csv("{DELTA_LSTM_DF}/vanilla_df_d.csv")
            vanilla_df = vanilla_df.fillna(-100)
        if DROP_ANNI:
            vanilla_df = vanilla_df.drop(
                columns=[
                    "annonascita",
                    "delta_annoprimoaccesso",
                    "delta_decesso",
                    "delta_annodiagnosidiabete",
                ]
            )

        len_input = len(vanilla_df.columns) - 3
        vanilla_model = Vanilla_LSTM.LightingVanillaLSTM(
            input_size=len_input, hidden_size=512
        )
        grouped_vanilla = vanilla_df.groupby(["idana", "idcentro"], group_keys=True)
        inputs = []
        labels = []
        max_history_len = 0
        count = 0

        for name, group in grouped_vanilla:
            if group.values.shape[0] > max_history_len:
                max_history_len = group.values.shape[0]

        k = 2
        while k * 2 < max_history_len:
            k = k * 2
        altrocount = 0
        print("k max history len: ", k)
        for name, group in grouped_vanilla:
            vanilla_patient_hystory = group.sort_values(by=["delta_data"])
            labels.append(vanilla_patient_hystory["label"].values[0])
            vanilla_patient_hystory = vanilla_patient_hystory.drop(
                columns=["idana", "idcentro", "label"]
            )

            if vanilla_patient_hystory.values.shape[0] > k:
                altrocount += 1
                inputs.append(vanilla_patient_hystory.values[k:])
            else:
                inputs.append(vanilla_patient_hystory.values)
            count += 1
            progressBar(count, len(grouped_vanilla))
        print("altrocount: ", altrocount)

        tensor_list = [
            torch.cat(
                (
                    torch.zeros(max_history_len - len(sublist), len_input) - 200.0,
                    torch.tensor(sublist),
                )
            )
            for sublist in inputs
        ]
        padded_tensor = pad_sequence(tensor_list, batch_first=True)  # batch_first=True
        # padded_tensor = padded_tensor.to(torch.long)
        padded_tensor = padded_tensor.to(torch.float32)
        bool_tensor = torch.tensor(labels, dtype=torch.bool)
        bool_tensor = torch.tensor(labels, dtype=torch.float32)
        print("Valori unici in bool_tensor:")
        print(torch.unique(bool_tensor, return_counts=True))

        # Now you can use train_loader, val_loader, and test_loader for training, validation, and testing.
        vanilla_dataset = Vanilla_LSTM.TensorDataset(padded_tensor, bool_tensor)

        # Define the sizes for train, validation, and test sets
        train_size = int(van_train * len(vanilla_dataset))
        val_size = int(van_val * len(vanilla_dataset))
        test_size = len(vanilla_dataset) - train_size - val_size

        # Split the dataset into train, validation, and test sets
        vanilla_train_dataset, vanilla_test_dataset, vanilla_val_dataset = random_split(
            vanilla_dataset, [train_size, test_size, val_size]
        )

        # Create DataLoader instances for train, validation, and test sets
        batch_size = 16  # Adjust as needed
        train_loader = DataLoader(
            vanilla_train_dataset, batch_size=batch_size, shuffle=True
        )
        val_loader = DataLoader(
            vanilla_val_dataset, batch_size=batch_size, shuffle=True
        )
        test_loader = DataLoader(
            vanilla_test_dataset, batch_size=batch_size, shuffle=True
        )

        vanilla_train_loader = Vanilla_LSTM.DataLoader(
            vanilla_train_dataset, batch_size=batch_size, shuffle=True
        )
        vanilla_val_loader = Vanilla_LSTM.DataLoader(
            vanilla_val_dataset, batch_size=batch_size, shuffle=True
        )
        vanilla_test_loader = Vanilla_LSTM.DataLoader(
            vanilla_test_dataset, batch_size=batch_size, shuffle=True
        )
        Vanilla_LSTM.evaluate_vanilla_LSTM(
            vanilla_model,
            train=vanilla_train_dataset,
            test=vanilla_test_dataset,
            val=vanilla_val_loader,
        )
        torch.save(vanilla_model.state_dict(), "delta_vanilla_lstm")
