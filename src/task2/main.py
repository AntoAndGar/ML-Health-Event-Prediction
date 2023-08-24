import concurrent.futures as futures
import datasets
import multiprocessing
import numpy as np
import pandas as pd
import pickle
import os
import re
import torch

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

from torch.utils.data import DataLoader

from torchmetrics.classification import BinaryAccuracy, BinaryF1Score

from transformers import (
    # AdamW,  # this does not work
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SEED = 0
rng = np.random.default_rng(SEED)
GEN_SEED = torch.Generator().manual_seed(SEED)
seed_everything(SEED, workers=True)
MODEL_NAME = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"

os.environ["TOKENIZERS_PARALLELISM"] = "true"

READ_DATA_PATH = "clean_data"
PRESCRIZIONI = True
CREATE_DATASET = False
PARALLEL_LOAD_DATASET = True
WRITE_DATASET = False
DATASET_NAME = "dataset_def.pkl"

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
    #print("Error: more than 2 different labels")
    raise("Error: more than 2 different labels")


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
    #temp = df_label_0_last_event["data_left"] >= (
    #    df_label_0_last_event["data_right"] - pd.Timedelta(days=186)
    temp = df_last_event_label_1["data_left"] < (
        df_last_event_label_1["data_right"] - np.timedelta64(6, "M")
    )
    df = (
        df_last_event_label_1[temp]
        .drop(columns=["data_right",
                        "sesso",
                        "annodiagnosidiabete",
                        "label",
                        "scolarita",
                        "statocivile",
                        "professione",
                        "annonascita",
                        "annoprimoaccesso",
                        "annodecesso"])
        .rename(columns={"data_left": "data"})
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
    # TODO: check if this is correct, because to me it seems silly that we have
    # to modify values with labels 1 to make them 0, at the end the model
    # will be confused by this
    duplication_factor = int(len(df_anagrafica_label_0) / len(df_anagrafica_label_1))
    # here the duplication factor is -1 because 1 time is already present in the original df
    # at which we append the duplicated df
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

    # Yes you can fix this if yoiu prefer but to me the code is unreadable to do so
    # FIXME: A value is trying to be set on a copy of a slice from a DataFrame.
    #       Try using .loc[row_indexer,col_indexer] = value instead
    #
    #       See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    #       new_dup_record["idana"] = -(
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

#####################
# LSTM
#####################
    
val = 0.1
test = 0.3
train = 1 - val - test

#TODO: Converti datatime in float

#TODO: Split in train, test (and validation?)


print("\n\n\t\tSTART WORKING WITH LSTM \n\n")

vanilla_df = Vanilla_LSTM.create_dataset(df_anagrafica, df_diagnosi, df_esami_par, df_esami_par_cal, df_esami_stru, df_pre_diab_farm, df_pre_diab_no_farm, df_pre_no_diab) 
print("dataset, created")
vanilla_model = Vanilla_LSTM.LightingVanillaLSTM(input_size=len(vanilla_df.columns)-3, hidden_size=1)
print("model created")
#grouped_vanilla = vanilla_df.groupby(["idana", "idcentro"], group_keys=True).apply(lambda x: x)
grouped_vanilla = vanilla_df.groupby(["idana", "idcentro"], group_keys=True)
print("grouped")
inputs = []
labels = []
dict = []
dict2 = {}
dict3 = {}
firstTime = True
for name, group in grouped_vanilla:
    vanilla_patient_hystory = group.sort_values(by=["data"])
    labels.append(torch.tensor(group["label"].values))
    if vanilla_patient_hystory.values.shape[0] == 1:
        #print(vanilla_patient_hystory)
        dict3[vanilla_patient_hystory["extra"].values[0]] = 1
        if vanilla_patient_hystory["idana"].values[0] > 0:
            daje = True
            print(vanilla_patient_hystory.values)
            if firstTime:
                print("Fist time")
                #print(vanilla_patient_hystory.values)
                print(vanilla_patient_hystory.values.shape)
                print(vanilla_patient_hystory.values.shape[0])
                firstTime = False
            print("DJ")
    vanilla_patient_hystory = vanilla_patient_hystory.drop(columns=["data","annonascita", "annoprimoaccesso", "annodecesso", "annodiagnosidiabete"])
    vanilla_patient_hystory = vanilla_patient_hystory.drop(columns=["idana", "idcentro", "label"])
    inputs.append(vanilla_patient_hystory.values)
    dict2[vanilla_patient_hystory.values.shape[0]] = dict2[vanilla_patient_hystory.values.shape[0]] + 1 if vanilla_patient_hystory.values.shape[0] in dict2 else 1


    dict.append(vanilla_patient_hystory.values.shape[0])
    dict.append(vanilla_patient_hystory.values.shape[1])
print("DDDJJJ")
print(dict3)
print("Shapes")
print("Shapes")
print(dict[0])
print(dict[-1])

try:
    print("I valori")
    vero_dict2 = list(dict.keys(dict2))
    print(vero_dict2)
except:
    print("I valori non stampabili")
print(dict2)
#print(dict2)  
input("inputs: ", inputs)
inputs = torch.tensor(inputs)
labels = torch.tensor(labels)
vanilla_dataset = Vanilla_LSTM.TensorDataset(inputs, labels)
dataloader = Vanilla_LSTM.DataLoader(vanilla_dataset, batch_size=16, shuffle=True)

Vanilla_LSTM.evaluate_vanilla_LSTM(vanilla_model, dataloader)

exit()

#####################
# PubMedBERT
#####################
tuple_dataset = []

if CREATE_DATASET:
    amd = pd.read_csv("amd_codes_for_bert.csv").rename({"codice": "codiceamd"}, axis=1)
    atc = pd.read_csv("atc_info_nodup.csv")
    # Converting Dataset for Deep Learning purposes

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

    df_esami_lab_par_cal = df_esami_lab_par_cal.merge(amd, on="codiceamd", how="left")
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
        df_esami_stru[["idcentro", "idana", "data", "codiceamd", "meaning", "valore"]]
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

    df_pres_diab_no_farm = df_pres_diab_no_farm.merge(amd, on="codiceamd", how="left")
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
        df_pres_no_diab[["idcentro", "idana", "data", "codiceamd", "meaning", "valore"]]
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
    # dm.setup("fit")
    # print(next(iter(dm.train_dataloader())))

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


evaluate_vanilla_LSTM()
evaluate_PubMedBERT()

############################
### Advanced Unbalancing ###
############################

# Source https://github.com/bardhprenkaj/ML_labs/blob/main/src/lab1/Data_Feature_preprocessing.ipynb
"""
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from matplotlib import pyplot
from numpy import where
"""
#####################
# SMOTE
#####################
