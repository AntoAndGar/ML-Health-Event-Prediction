import pandas as pd
import concurrent.futures as futures
import multiprocessing
import os
import re

import pickle
import numpy as np
from pytorch_lightning import (
    LightningDataModule,
    LightningModule,
    Trainer,
    seed_everything,
)
from pytorch_lightning.callbacks import ModelCheckpoint

import torch

from torch.utils.data import DataLoader
from transformers import (
    # AdamW,  # this does not work
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

import datasets
from datetime import datetime

# import evaluate
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SEED = 0
rng = np.random.default_rng(SEED)
GEN_SEED = torch.Generator().manual_seed(SEED)
seed_everything(SEED, workers=True)
MODEL_NAME = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"

os.environ["TOKENIZERS_PARALLELISM"] = "true"

READ_DATA_PATH = "clean_data"
PRESCRIZIONI = False
LOAD_DATASET = True
PARALLEL_LOAD_DATASET = True
WRITE_DATASET = False

if PRESCRIZIONI:
    file_names = [
        "anagraficapazientiattivi_c_pres",
        "diagnosi_c_pres",
        "esamilaboratorioparametri_c_pres",
        "esamilaboratorioparametricalcolati_c_pres",
        "esamistrumentali_c_pres",
        "prescrizionidiabetefarmaci_c_pres",
        "prescrizionidiabetenonfarmaci_c_pres",
        "prescrizioninondiabete_c",
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


def read_csv(filename):
    return pd.read_csv(filename, header=0)


print("Generating Futures...")
# read all the dataset concurrently and store them in a dictionary with the name of the file as key
with futures.ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
    df_list = dict()
    for name in file_names:
        df_list[str(name)] = executor.submit(read_csv, f"{READ_DATA_PATH}/{name}.csv")

print("Loading data...")
# Load dataset
if PRESCRIZIONI:
    df_anagrafica = df_list["anagraficapazientiattivi_c_pres"].result()
    df_diagnosi = df_list["diagnosi_c_pres"].result()
    df_esami_par = df_list["esamilaboratorioparametri_c_pres"].result()
    df_esami_par_cal = df_list["esamilaboratorioparametricalcolati_c_pres"].result()
    df_esami_stru = df_list["esamistrumentali_c_pres"].result()
    df_pre_diab_farm = df_list["prescrizionidiabetefarmaci_c_pres"].result()
    df_pre_diab_no_farm = df_list["prescrizionidiabetenonfarmaci_c_pres"].result()
    df_pre_no_diab = df_list["prescrizioninondiabete_c_pres"].result()
else:
    df_anagrafica = df_list["anagraficapazientiattivi_c"].result()
    df_diagnosi = df_list["diagnosi_c"].result()
    df_esami_par = df_list["esamilaboratorioparametri_c"].result()
    df_esami_par_cal = df_list["esamilaboratorioparametricalcolati_c"].result()
    df_esami_stru = df_list["esamistrumentali_c"].result()
    df_pre_diab_farm = df_list["prescrizionidiabetefarmaci_c"].result()
    df_pre_diab_no_farm = df_list["prescrizionidiabetenonfarmaci_c"].result()
    df_pre_no_diab = df_list["prescrizioninondiabete_c"].result()

list_of_df = [
    df_diagnosi,
    df_esami_par,
    df_esami_par_cal,
    df_esami_stru,
    df_pre_diab_farm,
    df_pre_diab_no_farm,
    df_pre_no_diab,
]


## Cast string to datatime
def cast_to_datetime(df, col, format="%Y-%m-%d"):
    df[col] = pd.to_datetime(df[col], format=format)
    return df[col]


for col in ["annonascita", "annoprimoaccesso", "annodecesso", "annodiagnosidiabete"]:
    df_anagrafica[col] = cast_to_datetime(df_anagrafica, col, format="%Y-%m-%d")

## Cast string to datetime
for df in list_of_df:
    df["data"] = cast_to_datetime(df, "data", format="%Y-%m-%d")

del file_names, read_csv, df_list

### Point 2.1 ####
print("Point 2.1")
# print(df_anagrafica.head())
print(df_anagrafica.label.value_counts())

df_anagrafica_label_0 = df_anagrafica[df_anagrafica.label == 0]
df_anagrafica_label_1 = df_anagrafica[df_anagrafica.label == 1]

select_all_events = pd.concat(
    objs=(
        idf.set_index(["idana", "idcentro"])
        for idf in [
            df_diagnosi[["idana", "idcentro", "data"]],
            df_esami_par[["idana", "idcentro", "data"]],
            df_esami_par_cal[["idana", "idcentro", "data"]],
            df_esami_stru[["idana", "idcentro", "data"]],
        ]
    ),
    # ignore_index=True,
    join="inner",
)

# print(select_all_events)
# select_all_events["data"] = pd.to_datetime(select_all_events["data"], format="%Y-%m-%d")

last_event = select_all_events.groupby(["idana", "idcentro"], group_keys=True)[
    "data"
].max()

# NOTE: I think the prof is wrong in the project description and here we must delete the last 6 months
# of patient with label 1
last_event_patient_label_1 = df_anagrafica_label_1.join(
    last_event, on=["idana", "idcentro"]
)


def dropLastSixMonths(df: pd.DataFrame) -> pd.DataFrame:
    df_last_event_label_1 = df.merge(
        last_event_patient_label_1,
        on=["idana", "idcentro"],
        how="left",
        suffixes=("_left", "_right"),
    )
    # TODO: here is < or <= ?
    temp = df_last_event_label_1["data_left"] < (
        df_last_event_label_1["data_right"] - pd.DateOffset(months=6)
    )
    df = (
        df_last_event_label_1[temp]
        .drop(columns=["data_right"])
        .rename(columns={"data_left": "data"})
    )
    return df


print("Before: ", len(df_diagnosi))
df_diagnosi = dropLastSixMonths(df_diagnosi)
print("After: ", len(df_diagnosi))

print("Before: ", len(df_esami_par))
df_esami_par = dropLastSixMonths(df_esami_par)
print("After: ", len(df_esami_par))

print("Before: ", len(df_esami_par_cal))
df_esami_par_cal = dropLastSixMonths(df_esami_par_cal)
print("After: ", len(df_esami_par_cal))

print("Before: ", len(df_esami_stru))
df_esami_stru = dropLastSixMonths(df_esami_stru)
print("After: ", len(df_esami_stru))

balancing = "standard"

if balancing == "lossy":
    temp_balanced_aa = df_anagrafica_label_1.sample(
        n=len(df_anagrafica_label_0), random_state=rng
    )
    balanced_aa = pd.concat([temp_balanced_aa, df_anagrafica_label_0])
    print(balanced_aa.label.value_counts())
    balanced_aa_keys = balanced_aa[["idana", "idcentro"]].drop_duplicates()
    df_diagnosi = df_diagnosi.merge(
        balanced_aa_keys, on=["idcentro", "idana"], how="inner"
    )
    df_esami_par = df_esami_par.merge(
        balanced_aa_keys, on=["idcentro", "idana"], how="inner"
    )
    df_esami_par_cal = df_esami_par_cal.merge(
        balanced_aa_keys, on=["idcentro", "idana"], how="inner"
    )
    df_esami_stru = df_esami_stru.merge(
        balanced_aa_keys, on=["idcentro", "idana"], how="inner"
    )
    df_pre_diab_farm = df_pre_diab_farm.merge(
        balanced_aa_keys, on=["idcentro", "idana"], how="inner"
    )
    df_pre_diab_no_farm = df_pre_diab_no_farm.merge(
        balanced_aa_keys, on=["idcentro", "idana"], how="inner"
    )
    df_pre_no_diab = df_pre_no_diab.merge(
        balanced_aa_keys, on=["idcentro", "idana"], how="inner"
    )

elif balancing == "standard":
    # TODO: check if this is correct, because to me it seems silly that we have
    # to modify values with labels 1 to make them 0, at the end the model
    # will be confused by this
    duplication_factor = 2
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
        # shuffle and delete events at random
        new_dup_record = new_dup_record[new_dup_record["duplicated"]].sample(
            frac=fraction, random_state=rng
        )

        # shuffle data
        noise = pd.to_timedelta(
            rng.normal(0, 5, len(new_dup_record)).astype("int"), unit="d"
        )
        new_dup_record["data"] = new_dup_record["data"] + noise
        # TODO: while here we are adding noise to the date, we should ensure that the new date is not
        # in the 6 months after last event, to ensure we are not creating other False.
        # Maybe do also here a dropLastSixMonths? nah this is improbable to happen because the
        # noise is small and also the probability of having a date in the 6 months after the last event
        # new_dup_record = dropLastSixMonths(new_dup_record)

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

    # Yes you can fix this if yoiu prefer but to me the code is unreadable to do so
    # FIXME: A value is trying to be set on a copy of a slice from a DataFrame.
    #       Try using .loc[row_indexer,col_indexer] = value instead
    #
    #       See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    #       new_dup_record["label"] = False
    new_dup_record["label"] = False

    new_dup_record = new_dup_record.drop(["duplicate_identifier", "duplicated"], axis=1)
    df_anagrafica = pd.concat([df_anagrafica, new_dup_record], ignore_index=True)
    print("After balance: ", len(df_anagrafica))
    print(df_anagrafica.label.value_counts())
    print(df_anagrafica.head())
    print(df_anagrafica.tail(10))

    print("Before balance: ", len(df_diagnosi))
    df_diagnosi = balance(df_diagnosi, 0.50)
    print("After balance: ", len(df_diagnosi))

    print("Before balance: ", len(df_esami_par))
    df_esami_par = balance(df_esami_par, 0.50)
    print("After balance: ", len(df_esami_par))

    print("Before balance: ", len(df_esami_par_cal))
    df_esami_par_cal = balance(df_esami_par_cal, 0.50)
    print("After balance: ", len(df_esami_par_cal))

    print("Before balance: ", len(df_esami_stru))
    df_esami_stru = balance(df_esami_stru, 0.50)
    print("After balance: ", len(df_esami_stru))

tuple_dataset = []

if LOAD_DATASET:
    amd = pd.read_csv("amd_codes_for_bert.csv").rename({"codice": "codiceamd"}, axis=1)
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

    df_esami_par = df_esami_par.merge(amd, on="codiceamd", how="left")
    df_esami_par = (
        df_esami_par[["idcentro", "idana", "data", "codiceamd", "meaning", "valore"]]
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

    df_esami_par_cal = df_esami_par_cal.merge(amd, on="codiceamd", how="left")
    df_esami_par_cal = (
        df_esami_par_cal[
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
    df_pre_diab_farm = (
        df_pre_diab_farm.merge(
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

    df_pre_diab_no_farm = df_pre_diab_no_farm.merge(amd, on="codiceamd", how="left")
    df_pre_diab_no_farm = (
        df_pre_diab_no_farm[
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

    df_pre_no_diab = df_pre_no_diab.merge(amd, on="codiceamd", how="left")
    df_pre_no_diab = (
        df_pre_no_diab[["idcentro", "idana", "data", "codiceamd", "meaning", "valore"]]
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

    list_of_df = {
        "diagnosis": df_diagnosi,
        "exam parameter": df_esami_par,
        "exam parameter calculated": df_esami_par_cal,
        "exam strumental": df_esami_stru,
        "prescription diabete drugs": df_pre_diab_farm,
        "prescription diabete not drugs": df_pre_diab_no_farm,
        "prescription not diabete": df_pre_no_diab,
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
        # Iterate over each DataFrame in the dictionary list_of_df
        for name, df in list_of_df.items():
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
        with open("dataset_clean.pkl", "wb") as f:
            pickle.dump(tuple_dataset, f)
        print("stored dataset")
else:
    with open("dataset_clean.pkl", "rb") as f:
        tuple_dataset = pickle.load(f)

    print("loaded dataset")
    print("dataset: ", len(tuple_dataset))
    # print(tuple_dataset[:1])


#####################
# LSTM
#####################


def evaluate_vanilla_LSTM():
    print("Using {torch.cuda.get_device_name(DEVICE)}")

    # why lose time using keras or tensorflow ?
    # when we can use pytorch (pytorch lightning I mean, but also pytorch is ok)
    return

    from keras.models import Sequential
    from keras.layers import LSTM, Dense
    from keras.optimizers import Adam
    from keras.losses import BinaryCrossentropy
    from keras.metrics import BinaryAccuracy

    # At first, we merge with the patient data
    features_lstm = pd.merge(df_diagnosi, df_anagrafica, on=["idcenter", "idpatient"])

    # We create a single ID column that combines the other three:
    features_lstm["id"] = features_lstm.apply(
        lambda x: f"{x['idcenter']}_{x['idpatient']}", axis=1
    )

    # We reorder the columns
    features_lstm = features_lstm[
        ["id"] + [x for x in features_lstm.columns if x != "id"]
    ]

    # We drop the other ID_columns
    features_lstm.drop(columns=["idcenter", "idpatient"], inplace=True)

    # We categorize the columns that contain text
    categorical_columns = ["amdcode", "value", "sex"]
    for col in categorical_columns:
        features_lstm[col] = features_lstm[col].astype("category")
        features_lstm[col] = features_lstm[col].cat.codes

    # We convert every columns into float type
    numerical_columns = [
        col for col in features_lstm.columns if col not in ["id", "date"]
    ]
    for col in numerical_columns:
        features_lstm[col] = features_lstm[col].astype("float")

    features_lstm.head(10)

    X_columns = [col for col in df.columns if col not in ["id", "label", "date"]]
    y_columns = ["label"]

    Vanilla_LSTM = Sequential()
    Vanilla_LSTM
    Vanilla_LSTM.add(
        LSTM(
            100,
            activation="tanh",
            return_sequences=True,
            input_shape=(1, len(X_columns)),
        )
    )
    Vanilla_LSTM.add(LSTM(49, activation="tanh"))
    Vanilla_LSTM.add(Dense(1, activation="sigmoid"))
    Vanilla_LSTM.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss=BinaryCrossentropy(),
        metrics=[BinaryAccuracy()],
    )

    grouped_events = features_lstm.groupby(["id"])

    for it, (ids, features) in enumerate(grouped_events):
        batch = features[features["id"] == ids].sort_values(["date"])
        X = batch[X_columns]
        X = np.resize(X, (X.shape[0], 1, X.shape[1]))
        y = batch[y_columns]
        if it % 200 == 0:
            print(f"Patient {it}/{len(df_anagrafica)}")
        Vanilla_LSTM.fit(
            X, y, batch_size=len(X), epochs=10, verbose=1 if it % 200 == 0 else 0
        )

    # We take a single batch to evaluate the model
    rand_index = random.randint(0, len(df_anagrafica))
    rand_id = tuple(df_anagrafica.iloc[rand_index, :3])
    rand_id = f"{rand_id[0]}_{rand_id[1]}_{rand_id[2]}"
    rand_batch = features_lstm[features_lstm["id"] == rand_id]
    print(rand_batch)

    X = rand_batch[X_columns]
    X = np.resize(X, (X.shape[0], 1, X.shape[1]))
    Vanilla_LSTM.evaluate(x=X, y=rand_batch[y_columns])


def evaluate_T_LSTM():
    return


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
        precision=32,
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
