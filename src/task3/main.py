import concurrent.futures
import multiprocessing
import model
import numpy as np
import pandas as pd
import re
import torch
import optuna

from torch.utils.data import DataLoader, TensorDataset, random_split

SEED = 0
rng = np.random.default_rng(SEED)
GEN_SEED = torch.Generator().manual_seed(SEED)
torch.manual_seed(SEED)

PRESCRIZIONI = True
READ_DATA_PATH = "balanced_data"

if PRESCRIZIONI:
    file_names = [
        "anagraficapazientiattivi_b_pres",
        "diagnosi_b_pres",
        "esamilaboratorioparametri_b_pres",
        "esamilaboratorioparametricalcolati_b_pres",
        "esamistrumentali_b_pres",
        "prescrizionidiabetefarmaci_b_pres",
        "prescrizionidiabetenonfarmaci_b_pres",
        "prescrizioninondiabete_b_pres",
    ]
else:
    file_names = [
        "anagraficapazientiattivi_b",
        "diagnosi_b",
        "esamilaboratorioparametri_b",
        "esamilaboratorioparametricalcolati_b",
        "esamistrumentali_b",
        "prescrizionidiabetefarmaci_b",
        "prescrizionidiabetenonfarmaci_b",
        "prescrizioninondiabete_b",
    ]


def read_csv(filename):
    return pd.read_csv(filename, header=0)


print("Generating Futures...")
# Read all the dataset concurrently and store them in a dictionary with the name of the file as key
with concurrent.futures.ThreadPoolExecutor(
    max_workers=multiprocessing.cpu_count()
) as executor:
    df_list = dict()
    for name in file_names:
        df_list[str(name)] = executor.submit(read_csv, f"{READ_DATA_PATH}/{name}.csv")

if PRESCRIZIONI:
    df_anagrafica = df_list["anagraficapazientiattivi_b_pres"].result()
    df_diagnosi = df_list["diagnosi_b_pres"].result()
    df_esami_lab_par = df_list["esamilaboratorioparametri_b_pres"].result()
    df_esami_lab_par_cal = df_list["esamilaboratorioparametricalcolati_b_pres"].result()
    df_esami_stru = df_list["esamistrumentali_b_pres"].result()
    df_pres_diab_farm = df_list["prescrizionidiabetefarmaci_b_pres"].result()
    df_pres_diab_no_farm = df_list["prescrizionidiabetenonfarmaci_b_pres"].result()
    df_pres_no_diab = df_list["prescrizioninondiabete_b_pres"].result()
else:
    df_anagrafica = df_list["anagraficapazientiattivi_b"].result()
    df_diagnosi = df_list["diagnosi_b"].result()
    df_esami_lab_par = df_list["esamilaboratorioparametri_b"].result()
    df_esami_lab_par_cal = df_list["esamilaboratorioparametricalcolati_b"].result()
    df_esami_stru = df_list["esamistrumentali_b"].result()
    df_pres_diab_farm = df_list["prescrizionidiabetefarmaci_b"].result()
    df_pres_diab_no_farm = df_list["prescrizionidiabetenonfarmaci_b"].result()
    df_pres_no_diab = df_list["prescrizioninondiabete_b"].result()

#######################################
############### STEP 1 ################
#######################################

# In this step we have considered records from table diagnosi as macro events
# While the other ones have been considered as micro events (esami and prescrizioni)
# So now we are going to delete the dates from the micro events
df_esami_lab_par = df_esami_lab_par.drop(columns="data")
df_esami_lab_par_cal = df_esami_lab_par_cal.drop(columns="data")
df_esami_stru = df_esami_stru.drop(columns="data")
df_pres_diab_farm = df_pres_diab_farm.drop(columns="data").rename(
    {"codiceatc": "codiceamd"}, axis=1
)
df_pres_diab_no_farm = df_pres_diab_no_farm.drop(columns="data")
df_pres_no_diab = df_pres_no_diab.drop(columns="data")

# Now we are going to prepare data for the model
all_events_concat = pd.concat(
    objs=(
        df_diagnosi,
        df_esami_lab_par,
        df_esami_lab_par_cal,
        df_esami_stru,
        df_pres_diab_farm,
        df_pres_diab_no_farm,
    )
)
# Split between train and test dataset
train_size = 0.8
test_size = 0.2
# Creating a dataframe with 50%
# values of original dataframe

##########################
### creating test dataset
##########################

test_anagrafica = df_anagrafica.sample(frac=test_size, random_state=2)

# Creating dataframe with
# rest of the 50% values
train_anagrafica = df_anagrafica.drop(test_anagrafica.index)

final_df = test_anagrafica.merge(
    all_events_concat, on=["idana", "idcentro"], how="inner"
)

# First we delete "idana" and "idcentro" as they don't give informations to the model
# final_df = final_df.drop(columns=["idana", "idcentro"])

# Here we convert feature "sesso" into numeric feature
final_df["sesso"] = final_df["sesso"].replace(["M", "F"], [0.0, 1.0])

# Now we want to convert every date into a numeric progressive value
# We chose them as the number of months that have passed from the birth
final_df["annonascita"] = pd.to_datetime(final_df["annonascita"], format="%Y-%m-%d")
final_df["annodiagnosidiabete"] = pd.to_datetime(
    final_df["annodiagnosidiabete"], format="%Y-%m-%d"
)
final_df["annoprimoaccesso"] = pd.to_datetime(
    final_df["annoprimoaccesso"], format="%Y-%m-%d"
)
final_df["annodecesso"] = pd.to_datetime(final_df["annodecesso"], format="%Y-%m-%d")
final_df["data"] = pd.to_datetime(final_df["data"], format="%Y-%m-%d")

final_df["annodiagnosidiabete"] = (
    final_df["annodiagnosidiabete"] - final_df["annonascita"]
) / pd.Timedelta(days=31)
final_df["annoprimoaccesso"] = (
    final_df["annoprimoaccesso"] - final_df["annonascita"]
) / pd.Timedelta(days=31)
final_df["annodecesso"] = (
    final_df["annodecesso"] - final_df["annonascita"]
) / pd.Timedelta(days=31)
final_df["data"] = (final_df["data"] - final_df["annonascita"]) / pd.Timedelta(days=31)

# We delete the date of the birth since would be zero for all records
# We also delete columns scolarita, statocivile and professione since they have a percentage of NaN values above 50%
# We delete also the column "descrizionefarmaco" since it is a description of the drug and it is very resource expensive to utilize it
final_df = final_df.drop(
    columns=[
        "annonascita",
        "scolarita",
        "statocivile",
        "professione",
        "descrizionefarmaco",
    ]
)

# Now we substitute all categorical feature into a numeric one
final_df["codiceamd"] = pd.Categorical(final_df["codiceamd"]).codes.astype(float)
final_df["codiceamd"] = final_df["codiceamd"].replace(-1.0, np.nan)

final_df["valore"] = pd.Categorical(final_df["valore"]).codes.astype(float)
final_df["valore"] = final_df["valore"].replace(-1.0, np.nan)

final_df["codicestitch"] = pd.Categorical(final_df["codicestitch"]).codes.astype(float)
final_df["codicestitch"] = final_df["codicestitch"].replace(-1.0, np.nan)

# final_df["descrizionefarmaco"] = pd.Categorical(final_df["descrizionefarmaco"]).codes.astype(float)
# final_df["descrizionefarmaco"] = final_df["descrizionefarmaco"].replace(-1.0, np.nan)


# We convert boolean label into numeric value
final_df["label"] = final_df["label"].replace([False, True], [0.0, 1.0])

# And we replace all the remaining NaN values with the value -100 in order to be ignored by the model
final_df = final_df.fillna(-100)

final_df = final_df.sort_values(by=["idana", "idcentro", "data"])
final_df = final_df.drop(columns=["idana", "idcentro"])
# Then we construct the TensorDataset object
data = final_df.drop("label", axis=1).values
labels = final_df["label"].values

test_dataset = TensorDataset(torch.FloatTensor(data), torch.LongTensor(labels))
print("Created test dataset")

##########################
### creating train dataset
##########################

final_df = train_anagrafica.merge(
    all_events_concat, on=["idana", "idcentro"], how="inner"
)

# First we delete "idana" and "idcentro" as they don't give informations to the model
# final_df = final_df.drop(columns=["idana", "idcentro"])

# Here we convert feature "sesso" into numeric feature
final_df["sesso"] = final_df["sesso"].replace(["M", "F"], [0.0, 1.0])

# Now we want to convert every date into a numeric progressive value
# We chose them as the number of months that have passed from the birth
final_df["annonascita"] = pd.to_datetime(final_df["annonascita"], format="%Y-%m-%d")
final_df["annodiagnosidiabete"] = pd.to_datetime(
    final_df["annodiagnosidiabete"], format="%Y-%m-%d"
)
final_df["annoprimoaccesso"] = pd.to_datetime(
    final_df["annoprimoaccesso"], format="%Y-%m-%d"
)
final_df["annodecesso"] = pd.to_datetime(final_df["annodecesso"], format="%Y-%m-%d")
final_df["data"] = pd.to_datetime(final_df["data"], format="%Y-%m-%d")

final_df["annodiagnosidiabete"] = (
    final_df["annodiagnosidiabete"] - final_df["annonascita"]
) / pd.Timedelta(days=31)
final_df["annoprimoaccesso"] = (
    final_df["annoprimoaccesso"] - final_df["annonascita"]
) / pd.Timedelta(days=31)
final_df["annodecesso"] = (
    final_df["annodecesso"] - final_df["annonascita"]
) / pd.Timedelta(days=31)
final_df["data"] = (final_df["data"] - final_df["annonascita"]) / pd.Timedelta(days=31)

# We delete the date of the birth since would be zero for all records
# We also delete columns scolarita, statocivile and professione since they have a percentage of NaN values above 50%
# We delete also the column "descrizionefarmaco" since it is a description of the drug and it is very resource expensive to utilize it
final_df = final_df.drop(
    columns=[
        "annonascita",
        "scolarita",
        "statocivile",
        "professione",
        "descrizionefarmaco",
    ]
)

# Now we substitute all categorical feature into a numeric one
final_df["codiceamd"] = pd.Categorical(final_df["codiceamd"]).codes.astype(float)
final_df["codiceamd"] = final_df["codiceamd"].replace(-1.0, np.nan)

final_df["valore"] = pd.Categorical(final_df["valore"]).codes.astype(float)
final_df["valore"] = final_df["valore"].replace(-1.0, np.nan)

final_df["codicestitch"] = pd.Categorical(final_df["codicestitch"]).codes.astype(float)
final_df["codicestitch"] = final_df["codicestitch"].replace(-1.0, np.nan)

# final_df["descrizionefarmaco"] = pd.Categorical(final_df["descrizionefarmaco"]).codes.astype(float)
# final_df["descrizionefarmaco"] = final_df["descrizionefarmaco"].replace(-1.0, np.nan)


# We convert boolean label into numeric value
final_df["label"] = final_df["label"].replace([False, True], [0.0, 1.0])

# And we replace all the remaining NaN values with the value -100 in order to be ignored by the model
final_df = final_df.fillna(-100)

final_df = final_df.sort_values(by=["idana", "idcentro", "data"])
final_df = final_df.drop(columns=["idana", "idcentro"])

# Then we construct the TensorDataset object
data = final_df.drop("label", axis=1).values
labels = final_df["label"].values

train_dataset = TensorDataset(torch.FloatTensor(data), torch.LongTensor(labels))
print("Created train dataset")

# Training step
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Utilizing device: ", device)

INPUT_SIZE = 10
NUM_CLASSES = 2


def train(model, num_epochs, data_loader, device, criterion, optimizer, batch_size):
    model.train()
    for epoch in range(num_epochs):
        for i, (inputs, labels) in enumerate(data_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            loss = criterion(outputs, labels)
            loss.backward()

            optimizer.step()

            if (i + 1) % 500 == 0:
                print(
                    "Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}".format(
                        epoch + 1,
                        num_epochs,
                        i + 1,
                        len(data_loader),
                        loss.item(),
                    )
                )


def evaluate(my_model, test_dataloader):
    correct = 0
    total = 0
    my_model.eval()
    with torch.no_grad():
        for inputs, labels in test_dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = my_model(inputs)

            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total


def objective(trial):
    batch_size = trial.suggest_categorical("batch_size", [128, 256])
    train_data_loader = DataLoader(
        train_dataset, batch_size, num_workers=8, shuffle=False
    )
    test_dataloader = DataLoader(test_dataset, batch_size, num_workers=4, shuffle=False)
    gru_num_layers = trial.suggest_int("gru_num_layers", 1, 2)
    gru_hidden_size = trial.suggest_categorical("gru_hidden_size", [16, 32, 64])

    net = model.Model(
        input_size=INPUT_SIZE,
        hidden_size=gru_hidden_size,
        num_layers=gru_num_layers,
        num_classes=NUM_CLASSES,
    ).to(device)

    n_epochs = trial.suggest_int("n_epochs", 5, 15, step=5)
    learning_rate = trial.suggest_float("learning_rate", 5e-5, 7e-3, log=True)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(net.parameters(), learning_rate)

    train(
        net,
        n_epochs,
        train_data_loader,
        device,
        criterion,
        optimizer,
        batch_size,
    )

    accuracy = evaluate(net, test_dataloader)

    return accuracy


study = optuna.create_study(study_name="Bayesian optimization", direction="maximize")
study.optimize(objective, n_trials=20)
print("Best accuracy: ", study.best_value)
print("Best hyperparameters", study.best_params)
