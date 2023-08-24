import concurrent.futures as futures
import multiprocessing
import numpy as np
import pandas as pd

WRITE_DATA_PATH = "clean_data"
PRESCRIZIONI = True
WRITE_CSV = False

"""
AMD047: Myocardial infarction
AMD048: Coronary angioplasty
AMD049: Coronary bypass
AMD071: Ictus
AMD081: Lower limb angioplasty
AMD082: Peripheral By-pass Lower Limbs
AMD208: Revascularization of intracranial and neck vessels
AMD303: Ischemic stroke
"""
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

# Import the data
print("############## STARTING COMPUTATION ##############")

file_names = [
    "anagraficapazientiattivi",
    "diagnosi",
    "esamilaboratorioparametri",
    "esamilaboratorioparametricalcolati",
    "esamistrumentali",
    "prescrizionidiabetefarmaci",
    "prescrizionidiabetenonfarmaci",
    "prescrizioninondiabete",
]


def read_csv(filename):
    return pd.read_csv(filename, header=0, index_col=0)


# Read all datasets concurrently and store them in a dictionary with the name of the file as key
with futures.ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
    df_list = dict()
    for name in file_names:
        df_list[str(name)] = executor.submit(read_csv, f"data/{name}.csv")

df_anagrafica_attivi = df_list["anagraficapazientiattivi"].result()
print(
    f"Number of records in anagrafica pazienti attivi: {len(df_anagrafica_attivi)}"
)  # 250000

df_diagnosi = df_list["diagnosi"].result()
print(f"Number of records in diagnosi: {len(df_diagnosi)}")  # 4427337

df_esami_lab_par = df_list["esamilaboratorioparametri"].result()
print(
    f"Number of records in esami laboratorio parametri: {len(df_esami_lab_par)}"
)  # 28628530

df_esami_lab_par_cal = df_list["esamilaboratorioparametricalcolati"].result()
print(
    f"Number of records in esami laboratorio parametri calcolati: {len(df_esami_lab_par_cal)}"
)  # 10621827

df_esami_strumentali = df_list["esamistrumentali"].result()
print(f"Number of records in esami strumentali: {len(df_esami_strumentali)}")  # 1015740

df_prescrizioni_diabete_farmaci = df_list["prescrizionidiabetefarmaci"].result()
print(
    f"Number of records in prescrizioni diabete farmaci: {len(df_prescrizioni_diabete_farmaci)}"
)  # 7012648

df_prescrizioni_diabete_non_farmaci = df_list["prescrizionidiabetenonfarmaci"].result()
print(
    f"Number of records in prescrizioni diabete non farmaci: {len(df_prescrizioni_diabete_non_farmaci)}"
)  # 548467

df_prescrizioni_non_diabete = df_list["prescrizioninondiabete"].result()
print(
    f"Number of records in prescrizioni non diabete: {len(df_prescrizioni_non_diabete)}"
)  # 5083861

print("############## FUTURES CREATED ##############")

del df_list

#######################################
############### STEP 1 ################
#######################################

print("########## STEP 1 ##########")

print(
    "Number of patients in anagrafica who are present in diagnosi:",
    len(
        df_anagrafica_attivi[["idana", "idcentro"]]
        .drop_duplicates()
        .merge(
            df_diagnosi[["idana", "idcentro"]].drop_duplicates(),
            how="inner",
            on=["idana", "idcentro"],
        )
    ),
)  # 226303

# Diagnoses related to cardiovascular problems, this is the only table that contains the cardiovascular event codes
df_diagnosi_problemi_cuore = df_diagnosi[
    df_diagnosi["codiceamd"].isin(AMD_OF_CARDIOVASCULAR_EVENT)
]

print(
    "Number of records present in diagnosi related to cardiovascular problem (with amd code in the wanted list):",
    len(df_diagnosi_problemi_cuore),
)  # 233204

# Retrieving unique keys of patients of interest in order to filter other tables
df_diagnosi_problemi_cuore_keys = df_diagnosi_problemi_cuore[
    ["idana", "idcentro"]
].drop_duplicates()

print(
    "Number of patients present in diagnosi with cardiovascular problem (with amd code in the wanted list):",
    len(df_diagnosi_problemi_cuore_keys),
)  # 50000

# Filtering all the tables to have only patients with cardiovascular events
df_anagrafica_attivi = df_anagrafica_attivi.merge(
    df_diagnosi_problemi_cuore_keys, on=["idcentro", "idana"], how="inner"
)

print(
    f"Number of records in anagrafica pazienti attivi after step 1: {len(df_anagrafica_attivi)}"
)  # 50000

df_diagnosi = df_diagnosi.merge(
    df_diagnosi_problemi_cuore_keys, on=["idcentro", "idana"], how="inner"
)

print(f"Number of records in diagnosi after step 1: {len(df_diagnosi)}")  # 1938342

df_esami_lab_par = df_esami_lab_par.merge(
    df_diagnosi_problemi_cuore_keys, on=["idcentro", "idana"], how="inner"
)

print(
    f"Number of records in esami laboratorio parametri after step 1: {len(df_esami_lab_par)}"
)  # 7371159

df_esami_lab_par_cal = df_esami_lab_par_cal.merge(
    df_diagnosi_problemi_cuore_keys, on=["idcentro", "idana"], how="inner"
)

print(
    f"Number of records in esami laboratorio parametri calcolati after step 1: {len(df_esami_lab_par_cal)}"
)  # 2769151

df_esami_strumentali = df_esami_strumentali.merge(
    df_diagnosi_problemi_cuore_keys, on=["idcentro", "idana"], how="inner"
)

print(
    f"Number of records in esami strumentali after step 1: {len(df_esami_strumentali)}"
)  # 290793

df_prescrizioni_diabete_farmaci = df_prescrizioni_diabete_farmaci.merge(
    df_diagnosi_problemi_cuore_keys, on=["idcentro", "idana"], how="inner"
)

print(
    f"Number of records in prescrizioni diabete farmaci after step 1: {len(df_prescrizioni_diabete_farmaci)}"
)  # 1989613

df_prescrizioni_diabete_non_farmaci = df_prescrizioni_diabete_non_farmaci.merge(
    df_diagnosi_problemi_cuore_keys, on=["idcentro", "idana"], how="inner"
)

print(
    f"Number of records in prescrizioni diabete non farmaci after step 1: {len(df_prescrizioni_diabete_non_farmaci)}"
)  # 150340

df_prescrizioni_non_diabete = df_prescrizioni_non_diabete.merge(
    df_diagnosi_problemi_cuore_keys, on=["idcentro", "idana"], how="inner"
)

print(
    f"Number of records in prescrizioni non diabete after step 1: {len(df_prescrizioni_non_diabete)}"
)  # 1995073

del df_diagnosi_problemi_cuore, df_diagnosi_problemi_cuore_keys

#######################################
############### STEP 2 ################
#######################################

print("########## STEP 2 ##########")

# First of all we will handle the NaN values on the futures regarding dates
print(
    "Number of records in anagrafica pazienti attivi where anno nascita is NaN:",
    sum(df_anagrafica_attivi["annonascita"].isna()),
)  # 0

print(
    "Number of records in anagrafica pazienti attivi where anno primo accesso is NaN:",
    sum(df_anagrafica_attivi["annoprimoaccesso"].isna()),
)  # 6974

print(
    "Number of records in anagrafica pazienti attivi where anno diagnosi diabete is NaN:",
    sum(df_anagrafica_attivi["annodiagnosidiabete"].isna()),
)  # 526

print(
    "Number of records in diagnosi where data in NaN:", sum(df_diagnosi["data"].isna())
)  # 479

print(
    "Number of records in esami laboratorio parametri where data is NaN:",
    sum(df_esami_lab_par["data"].isna()),
)  # 0

print(
    "Number of records in esami laboratorio parametri calcolati where data is NaN:",
    sum(df_esami_lab_par_cal["data"].isna()),
)  # 0

print(
    "Number of records in esami strumentali where data is NaN:",
    sum(df_esami_strumentali["data"].isna()),
)  # 0

print(
    "Number of records in prescrizioni diabete farmaci where data is NaN:",
    sum(df_prescrizioni_diabete_farmaci["data"].isna()),
)  # 0

print(
    "Number of records in prescrizioni diabete non farmaci where data is Nan:",
    sum(df_prescrizioni_diabete_non_farmaci["data"].isna()),
)  # 0

print(
    "Number of records in prescrizioni non diabete where data is NaN:",
    sum(df_prescrizioni_non_diabete["data"].isna()),
)  # 0

# Dropping records which have NaN values in "data" from table "diagnosi"
df_diagnosi = df_diagnosi[df_diagnosi["data"].notna()]

df_list = [
    df_diagnosi,
    df_esami_lab_par,
    df_esami_lab_par_cal,
    df_esami_strumentali,
    df_prescrizioni_diabete_farmaci,
    df_prescrizioni_diabete_non_farmaci,
    df_prescrizioni_non_diabete,
]


# Casting "data" feature to datetime in all tables except "anagrafica"
def cast_to_datetime(df, col, format="%Y-%m-%d"):
    df[col] = pd.to_datetime(df[col], format=format)
    return df[col]


for df in df_list:
    df["data"] = cast_to_datetime(df, "data", format="%Y-%m-%d")

# Casting also "anagrafica" dates in order to make comparisons
df_anagrafica_attivi["annodiagnosidiabete"] = pd.to_datetime(
    df_anagrafica_attivi["annodiagnosidiabete"], format="%Y"
)

df_anagrafica_attivi["annonascita"] = pd.to_datetime(
    df_anagrafica_attivi["annonascita"], format="%Y"
)

df_anagrafica_attivi["annoprimoaccesso"] = pd.to_datetime(
    df_anagrafica_attivi["annoprimoaccesso"], format="%Y"
)

df_anagrafica_attivi["annodecesso"] = pd.to_datetime(
    df_anagrafica_attivi["annodecesso"], format="%Y"
)

# We are going to fill the NaN records of "anno primo accesso" with the date of the least recent event corresponding to that id
df_anagrafica_attivi_primo_accesso_nan = df_anagrafica_attivi[
    df_anagrafica_attivi["annoprimoaccesso"].isna()
]

esami_and_prescrizioni_concat = pd.concat(
    objs=(
        idf.set_index(["idana", "idcentro"])
        for idf in [
            df_diagnosi[["idana", "idcentro", "data"]],
            df_esami_lab_par[["idana", "idcentro", "data"]],
            df_esami_lab_par_cal[["idana", "idcentro", "data"]],
            df_esami_strumentali[["idana", "idcentro", "data"]],
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
                df_prescrizioni_diabete_farmaci[["idana", "idcentro", "data"]],
                df_prescrizioni_diabete_non_farmaci[["idana", "idcentro", "data"]],
                df_prescrizioni_non_diabete[["idana", "idcentro", "data"]],
            ]
        ),
        join="inner",
    ).reset_index()

esami_and_prescrizioni_grouped = esami_and_prescrizioni_concat.groupby(
    ["idana", "idcentro"]
).min()

df_anagrafica_attivi_con_data_minima = df_anagrafica_attivi_primo_accesso_nan.merge(
    esami_and_prescrizioni_grouped, on=["idana", "idcentro"]
)

df_anagrafica_attivi_con_data_minima = df_anagrafica_attivi_con_data_minima[
    (
        df_anagrafica_attivi_con_data_minima["data"]
        > df_anagrafica_attivi_con_data_minima["annonascita"]
    )
    & (
        df_anagrafica_attivi_con_data_minima["data"]
        < df_anagrafica_attivi_con_data_minima["annodecesso"].fillna(pd.Timestamp.now())
    )
]

df_anagrafica_attivi = df_anagrafica_attivi.merge(
    df_anagrafica_attivi_con_data_minima[["idana", "idcentro", "data"]],
    on=["idana", "idcentro"],
    how="left",
)

df_anagrafica_attivi.loc[
    df_anagrafica_attivi["annoprimoaccesso"].isna(), "annoprimoaccesso"
] = df_anagrafica_attivi.loc[df_anagrafica_attivi["annoprimoaccesso"].isna(), "data"]

df_anagrafica_attivi = df_anagrafica_attivi.drop(columns="data")

df_anagrafica_attivi = df_anagrafica_attivi[
    df_anagrafica_attivi["annoprimoaccesso"].notna()
]

del (
    df_anagrafica_attivi_primo_accesso_nan,
    esami_and_prescrizioni_concat,
    esami_and_prescrizioni_grouped,
    df_anagrafica_attivi_con_data_minima,
)

# Now we take care of the NaN records of "anno diagnosi diabete"
print(
    "Number of records in anagrafica where tipo diabete is NaN:",
    sum(df_anagrafica_attivi["tipodiabete"].isna()),
)  # 0

# Since all of the patients have diabetes we can fill the records where "anno diagnosi diabete" is NaN
# with the value of "anno primo accesso"
df_anagrafica_attivi.loc[
    df_anagrafica_attivi["annodiagnosidiabete"].isna(), "annodiagnosidiabete"
] = df_anagrafica_attivi.loc[
    df_anagrafica_attivi["annodiagnosidiabete"].isna(), "annoprimoaccesso"
]

# Now we are going to compare dates inside "anagrafica" in order to spot inconsistent ones
print(
    "Number of records in anagrafica where anno nascita is less than 1900:",
    sum(df_anagrafica_attivi["annonascita"] < pd.to_datetime(1900, format="%Y")),
)  # 0

print(
    "Number of records in anagrafica where anno decesso is more than 2023:",
    sum(df_anagrafica_attivi["annodecesso"] > pd.to_datetime(2023, format="%Y")),
)  # 0

print(
    "Number of records in anagrafica where anno diagnosi diabete is less than anno nascita:",
    sum(
        df_anagrafica_attivi["annodiagnosidiabete"]
        < df_anagrafica_attivi["annonascita"]
    ),
)  # 0

print(
    "Number of records in anagrafica where anno primo accesso is less than anno nascita:",
    sum(df_anagrafica_attivi["annoprimoaccesso"] < df_anagrafica_attivi["annonascita"]),
)  # 0

print(
    "Number of records in anagrafica where anno decesso is less than anno nascita:",
    sum(
        df_anagrafica_attivi["annodecesso"].fillna(pd.Timestamp.now())
        < df_anagrafica_attivi["annonascita"]
    ),
)  # 0

print(
    "Number of records in anagrafica where anno nascita is more than anno decesso:",
    sum(
        df_anagrafica_attivi["annonascita"]
        > df_anagrafica_attivi["annodecesso"].fillna(pd.Timestamp.now())
    ),
)  # 0

print(
    "Number of records in anagrafica where anno primo accesso is more than anno decesso:",
    sum(
        df_anagrafica_attivi["annoprimoaccesso"]
        > df_anagrafica_attivi["annodecesso"].fillna(pd.Timestamp.now())
    ),
)  # 7

df_anagrafica_attivi = df_anagrafica_attivi[
    df_anagrafica_attivi["annoprimoaccesso"]
    <= df_anagrafica_attivi["annodecesso"].fillna(pd.Timestamp.now())
]

print(
    "Number of records in anagrafica where anno diagnosi diabete is more than anno decesso:",
    sum(
        df_anagrafica_attivi["annodiagnosidiabete"]
        > df_anagrafica_attivi["annodecesso"].fillna(pd.Timestamp.now())
    ),
)  # 1

df_anagrafica_attivi = df_anagrafica_attivi[
    df_anagrafica_attivi["annodiagnosidiabete"]
    <= df_anagrafica_attivi["annodecesso"].fillna(pd.Timestamp.now())
]

# Step 2 for tables other than anagrafica
# First we extract ids and dates of patients of interest
aa_cuore_dates = df_anagrafica_attivi[
    [
        "idana",
        "idcentro",
        "annonascita",
        "annodecesso",
    ]
].drop_duplicates()


# Here we merge every table with id of anagrafica and delete all inconsistent dates
def clean_between_dates(df, col="data", col_start="annonascita", col_end="annodecesso"):
    # This creates a temporary df with only patients of interest
    df1 = df.merge(aa_cuore_dates, on=["idana", "idcentro"], how="inner")

    # Here we filter inconsistent dates
    df1 = df1[
        (df1[col].dt.year >= df1[col_start].dt.year)
        & (df1[col].dt.year <= df1[col_end].fillna(pd.Timestamp.now()).dt.year)
    ]

    df1 = df1.drop(columns=[col_start, col_end])

    return df1


print(
    f"Number of records in anagrafica pazienti attivi after step 2: {len(df_anagrafica_attivi)}"
)  # 49193

df_diagnosi = clean_between_dates(df_diagnosi)

print(f"Number of records in diagnosi after step 2: {len(df_diagnosi)}")  # 1910906

df_esami_lab_par = clean_between_dates(df_esami_lab_par)

print(
    f"Number of records in esami laboratorio parametri after step 2: {len(df_esami_lab_par)}"
)  # 7267641

df_esami_lab_par_cal = clean_between_dates(df_esami_lab_par_cal)

print(
    f"Number of records in esami laboratorio parametri calcolati after step 2: {len(df_esami_lab_par_cal)}"
)  # 2729420

df_esami_strumentali = clean_between_dates(df_esami_strumentali)

print(
    f"Number of records in esami strumentali after step 2: {len(df_esami_strumentali)}"
)  # 288237

df_prescrizioni_diabete_farmaci = clean_between_dates(df_prescrizioni_diabete_farmaci)

print(
    f"Number of records in prescrizioni diabete farmaci after step 2: {len(df_prescrizioni_diabete_farmaci)}"
)  # 1956231

df_prescrizioni_diabete_non_farmaci = clean_between_dates(
    df_prescrizioni_diabete_non_farmaci
)

print(
    f"Number of records in prescrizioni diabete non farmaci after step 2: {len(df_prescrizioni_diabete_non_farmaci)}"
)  # 148844

df_prescrizioni_non_diabete = clean_between_dates(df_prescrizioni_non_diabete)

print(
    f"Number of records in prescrizioni non diabete after step 2: {len(df_prescrizioni_non_diabete)}"
)  # 1961712

"""
df_list = [
    df_diagnosi,
    df_esami_lab_par,
    df_esami_lab_par_cal,
    df_esami_strumentali,
    df_prescrizioni_diabete_farmaci,
    df_prescrizioni_diabete_non_farmaci,
    df_prescrizioni_non_diabete,
]

for df in df_list:
    df = df.merge(aa_cuore_dates, on=["idana", "idcentro"], how="inner")
    print(len(df[df["data"].dt.year < df["annonascita"].dt.year]))
    print(len(df[df["data"].dt.year > df["annodecesso"].fillna(pd.Timestamp.now()).dt.year]))
"""

del df_list, aa_cuore_dates

#######################################
############### STEP 3 ################
#######################################

print("########## STEP 3 ##########")

# First of all we need to combine all the tables regarding patients events
esami_and_prescrizioni_concat = pd.concat(
    objs=(
        idf.set_index(["idana", "idcentro"])
        for idf in [
            df_diagnosi[["idana", "idcentro", "data"]],
            df_esami_lab_par[["idana", "idcentro", "data"]],
            df_esami_lab_par_cal[["idana", "idcentro", "data"]],
            df_esami_strumentali[["idana", "idcentro", "data"]],
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
                df_prescrizioni_diabete_farmaci[["idana", "idcentro", "data"]],
                df_prescrizioni_diabete_non_farmaci[["idana", "idcentro", "data"]],
                df_prescrizioni_non_diabete[["idana", "idcentro", "data"]],
            ]
        ),
        join="inner",
    ).reset_index()

# Then we extract the least recent and the more recent event
esami_and_prescrizioni_grouped = esami_and_prescrizioni_concat.groupby(
    ["idana", "idcentro"]
).agg({"data": ["min", "max"]})

# And we compute the difference
esami_and_prescrizioni_grouped["diff"] = (
    esami_and_prescrizioni_grouped["data"]["max"]
    - esami_and_prescrizioni_grouped["data"]["min"]
)

print(
    "Number of records in esami and prescrizioni concateneted where the clinical history spans less than a month:",
    len(
        esami_and_prescrizioni_grouped[
            esami_and_prescrizioni_grouped["diff"] <= pd.Timedelta("31 days")
        ]
    ),
)  # 797

# Finally we delete all the patients whose clinical history spans less than a month
esami_and_prescrizioni_grouped = esami_and_prescrizioni_grouped[
    esami_and_prescrizioni_grouped["diff"] > pd.Timedelta("31 days")
]

# Now we select the ids in order to filter all the tables
esami_and_prescrizioni_grouped_keys = (
    esami_and_prescrizioni_grouped.stack()
    .reset_index()[["idana", "idcentro"]]
    .drop_duplicates()
)

df_anagrafica_attivi = df_anagrafica_attivi.merge(
    esami_and_prescrizioni_grouped_keys, on=["idcentro", "idana"], how="inner"
)

print(
    f"Number of records in anagrafica pazienti attivi after step 3: {len(df_anagrafica_attivi)}"
)  # 48354

df_diagnosi = df_diagnosi.merge(
    esami_and_prescrizioni_grouped_keys, on=["idcentro", "idana"], how="inner"
)

print(f"Number of records in diagnosi after step 3: {len(df_diagnosi)}")  # 1905915

df_esami_lab_par = df_esami_lab_par.merge(
    esami_and_prescrizioni_grouped_keys, on=["idcentro", "idana"], how="inner"
)

print(
    f"Number of records in esami laboratorio parametri after step 3: {len(df_esami_lab_par)}"
)  # 7259736

df_esami_lab_par_cal = df_esami_lab_par_cal.merge(
    esami_and_prescrizioni_grouped_keys, on=["idcentro", "idana"], how="inner"
)

print(
    f"Number of records in esami laboratorio parametri calcolati after step 3: {len(df_esami_lab_par_cal)}"
)  # 2726328

df_esami_strumentali = df_esami_strumentali.merge(
    esami_and_prescrizioni_grouped_keys, on=["idcentro", "idana"], how="inner"
)

print(
    f"Number of records in esami strumentali after step 3: {len(df_esami_strumentali)}"
)  # 287848

df_prescrizioni_diabete_farmaci = df_prescrizioni_diabete_farmaci.merge(
    esami_and_prescrizioni_grouped_keys, on=["idcentro", "idana"], how="inner"
)

print(
    f"Number of records in prescrizioni diabete farmaci after step 3: {len(df_prescrizioni_diabete_farmaci)}"
)  # 1954076

df_prescrizioni_diabete_non_farmaci = df_prescrizioni_diabete_non_farmaci.merge(
    esami_and_prescrizioni_grouped_keys, on=["idcentro", "idana"], how="inner"
)

print(
    f"Number of records in prescrizioni diabete non farmaci after step 3: {len(df_prescrizioni_diabete_non_farmaci)}"
)  # 148638

df_prescrizioni_non_diabete = df_prescrizioni_non_diabete.merge(
    esami_and_prescrizioni_grouped_keys, on=["idcentro", "idana"], how="inner"
)

print(
    f"Number of records in prescrizioni non diabete after step 3: {len(df_prescrizioni_non_diabete)}"
)  # 1959007

del esami_and_prescrizioni_grouped, esami_and_prescrizioni_grouped_keys

#######################################
############### STEP 4 ################
#######################################

print("########## STEP 4 ##########")

"""
AMD004
AMD005
AMD006
AMD007
AMD008
AMD009
AMD111
STITCH001
STITCH002
STITCH003
STITCH004
STITCH005
"""

# Here we are going to change the ranges of the codes above
amd004 = df_esami_lab_par[df_esami_lab_par["codiceamd"] == "AMD004"]["valore"]

print(
    "Number of records in esami laboratorio parametri where the AMD004 value is outside the correct range:",
    len(amd004[amd004 < 40]) + len(amd004[amd004 > 200]),
)  # 1410

mask = df_esami_lab_par["codiceamd"] == "AMD004"
df_esami_lab_par.loc[mask, "valore"] = df_esami_lab_par.loc[mask, "valore"].clip(
    40, 200
)

amd005 = df_esami_lab_par[df_esami_lab_par["codiceamd"] == "AMD005"]["valore"]

print(
    "Number of records in esami laboratorio parametri where the AMD005 value is outside the correct range:",
    len(amd005[amd005 < 40]) + len(amd005[amd005 > 130]),
)  # 764

mask = df_esami_lab_par["codiceamd"] == "AMD005"
df_esami_lab_par.loc[mask, "valore"] = df_esami_lab_par.loc[mask, "valore"].clip(
    40, 130
)

amd007 = df_esami_lab_par[df_esami_lab_par["codiceamd"] == "AMD007"]["valore"]

print(
    "Number of records in esami laboratorio parametri where the AMD007 value is outside the correct range:",
    len(amd007[amd007 < 50]) + len(amd007[amd007 > 500]),
)  # 3023

mask = df_esami_lab_par["codiceamd"] == "AMD007"
df_esami_lab_par.loc[mask, "valore"] = df_esami_lab_par.loc[mask, "valore"].clip(
    50, 500
)

amd008 = df_esami_lab_par[df_esami_lab_par["codiceamd"] == "AMD008"]["valore"]

print(
    "Number of records in esami laboratorio parametri where the AMD008 value is outside the correct range:",
    len(amd008[amd008 < 5]) + len(amd008[amd008 > 15]),
)  # 5269

mask = df_esami_lab_par["codiceamd"] == "AMD008"
df_esami_lab_par.loc[mask, "valore"] = df_esami_lab_par.loc[mask, "valore"].clip(5, 15)

stitch002 = df_esami_lab_par_cal[df_esami_lab_par_cal["codicestitch"] == "STITCH002"][
    "valore"
]

print(
    "Number of records in esami laboratorio parametri where the STITCH002 value is outside the correct range:",
    len(stitch002[stitch002 < 30]) + len(stitch002[stitch002 > 300]),
)  # 4159

mask = df_esami_lab_par_cal["codicestitch"] == "STITCH002"
df_esami_lab_par_cal.loc[mask, "valore"] = df_esami_lab_par_cal.loc[
    mask, "valore"
].clip(30, 300)

stitch003 = df_esami_lab_par_cal[df_esami_lab_par_cal["codicestitch"] == "STITCH003"][
    "valore"
]

print(
    "Number of records in esami laboratorio parametri where the STITCH003 value is outside the correct range:",
    len(stitch003[stitch003 < 60]) + len(stitch003[stitch003 > 330]),
)  # 9636

mask = df_esami_lab_par_cal["codicestitch"] == "STITCH003"
df_esami_lab_par_cal.loc[mask, "valore"] = df_esami_lab_par_cal.loc[
    mask, "valore"
].clip(60, 330)

print("No changes to the dimension of the tables with respect to the previous step")

#######################################
############### STEP 5 ################
#######################################

print("########## STEP 5 ##########")

# First of all we have to select those patients that have at least two events
# So we filter all the events that we concateneted at step 3
esami_and_prescrizioni_concat = esami_and_prescrizioni_concat.merge(
    df_anagrafica_attivi[["idana", "idcentro"]].drop_duplicates(),
    on=["idana", "idcentro"],
    how="inner",
)

# We compute the count of the all the events related to a patient
events_count = (
    esami_and_prescrizioni_concat.groupby(["idana", "idcentro"])
    .size()
    .reset_index(name="count")
)

print(
    "Number of patients that have less than 2 events in their clinical history:",
    len(events_count[events_count["count"] < 2]),
)  # 0

# And we filter those that have less than 2 events in their history
events_count = events_count[events_count["count"] >= 2]

esami_and_prescrizioni_concat = esami_and_prescrizioni_concat.merge(
    events_count[["idana", "idcentro"]].drop_duplicates(),
    on=["idana", "idcentro"],
    how="inner",
)

# Now we take the last event in order to create the new label
last_event = esami_and_prescrizioni_concat.groupby(["idana", "idcentro"]).max()

df_anagrafica_attivi = df_anagrafica_attivi.merge(
    last_event, on=["idana", "idcentro"], how="inner"
)

df_anagrafica_attivi = df_anagrafica_attivi.rename(columns={"data": "last_event"})

df_diagnosi = df_diagnosi.merge(
    esami_and_prescrizioni_concat[["idana", "idcentro"]].drop_duplicates(),
    on=["idana", "idcentro"],
    how="inner",
)

df_diagnosi_cardio_filtered = df_diagnosi[
    df_diagnosi["codiceamd"].isin(AMD_OF_CARDIOVASCULAR_EVENT)
]

last_problem = (
    df_diagnosi_cardio_filtered[["idana", "idcentro", "data"]]
    .groupby(["idana", "idcentro"])
    .max()
)

df_anagrafica_attivi = df_anagrafica_attivi.merge(
    last_problem, on=["idana", "idcentro"], how="inner"
)

df_anagrafica_attivi = df_anagrafica_attivi.rename(columns={"data": "last_problem"})

df_anagrafica_attivi["label"] = (
    df_anagrafica_attivi["last_event"] - df_anagrafica_attivi["last_problem"]
) <= pd.Timedelta(days=186)


print(df_anagrafica_attivi["label"].value_counts())

# delete wanted_patient with trajectory less than 6 months
events_max_min = esami_and_prescrizioni_concat.groupby(["idana", "idcentro"]).agg(
    {"data": ["min", "max"]}
)

events_max_min["diff"] = events_max_min["data"]["max"] - events_max_min["data"]["min"]

print(
    "Number of patients that have a clinical history shorter than or equal to 6 months:",
    len(events_max_min[events_max_min["diff"] <= pd.Timedelta(days=186)]),
)  # 736

events_max_min = events_max_min[events_max_min["diff"] > pd.Timedelta(days=186)]

events_max_min_keys = (
    events_max_min.stack().reset_index()[["idana", "idcentro"]].drop_duplicates()
)

df_anagrafica_attivi = df_anagrafica_attivi.drop(columns=["last_event", "last_problem"])

df_anagrafica_attivi = df_anagrafica_attivi.merge(
    events_max_min_keys, on=["idana", "idcentro"], how="inner"
)

#######################################
############### STEP 6 ################
#######################################

# some things for point 6 are done in point 2 and 3 to speed up computations
print("############## POINT 6 START ##############")

print("Patients labels: ")
print(df_anagrafica_attivi["label"].value_counts())

print("Number of patients: ", len(df_anagrafica_attivi))
print("Patient registry before point 6: ")
print(df_anagrafica_attivi.isna().sum())
# annoprimoaccesso è stato fillato precedentemente e le informazioni demografiche sono
# spesso mancanti ma possono essere tenute usando un [UNK] in seguito

# print("tipi possibili di diabete: ", df_anagrafica_attivi["tipodiabete"].unique())
# since tipodiabete is always the same we could delete the column to lower the memory usage
df_anagrafica_attivi = df_anagrafica_attivi.drop(columns=["tipodiabete"])

# delete columns origine because it's almost always nan
df_anagrafica_attivi = df_anagrafica_attivi.drop(columns=["origine"])

print("Patient registry after point 6: ")
print(df_anagrafica_attivi.isna().sum())

print("Diagnosis before point 6: ")
df_diagnosi = df_diagnosi.merge(
    df_anagrafica_attivi[["idana", "idcentro"]], on=["idana", "idcentro"], how="inner"
)
print(df_diagnosi.isna().sum())
# qui ci sono 33k righe con valore a nan
# non sono interessanti poichè non sono tra le diagnosi per noi interessanti (quelle cardiovascolari),
# quindi non vanno riempite ma le terrei per avere comunque altre informazioni sui pazienti
df_diagnosi_nan = (
    df_diagnosi[df_diagnosi["valore"].isna()]
    .groupby(["codiceamd"])
    .size()
    .sort_values(ascending=False)
)
print("nan diagnosis values:\n", df_diagnosi_nan)

# here we modify the valore of cardiovascolar events interesting to us to have a more cleaned dataset
# print(
#     "AMD049: ",
#     df_diagnosi[df_diagnosi["codiceamd"] == "AMD049"]["valore"].value_counts(),
# )
# modify the values of the column valore where codiceamd == amd049 to S
# because imbalanced wrt the other values see below:
# valore
# S       34975
# 36.1      108
mask = df_diagnosi["codiceamd"] == "AMD049"
df_diagnosi.loc[mask, "valore"] = "S"

# print(
#     "AMD303: ",
#     df_diagnosi[df_diagnosi["codiceamd"] == "AMD303"]["valore"].value_counts(),
# )
# modify the values of the column valore where codiceamd == amd303 to 434.91
# because imbalanced wrt the other values see below:
# valore
# 434.91    10128
# 433.01        7
# 433.11        2
# 434.01        2
# 433.21        1
# 433.91        1
mask = df_diagnosi["codiceamd"] == "AMD303"
df_diagnosi.loc[mask, "valore"] = "434.91"

# print(
#     "AMD081: ",
#     df_diagnosi[df_diagnosi["codiceamd"] == "AMD081"]["valore"].value_counts(),
# )
# modify the values of the column valore where codiceamd == amd081 to 39.5
# because imbalanced wrt the other values see below:
# valore
# 39.5     9212
# 39.50     780
mask = df_diagnosi["codiceamd"] == "AMD081"
df_diagnosi.loc[mask, "valore"] = "39.5"

# amd047 and amd071 are unbalanced but not so much so I don't modify them
# print(
#     "AMD047: ",
#     df_diagnosi[df_diagnosi["codiceamd"] == "AMD047"]["valore"].value_counts(),
# )
# print(
#     "AMD071: ",
#     df_diagnosi[df_diagnosi["codiceamd"] == "AMD071"]["valore"].value_counts(),
# )

print("Diagnosis after point 6: ")
print(df_diagnosi.isna().sum())

# I think the values for all the wanted codiceamd are not extremely relevant so I have filled them,
# only because my lack of medical knowledge

print("Exams lab parameters before point 6: ")
df_esami_lab_par = df_esami_lab_par.merge(
    df_anagrafica_attivi[["idana", "idcentro"]], on=["idana", "idcentro"], how="inner"
)

print(df_esami_lab_par.isna().sum())
# qui ci sono 30k righe con valore a nan

df_esami_lab_par_nan = (
    df_esami_lab_par[df_esami_lab_par["valore"].isna()]
    .groupby(["codiceamd"])
    .size()
    .sort_values(ascending=False)
)
print("nan diagnosis values:\n", df_esami_lab_par_nan)
# i seguenti codice amd hanno i rispettivi nan:
# codiceamd
# AMD009    28581
# AMD001     1580

df_esami_lab_par_temp = df_esami_lab_par.merge(
    df_anagrafica_attivi[["idana", "idcentro", "sesso"]],
    on=["idana", "idcentro"],
    how="inner",
)
# amd001 represent the height of the patient so I can fill the nan with the mean of the height by sex
mask = (df_esami_lab_par["codiceamd"] == "AMD001") & df_esami_lab_par["valore"].isna()
df_esami_lab_par.loc[mask, "valore"] = df_esami_lab_par_temp.groupby(["sesso"])[
    "valore"
].transform(lambda x: x.fillna(x.mean()))

# ora i nan sono solo i 28k degli amd009 per cui non si può effettuare un fill in quanto dati medici
print("Exams lab parameters after point 6: ")
print(df_esami_lab_par.isna().sum())

print("Exams lab parameters calculated before point 6: ")
df_esami_lab_par_cal = df_esami_lab_par_cal.merge(
    df_anagrafica_attivi[["idana", "idcentro"]], on=["idana", "idcentro"], how="inner"
)
print(df_esami_lab_par_cal.isna().sum())
# qui ci sono quasi 900k righe con codiceamd nan

print(df_esami_lab_par_cal.groupby(["codiceamd"]).size())
print(df_esami_lab_par_cal.groupby(["codicestitch"]).size())
# una parte dei codiciamd mancanti possono essere fillati in base al valore del codice stitch
# quindi va fatta un analisi raggruppando per codice stitch e poi per codice amd in modo da
# vedere quali sono le caratterisitche per il fill dei codici amd mancanti
# qui sotto si vede che gli 900k codici amd mancanti hanno tutti codici stitch 003 e 004
print(
    df_esami_lab_par_cal[df_esami_lab_par_cal["codiceamd"].isna()]["codicestitch"]
    .isin(["STITCH003", "STITCH004"])
    .sum()
)

# raggruppa per codice stitch e poi per codice amd
# da qui si vede proprio che i codici stitch e gli amd sono legati da:
# codicestitch  codiceamd
# STITCH001     AMD927       959396
# STITCH002     AMD013       339590
# STITCH005     AMD304       527468
# quindi non è possibile fare un fill dei codici amd mancanti in base al codice stitch
# poichè non ci sono relazioni tra gli stich 003 e 004 e gli amd
# praticamente gli stitch 003 e 004 sono l'unica informazione utilizzabile piuttosto di amd e stitch insieme
print(df_esami_lab_par_cal.groupby(["codicestitch", "codiceamd"]).size())
# TODO: che si fa si infila lo stitch nell'amd per queste 900k righe o si creano due nuovi amd appositi?
# oppure si potrebbe eliminare completamente il codice amd e usare solo il codice stitch con relativa descrizione
# da aggiungere come nel file amd_codes_for_bert.

print("Exams lab parameters calculated after point 6: ")
print(df_esami_lab_par_cal.isna().sum())

print("Exams instrumental before point 6: ")
df_esami_strumentali = df_esami_strumentali.merge(
    df_anagrafica_attivi[["idana", "idcentro"]], on=["idana", "idcentro"], how="inner"
)
print(df_esami_strumentali.isna().sum())
# qui ci sono 21k righe con valore a nan

print(df_esami_strumentali.groupby(["codiceamd"]).size())
# alcuni codici amd sono presenti in proporzioni molto maggiori rispetto ad altri

# ragruppando i codici amd per quantità di nan in valore
# si vede che i codici amd con valori nan sono solo amd125 con 21k righe e amd126 solo 4
# quindi si potrebbe fare un fill dei valori nan in base al valore più presente nel caso del codice amd126 che è N
# mentre nel caso del codice amd125 non si può fare un fill in quanto si tratta di fillare metà delle righe e quindi
# potrebbe portare pi problemi che benefici
df_esami_strumentali_nan = (
    df_esami_strumentali[df_esami_strumentali["valore"].isna()]
    .groupby(["codiceamd"])
    .size()
    .sort_values(ascending=False)
)
print("nan exams instrumental values:\n", df_esami_strumentali_nan)

print(
    "amd126:\n",
    df_esami_strumentali[df_esami_strumentali["codiceamd"] == "AMD126"][
        "valore"
    ].value_counts(),
)
print(
    "amd125:\n",
    df_esami_strumentali[df_esami_strumentali["codiceamd"] == "AMD125"][
        "valore"
    ].value_counts(),
)
# fill valore for codiceamd == amd126 that are nan with the value most present in the column
# valore for codiceamd == amd126 that is N
mask = (df_esami_strumentali["codiceamd"] == "AMD126") & df_esami_strumentali[
    "valore"
].isna()
df_esami_strumentali.loc[mask, "valore"] = "N"

# we don't fill the nan for codiceamd == amd125 because it's too much nan values circa 50%

print("Exams instrumental after point 6: ")
print(df_esami_strumentali.isna().sum())

print("Prescription diabete drugs before point 6: ")
df_prescrizioni_diabete_farmaci = df_prescrizioni_diabete_farmaci.merge(
    df_anagrafica_attivi[["idana", "idcentro"]], on=["idana", "idcentro"], how="inner"
)
print(df_prescrizioni_diabete_farmaci.isna().sum())
# qui ci sono 38 righe con codice atc nan
print(
    "Count by codiceatc:\n",
    df_prescrizioni_diabete_farmaci.groupby(["codiceatc"]).size(),
)
# print(
#     df_prescrizioni_diabete_farmaci.groupby(["codiceatc", "descrizionefarmaco"]).size()
# )
df_prescrizioni_diabete_farmaci_nan = (
    df_prescrizioni_diabete_farmaci[df_prescrizioni_diabete_farmaci["codiceatc"].isna()]
    .groupby(["descrizionefarmaco"])
    .size()
    .sort_values(ascending=False)
)
print("nan prescription diabete drugs", df_prescrizioni_diabete_farmaci_nan)

# siccome le descrizioni dei farmaci dei 38 con codice atc nan sono:
# descrizionefarmaco
# Altro               24
# Ipoglic. orale 1    12
# 30/70                2
# possiamo provare a fare un fill dei codici atc nan in base alla descrizione del farmaco
# vedendo quali codici atc presentano più volte quelle descrizioni
# print(
#     "Altro: ",
#     df_prescrizioni_diabete_farmaci[
#         df_prescrizioni_diabete_farmaci["descrizionefarmaco"] == "Altro"
#     ]["codiceatc"].value_counts(),
# )

# print(
#     "Ipoglic. orale 1: ",
#     df_prescrizioni_diabete_farmaci[
#         df_prescrizioni_diabete_farmaci["descrizionefarmaco"] == "Ipoglic. orale 1"
#     ]["codiceatc"].value_counts(),
# )
# print(
#     "30/70: ",
#     df_prescrizioni_diabete_farmaci[
#         df_prescrizioni_diabete_farmaci["descrizionefarmaco"] == "30/70"
#     ]["codiceatc"].value_counts(),
# )
# siccome dalla descrizione non è possibile capire quale codiceatc sia associato (nemmeno nel dataset non pulito)
# in quanto per queste descrizioni non vi è mai un codice atc associato, non è possibile effettuare un fill
# quindi si potrebbe eliminare queste 38 righe oppure creare 3 codiciatc nuovi

print("Prescription diabete drugs after point 6: ")
print(df_prescrizioni_diabete_farmaci.isna().sum())

print("Prescription diabete not drugs before point 6: ")
df_prescrizioni_diabete_non_farmaci = df_prescrizioni_diabete_non_farmaci.merge(
    df_anagrafica_attivi[["idana", "idcentro"]], on=["idana", "idcentro"], how="inner"
)

print(df_prescrizioni_diabete_non_farmaci.isna().sum())
# qui ci sono 15k righe con valore nan
print(
    "Count by codiceamd:\n",
    df_prescrizioni_diabete_non_farmaci.groupby(["codiceamd"]).size(),
)
# qui abbiamo un codice amd096 che è presente in sole 32 righe e quindi completamente
# sbilanciato rispetto agli altri codici amd presenti in grandi quantità, quindi lo scarterei,
# poi due codici amd086 e amd152 riportano la stessa descrizione ma differente valore e quindi
# non sono unibili in un unico codice (086 ha S/N e 152 un codice ministeriale).

# TODO: dal seguente codice si vede che gli unici amd con valori nan sono amd096 e amd152,
# quindi si potrebbe fare un fill dei valori nan in base al valore più presente nel caso del codice amd152
# mentre anche per questo motivo scarterei amd096 siccome soli 32 di cui 11 nan
df_prescrizioni_diabete_non_farmaci_nan = (
    df_prescrizioni_diabete_non_farmaci[
        df_prescrizioni_diabete_non_farmaci["valore"].isna()
    ]
    .groupby(["codiceamd"])
    .size()
    .sort_values(ascending=False)
)
print("nan prescription diabete not drugs:\n", df_prescrizioni_diabete_non_farmaci_nan)

# for now we are deleting only amd096 with nan values,
# but I think we must delete all amd096 because unbalanced
drop_mask = (
    df_prescrizioni_diabete_non_farmaci["codiceamd"] == "AMD096"
) & df_prescrizioni_diabete_non_farmaci["valore"].isna()

df_prescrizioni_diabete_non_farmaci = df_prescrizioni_diabete_non_farmaci.drop(
    df_prescrizioni_diabete_non_farmaci[drop_mask].index
)

# print(
#     "AMD152: ",
#     df_prescrizioni_diabete_non_farmaci[
#         df_prescrizioni_diabete_non_farmaci["codiceamd"] == "AMD152"
#     ]["valore"].value_counts(),
# )

# count the number of patient with amd152 whose valore is not nan
count152 = df_prescrizioni_diabete_non_farmaci[
    (df_prescrizioni_diabete_non_farmaci["codiceamd"] == "AMD152")
    & (~df_prescrizioni_diabete_non_farmaci["valore"].isna())
].shape[0]

# count the number of patients with amd152 and the respective values and create a dict
count152_dict = (
    df_prescrizioni_diabete_non_farmaci[
        (df_prescrizioni_diabete_non_farmaci["codiceamd"] == "AMD152")
        & (~df_prescrizioni_diabete_non_farmaci["valore"].isna())
    ]["valore"]
    .value_counts()
    .to_dict()
)

# for each value in the dict compute the probability of that value
for key in count152_dict:
    count152_dict[key] = count152_dict[key] / count152

# print(count152_dict)

# for each nan value in the column valore and with codiceamd equal to amd152 assign a value
# with a probabilistic approach where the probability of a value is the number of times that value
# appears in the dataset divided by the total number of values
df_prescrizioni_diabete_non_farmaci.loc[
    (df_prescrizioni_diabete_non_farmaci["codiceamd"] == "AMD152")
    & (df_prescrizioni_diabete_non_farmaci["valore"].isna()),
    "valore",
] = np.random.choice(
    list(count152_dict.keys()),
    size=df_prescrizioni_diabete_non_farmaci[
        (df_prescrizioni_diabete_non_farmaci["codiceamd"] == "AMD152")
        & (df_prescrizioni_diabete_non_farmaci["valore"].isna())
    ].shape[0],
    p=list(count152_dict.values()),
)

print("Prescription diabete not drugs after point 6: ")
print(df_prescrizioni_diabete_non_farmaci.isna().sum())

print("Prescription not diabete before point 6: ")
# qui non ci sono nan
df_prescrizioni_non_diabete = df_prescrizioni_non_diabete.merge(
    df_anagrafica_attivi[["idana", "idcentro"]], on=["idana", "idcentro"], how="inner"
)

print(df_prescrizioni_non_diabete.isna().sum())
print("here no nan")

# EXPORT THE CLEANED DATASETS
# Cleaned datasets are exported in the folder clean_data to be used in the next tasks.
# This  operation can take some minutes.

if WRITE_CSV:
    print("Exporting the cleaned datasets...")
    dict_file_names = {
        f"anagraficapazientiattivi_c{'_pres' if PRESCRIZIONI else ''}": df_anagrafica_attivi,
        f"diagnosi_c{'_pres' if PRESCRIZIONI else ''}": df_diagnosi,
        f"esamilaboratorioparametri_c{'_pres' if PRESCRIZIONI else ''}": df_esami_lab_par,
        f"esamilaboratorioparametricalcolati_c{'_pres' if PRESCRIZIONI else ''}": df_esami_lab_par_cal,
        f"esamistrumentali_c{'_pres' if PRESCRIZIONI else ''}": df_esami_strumentali,
        f"prescrizionidiabetefarmaci_c{'_pres' if PRESCRIZIONI else ''}": df_prescrizioni_diabete_farmaci,
        f"prescrizionidiabetenonfarmaci_c{'_pres' if PRESCRIZIONI else ''}": df_prescrizioni_diabete_non_farmaci,
        f"prescrizioninondiabete_c{'_pres' if PRESCRIZIONI else ''}": df_prescrizioni_non_diabete,
    }

    for i, (df_name, df) in enumerate(dict_file_names.items()):
        df.to_csv(f"{WRITE_DATA_PATH}/{df_name}.csv", index=False)
        print(f"{df_name}.csv exported ({i}/{len(dict_file_names)})")
    print("Exporting completed!")
