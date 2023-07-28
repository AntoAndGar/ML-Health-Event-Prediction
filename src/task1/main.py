import pandas as pd
from datetime import date
import concurrent.futures as futures
import multiprocessing

# from multiprocessing import Pool

# import the data
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


# read all the dataset concurrently and store them in a dictionary with the name of the file as key
with futures.ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
    df_list = dict()
    for name in file_names:
        df_list[str(name)] = executor.submit(read_csv, f"data/{name}.csv")

print("############## FUTURES CREATED ##############")

# with Pool(processes=multiprocessing.cpu_count()) as pool:
#     df_list1 = pool.map(read_csv, paths)

######## PUNTO 1 ########

# diagnosi table
df_diagnosi = df_list["diagnosi"].result()

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

print(
    "numero record presenti in diagnosi: ", len(df_diagnosi[["idana", "idcentro"]])
)  # 4427337
print(
    "numero pazienti presenti in diagnosi prima del punto 1: ",
    len(df_diagnosi[["idana", "idcentro"]].drop_duplicates()),
)  # 226303

# anagrafica table
df_anagrafica_attivi = df_list["anagraficapazientiattivi"].result()
# pd.read_csv("sample/anagraficapazientiattivi.csv", header=0, index_col=False)

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

print(
    "numero record presenti in anagrafica: ",
    len(df_anagrafica_attivi[["idana", "idcentro"]]),
)  # 250000
print(
    "numero pazienti presenti in anagrafica prima del punto 1: ",
    len(df_anagrafica_attivi[["idana", "idcentro"]].drop_duplicates()),
)  # 250000

print(
    "numero pazienti in anagrafica presenti in diagnosi:",
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

# Diagnosi relative a problemi cardiaci
df_diagnosi_problemi_cuore = df_diagnosi[
    df_diagnosi["codiceamd"].isin(AMD_OF_CARDIOVASCULAR_EVENT)
]

print(
    "numero pazienti presenti in diagnosi con codice amd in lista (con problemi al cuore): ",
    len(df_diagnosi_problemi_cuore[["idana", "idcentro"]].drop_duplicates()),
)  # 50000

# punto 6 fatto direttamente qui per alleggerire le computazioni
print(
    "numero pazienti presenti in diagnosi con codice amd in lista (con problemi al cuore) e con data presente: ",
    len(
        df_diagnosi_problemi_cuore[~df_diagnosi_problemi_cuore["data"].isna()][
            ["idana", "idcentro"]
        ].drop_duplicates()
    ),
)

# anagrafica pazienti con problemi al cuore, e relativa diagnosi
aa_prob_cuore = df_anagrafica_attivi.merge(
    df_diagnosi_problemi_cuore[~df_diagnosi_problemi_cuore["data"].isna()],
    on=["idcentro", "idana"],
    how="inner",
)

print(
    "numero pazienti presenti in anagrafica con problemi al cuore e data presente (dopo punto 1 e con 6): ",
    len(aa_prob_cuore[["idana", "idcentro"]].drop_duplicates()),
)

del df_diagnosi_problemi_cuore

# print("Valori presenti:", df_diagnosi_problemi_cuore["valore"].unique())

######## PUNTO 2 ########
print("############## POINT 2 START ##############")

print(
    "numero righe con anno diagnosi diabete minore dell'anno di nascita: ",
    sum(aa_prob_cuore["annodiagnosidiabete"] < aa_prob_cuore["annonascita"]),
)  # 0

print(
    "numero righe con anno primo accesso minore dell'anno di nascita: ",
    sum(aa_prob_cuore["annoprimoaccesso"] < aa_prob_cuore["annonascita"]),
)  # 0

print(
    "numero righe con anno decesso minore dell'anno di nascita: ",
    sum(
        aa_prob_cuore["annodecesso"].fillna(pd.Timestamp.now())
        < aa_prob_cuore["annonascita"]
    ),
)  # 3 # 0 se seleziono solo quelli con casi rilevanti

# TODO: elimina, siccome non più rilevanti in quanto non presenti tra i pazienti con problemi al cuore
# aa_prob_cuore = aa_prob_cuore[
#     aa_prob_cuore["annodecesso"].fillna(pd.Timestamp.now())
#     >= aa_prob_cuore["annonascita"]
# ]

# print(
#     "numero pazienti dopo scarto: ",
#     len(aa_prob_cuore[["idana", "idcentro"]].drop_duplicates()),
# )

# print(
#     "numero righe dopo scarto con anno decesso minore dell'anno di nascita: ",
#     sum(
#         aa_prob_cuore["annodecesso"].fillna(pd.Timestamp.now())
#         < aa_prob_cuore["annonascita"]
#     ),
# )

print(
    "numero pazienti con anno decesso maggiore dell'anno 2022: ",
    sum(
        aa_prob_cuore["annodecesso"] > pd.to_datetime(2023, format="%Y"),
    ),
)  # 0

# the conversion in datetime don't work for the year 0001 or 0000
# print(
#     "numero pazienti con anno di nascita negativo: ",
#     sum(
#         aa_prob_cuore["annonascita"] < pd.to_datetime("0001", format="%Y"),
#     ),
# )

print(
    "numero righe con anno primo accesso a N/A: ",
    sum(aa_prob_cuore["annoprimoaccesso"].isna()),
)  # 25571

print(
    "numero righe con anno diagnosi diabete a N/A: ",
    sum(aa_prob_cuore["annodiagnosidiabete"].isna()),
)  # 2234

print(
    "numero righe con anno primo accesso maggiore dell'anno decesso: ",
    sum(
        aa_prob_cuore["annoprimoaccesso"]
        > aa_prob_cuore["annodecesso"].fillna(pd.Timestamp.now())
    ),
)  # 34 (dopo scarto precedente 33)
# solo pazienti con eventi cardiovascolari: 14

aa_prob_cuore = aa_prob_cuore[
    (
        aa_prob_cuore["annoprimoaccesso"]
        <= aa_prob_cuore["annodecesso"].fillna(pd.Timestamp.now())
    )
    | (aa_prob_cuore["annoprimoaccesso"].isna())
]

print(
    "numero righe dopo scarto con anno primo accesso maggiore dell'anno decesso: ",
    sum(
        aa_prob_cuore["annoprimoaccesso"]
        > aa_prob_cuore["annodecesso"].fillna(pd.Timestamp.now())
    ),
)

print(
    "numero pazienti dopo scarto: ",
    len(aa_prob_cuore[["idana", "idcentro"]].drop_duplicates()),
)

# print("tipi possibili di diabete: ", aa_prob_cuore["tipodiabete"].unique())
# in anagrafica abbiamo solo pazienti con diagnosi di diabete di tipo 2 valore 5 in 'tipodiabete'
# quindi possiamo fillare l'annodiagnosidiabete con l'annoprimoaccesso

# visto che il tipo diabete è sempre lo stesso si può eliminare la colonna dal df per risparmiare memoria
aa_prob_cuore.drop(columns=["tipodiabete"], inplace=True)

print(
    "numero righe con anno diagnosi diabete maggiore dell'anno decesso: ",
    sum(
        aa_prob_cuore["annodiagnosidiabete"]
        > aa_prob_cuore["annodecesso"].fillna(pd.Timestamp.now())
    ),
)  # 22 (dopo scarto precedente 2)
# 14 solo paziente con eventi cadriovascolari

aa_prob_cuore = aa_prob_cuore[
    (
        aa_prob_cuore["annodiagnosidiabete"]
        <= aa_prob_cuore["annodecesso"].fillna(pd.Timestamp.now())
    )
    | aa_prob_cuore["annodiagnosidiabete"].isna()
]

print(
    "numero righe dopo scarto con anno diagnosi diabete maggiore dell'anno decesso: ",
    sum(
        aa_prob_cuore["annodiagnosidiabete"]
        > aa_prob_cuore["annodecesso"].fillna(pd.Timestamp.now())
    ),
)

print(
    "numero pazienti dopo scarto: ",
    len(aa_prob_cuore[["idana", "idcentro"]].drop_duplicates()),
)

print(
    "numero righe con anno diagnosi diabete a N/A, ma che hanno l'anno di primo accesso: ",
    len(
        aa_prob_cuore[
            aa_prob_cuore["annodiagnosidiabete"].isna()
            & aa_prob_cuore["annoprimoaccesso"].notnull()
        ][["idana", "idcentro"]]
    ),
)  # 1797
# TODO con questa info si potrebbe fillare l'annodiagnosidiabete con l'annoprimoaccesso

# print("info dataframe pazienti con problemi al cuore: ")
# print(aa_prob_cuore.info())

# questi son tutti a 0
print(
    "numero righe anno diagnosi diabete minore anno di nascita: ",
    sum(aa_prob_cuore["annodiagnosidiabete"] < aa_prob_cuore["annonascita"]),
)  # 0
print(
    "numero righe anno primo accesso minore anno di nascita: ",
    sum(aa_prob_cuore["annoprimoaccesso"] < aa_prob_cuore["annonascita"]),
)  # 0
print(
    "numero righe anno decesso minore anno di nascita: ",
    sum(aa_prob_cuore["annodecesso"] < aa_prob_cuore["annonascita"]),
)  # 0
# print(sum(aa_prob_cuore["annodecesso"] > 2022))  # 0
# print(sum(aa_prob_cuore["annonascita"] < 0))  # 0

# 7 pazienti hanno la data di primo accesso maggiore della data di decesso -> da scartare
# i 7 non sono presenti tra i pazienti con eventi cardiovascolari
print(
    "numero righe con data di primo accesso maggiore della data di decesso: ",
    sum(aa_prob_cuore["annoprimoaccesso"] > aa_prob_cuore["annodecesso"]),
)  # 14 righe di cui 7 unici # 0 tra i pazienti con eventi cardiovascolari

print(
    "numero pazienti unici con data di primo accesso maggiore della data di decesso: ",
    len(
        aa_prob_cuore[aa_prob_cuore["annoprimoaccesso"] > aa_prob_cuore["annodecesso"]][
            ["idana", "idcentro"]
        ].drop_duplicates()
    ),
)

# 5 pazienti hanno la data di diagnosi di diabete maggiore della data di decesso -> da scartare
# i 5 non sono presenti tra i pazienti con eventi cardiovascolari
print(
    "numero righe con data di diagnosi di diabete maggiore della data di decesso: ",
    sum(aa_prob_cuore["annodiagnosidiabete"] > aa_prob_cuore["annodecesso"]),
)  # 9 righe di cui 5 unici # 0 tra i pazienti con eventi cardiovascolari

print(
    "numero pazienti unici con data di diagnosi di diabete maggiore della data di decesso: ",
    len(
        aa_prob_cuore[
            aa_prob_cuore["annodiagnosidiabete"] > aa_prob_cuore["annodecesso"]
        ][["idana", "idcentro"]].drop_duplicates()
    ),
)

print(
    "numero righe con anno diagnosi diabete a N/A: ",
    sum(aa_prob_cuore["annodiagnosidiabete"].isna()),
)  # 2234

print(
    "numero pazienti unici con anno diagnosi diabete a N/A: ",
    len(
        aa_prob_cuore[aa_prob_cuore["annodiagnosidiabete"].isna()][
            ["idana", "idcentro"]
        ].drop_duplicates()
    ),
)  # 526

# in anagrafica abbiamo solo pazienti con diagnosi di diabete di tipo 2 valore 5 in 'tipodiabete'
# quindi possiamo fillare l'annodiagnosidiabete con l'annoprimoaccesso
print(
    "numero righe con anno diagnosi diabete a N/A ma con anno primo accesso presente: ",
    len(
        aa_prob_cuore[
            aa_prob_cuore["annodiagnosidiabete"].isna()
            & aa_prob_cuore["annoprimoaccesso"].notnull()
        ][["idana", "idcentro"]]
    ),
)  # 1797

print(
    "numero pazienti unici con anno diagnosi diabete a N/A ma con anno primo accesso presente: ",
    len(
        aa_prob_cuore[
            aa_prob_cuore["annodiagnosidiabete"].isna()
            & aa_prob_cuore["annoprimoaccesso"].notnull()
        ][["idana", "idcentro"]].drop_duplicates()
    ),
)  # 365

# fill annodiagnosidiabete with annoprimoaccesso where aa_prob_cuore["annodiagnosidiabete"].isna() & aa_prob_cuore["annoprimoaccesso"].notnull()
# for POINT 6
aa_prob_cuore.loc[
    aa_prob_cuore["annodiagnosidiabete"].isna()
    & aa_prob_cuore["annoprimoaccesso"].notnull(),
    "annodiagnosidiabete",
] = aa_prob_cuore.loc[
    aa_prob_cuore["annodiagnosidiabete"].isna()
    & aa_prob_cuore["annoprimoaccesso"].notnull(),
    "annoprimoaccesso",
]

print(
    "numero pazienti unici dopo fill con anno diagnosi diabete a N/A ma con anno primo accesso presente: ",
    len(
        aa_prob_cuore[
            aa_prob_cuore["annodiagnosidiabete"].isna()
            & aa_prob_cuore["annoprimoaccesso"].notnull()
        ][["idana", "idcentro"]].drop_duplicates()
    ),
)

print(
    "numero righe dopo fill con anno diagnosi diabete a N/A: ",
    sum(aa_prob_cuore["annodiagnosidiabete"].isna()),
)  # 437

print(
    "numero pazienti unici dopo fill con anno diagnosi diabete a N/A: ",
    len(
        aa_prob_cuore[aa_prob_cuore["annodiagnosidiabete"].isna()][
            ["idana", "idcentro"]
        ].drop_duplicates()
    ),
)  # 161

print(
    "numero righe dopo fill con anno primo accesso a N/A: ",
    sum(aa_prob_cuore["annoprimoaccesso"].isna()),
)  #

print(
    "numero pazienti unici dopo fill con anno primo accesso a N/A: ",
    len(
        aa_prob_cuore[aa_prob_cuore["annoprimoaccesso"].isna()][
            ["idana", "idcentro"]
        ].drop_duplicates()
    ),
)

print(len(aa_prob_cuore[["idana", "idcentro"]].drop_duplicates()))

print(aa_prob_cuore.isna().sum())

# print(aa_prob_cuore[aa_prob_cuore["data"].isna()])


def printSexInfo(dataset):
    dataset = dataset[["idcentro", "idana", "sesso"]].drop_duplicates()
    print("numero righe del df: ", len(dataset))

    print("Sex info")
    print(dataset["sesso"].unique())
    print("sesso ad N/A", dataset["sesso"].isna().sum())
    print("Maschi: ", sum(dataset["sesso"].isin(["M"])))
    print("Femmine: ", sum(dataset["sesso"].isin(["F"])))


def getDeaseasePercentage(dataset, deaseases):
    print("Deasease: ", deaseases)
    # print(dataset.columns)
    percent = "Percentage of deasease:\n"
    dataset = dataset[["idcentro", "idana", "codiceamd"]].drop_duplicates()
    print("numero righe del df: ", len(dataset))

    for deasease in deaseases:
        # print("Deasease: ", deasease)
        tempdataset = dataset[dataset["codiceamd"].isin([deasease])]
        tempdataset2 = tempdataset[["idana", "idcentro"]].drop_duplicates()
        total = len(dataset[["idana", "idcentro"]].drop_duplicates())
        percent += (
            deasease
            + ": "
            + str(len(tempdataset2) / total * 100)
            + "%\t"
            + str(len(tempdataset2))
            + " su "
            + str(total)
            + "\n"
        )
    print(percent)


def getInfoOnDiagnosi(df):
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print("Info on diagnosi")
    print(
        "il dizionario stampato è formattato così: 'chiave': [minori, uguali, maggiori] rispetto a data"
    )
    dates = ["annodiagnosidiabete", "annonascita", "annoprimoaccesso", "annodecesso"]
    stampami = dict()
    df = df[
        [
            "idcentro",
            "idana",
            "annodiagnosidiabete",
            "annonascita",
            "annoprimoaccesso",
            "annodecesso",
            "data",
        ]
    ].drop_duplicates()

    print("numero righe del df: ", len(df))

    for d in dates:
        # where the nan values goes? they are counted as what? minor or major?
        minor = (df[d] < df["data"]).sum()
        equal = (df[d] == df["data"]).sum()
        more = (df[d] > df["data"]).sum()
        stampami[d] = [minor, equal, more]

    print(stampami)


getInfoOnDiagnosi(aa_prob_cuore)
# Sesso
print("In anagrafica attivi abbiamo:")
printSexInfo(aa_prob_cuore)
print("Fra i pazienti con problemi al cuore abbiamo:")
printSexInfo(aa_prob_cuore)
# Deasease Distribution
getDeaseasePercentage(aa_prob_cuore, AMD_OF_CARDIOVASCULAR_EVENT)
# TODO: qui i numeri non tornano quindi significa che stessi pazienti hanno avuto più codici amd diversi
# ora vai a capire in ambito medico se significa che hanno più problemi diversi o che hanno avuto diverse diagnosi,
# che la malattia progredisce e quindi cambia codice amd, bho
# provo a capire quali sono i pazienti che hanno avuto più codici amd diversi:
print(
    "numero pazienti con più codici amd diversi: ",
    len(
        aa_prob_cuore[aa_prob_cuore[["idana", "idcentro"]].duplicated(keep=False)][
            ["idana", "idcentro"]
        ].drop_duplicates()
    ),
)

# print(
#     "pazienti con più codici amd diversi: ",
#     aa_prob_cuore[aa_prob_cuore[["idana", "idcentro"]].duplicated(keep=False)][["idana", "idcentro"]].drop_duplicates(),
# )

print(
    "numero pazienti con un unico codice amd: ",
    len(
        aa_prob_cuore[~aa_prob_cuore[["idana", "idcentro"]].duplicated(keep=False)][
            ["idana", "idcentro"]
        ].drop_duplicates()
    ),
)

# TODO: qui si potrebbe calcolare anche qual'è la percentuale in base al sesso e casomai anche per età

# TODO: qui si potrebbe pensare di controllare se l'anno di nascita è uguale all' anno decesso e la data (del controllo?)
# è maggiore dell'anno primo accesso e di diagnosi del diabete di settare a nan l'anno di decesso in modo da non dover
# eliminare quei dati (però chi ti dice che è il decesso l'errore e non le visite?)

# print(
#     "righe da eliminare: ",
#     aa_prob_cuore[aa_prob_cuore["annoprimoaccesso"] > aa_prob_cuore["annodecesso"]],
# )
aa_prob_cuore = aa_prob_cuore.drop(
    aa_prob_cuore[
        aa_prob_cuore["annoprimoaccesso"] > aa_prob_cuore["annodecesso"]
    ].index,
)
print(len(aa_prob_cuore[["idana", "idcentro"]].drop_duplicates()))

# TODO: qui si potrebbe pensare di controllare se l'anno di nascita è uguale all' anno decesso e la data (del controllo?)
# è maggiore dell'anno primo accesso e di diagnosi del diabete di settare a nan l'anno di decesso in modo da non dover
# eliminare quei dati (però chi ti dice che è il decesso l'errore e non le visite?)

# print(
#     "righe da eliminare: ",
#     aa_prob_cuore[aa_prob_cuore["annodiagnosidiabete"] > aa_prob_cuore["annodecesso"]],
# )
aa_prob_cuore = aa_prob_cuore.drop(
    aa_prob_cuore[
        aa_prob_cuore["annodiagnosidiabete"] > aa_prob_cuore["annodecesso"]
    ].index,
)  # già fatto sopra? ops

print("dopo scarto :")
print(len(aa_prob_cuore[["idana", "idcentro"]].drop_duplicates()))

# siccome più della metà dei pazienti che hanno problemi al cuore
# hanno l'anno di diagnosi di diabete minore dell'anno di primo accesso
# noi abbiamo deciso di fillare l'annodiagnosidiabete con l'anno primo accesso
print(
    "numero pazienti unici con anno diagnosi diabete minore dell'anno primo accesso: ",
    len(
        aa_prob_cuore[
            aa_prob_cuore["annodiagnosidiabete"] < aa_prob_cuore["annoprimoaccesso"]
        ][["idana", "idcentro"]].drop_duplicates()
    ),
)  # 27592

print(
    "numero pazienti unici con anno diagnosi diabete maggiore dell'anno primo accesso: ",
    len(
        aa_prob_cuore[
            aa_prob_cuore["annodiagnosidiabete"] >= aa_prob_cuore["annoprimoaccesso"]
        ][["idana", "idcentro"]].drop_duplicates()
    ),
)  # 15426

# aa_prob_cuore.loc[
#     aa_prob_cuore["annodiagnosidiabete"].isna()
#     & aa_prob_cuore["annoprimoaccesso"].notnull(),
#     "annodiagnosidiabete",
# ] = aa_prob_cuore.loc[
#     aa_prob_cuore["annodiagnosidiabete"].isna()
#     & aa_prob_cuore["annoprimoaccesso"].notnull(),
#     "annoprimoaccesso",
# ]  # già fatto sopra? ops

print("All filtered :")
aa_prob_cuore = aa_prob_cuore.dropna(subset="annodiagnosidiabete")
print(len(aa_prob_cuore[["idana", "idcentro"]].drop_duplicates()))  # 49829

### Punto 2 per dataset diversi da anagrafica attivi e diagnosi

## Carica i dataset
print("############## LOADING DATASETS ##############")

df_esami_par = df_list["esamilaboratorioparametri"].result()
print("loaded esami parametri")

df_esami_par_cal = df_list["esamilaboratorioparametricalcolati"].result()
print("loaded esami parametri calcolati")

df_esami_stru = df_list["esamistrumentali"].result()
print("loaded esami strumentali")

df_diagnosi = df_list["diagnosi"].result()
print("loaded diagnosi")

df_prescrizioni_diabete_farmaci = df_list["prescrizionidiabetefarmaci"].result()
print("loaded prescrizioni diabete farmaci")

df_prescrizioni_diabete_non_farmaci = df_list["prescrizionidiabetenonfarmaci"].result()
print("loaded prescrizioni diabete non farmaci")

df_prescirizioni_non_diabete = df_list["prescrizioninondiabete"].result()
print("loaded prescrizioni non diabete")
del df_list

## Calcola le chiavi dei pazienti di interesse
aa_cuore_dates = aa_prob_cuore[
    [
        "idana",
        "idcentro",
        "annonascita",
        "annoprimoaccesso",
        "annodecesso",
    ]
].drop_duplicates()
print(len(aa_cuore_dates))
# print(aa_prob_cuore.head())
# print(aa_prob_cuore.info())


## Cast string to datatime
def cast_to_datetime(df, col, format="%Y-%m-%d"):
    df[col] = pd.to_datetime(df[col], format=format)
    return df[col]


## Rimuovi pazienti non di interesse
print("############## FILTERING DATASETS ##############")

list_of_df = [
    df_esami_par,
    df_esami_par_cal,
    df_esami_stru,
    df_diagnosi,
    df_prescrizioni_diabete_farmaci,
    df_prescrizioni_diabete_non_farmaci,
    df_prescirizioni_non_diabete,
]

## Cast string to datetime
for df in list_of_df:
    df["data"] = cast_to_datetime(df, "data", format="%Y-%m-%d")

# print(df_esami_par.head())
# print(df_esami_par.info())


def clean_between_dates(df, col="data", col_start="annonascita", col_end="annodecesso"):
    # this create a temporary df with only the patients of interest
    df1 = df.merge(aa_cuore_dates, on=["idana", "idcentro"], how="inner")
    print(
        "numero pazienti: ",
        len(df1[["idana", "idcentro"]].drop_duplicates()),
    )
    # non conosco la data in cui il dataset è stato samplato quindi ho usato il timestamp corrente(adesso) come workaround.
    df1 = df1[
        (df1[col] >= df1[col_start])
        & (df1[col] <= df1[col_end].fillna(pd.Timestamp.now()))
    ]
    # this ensure that the columns of patients of interest are the same as the original
    # df filtered only by the keys of the processed patients
    df = df.merge(
        df1[["idana", "idcentro"]].drop_duplicates(), on=["idana", "idcentro"]
    )
    return df


# here diagnosi have 456 rows with data nan so we drop them because we can't know when they were diagnosed
df_diagnosi = clean_between_dates(df_diagnosi).dropna(subset=["data"])

df_esami_par = clean_between_dates(df_esami_par)

df_esami_par_cal = clean_between_dates(df_esami_par_cal)

df_esami_stru = clean_between_dates(df_esami_stru)

df_prescrizioni_diabete_farmaci = clean_between_dates(df_prescrizioni_diabete_farmaci)

df_prescrizioni_diabete_non_farmaci = clean_between_dates(
    df_prescrizioni_diabete_non_farmaci
)

df_prescirizioni_non_diabete = clean_between_dates(df_prescirizioni_non_diabete)

print("diagnosi:\n", df_diagnosi.isna().sum())

del list_of_df, aa_cuore_dates

print("Pulite le date per il punto 2")
### Punto 3
## Append datasets
print("############## POINT 3 START ##############")
# print("diagnosi: ", df_diagnosi.isna().sum())
# print("esami par: ", df_esami_par.isna().sum())
# print("esami par cal: ", df_esami_par_cal.isna().sum())
# print("esami stru: ", df_esami_stru.isna().sum())

# TODO: verify if the concat is correct as the same as merge, and also if is the best way to do this
# TODO: here we must use also prescrizioni? or only esami and diagnosi? I think for me has more sense to only look
#  at esami and diagnosi (and also the specific tal about examinations and diagnosis and not about prescrizioni),
#  but I'm not 100% sure
df_diagnosi_and_esami = pd.concat(
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
).reset_index()  # 49768

print(len(df_diagnosi_and_esami[["idana", "idcentro"]].drop_duplicates()))

print(
    "numero pazienti di interesse inizio punto 3: ",
    len(df_diagnosi_and_esami[["idana", "idcentro"]].drop_duplicates()),
)
# print(df_diagnosi_and_esami.head())
# print(df_diagnosi_and_esami.info())

groups_diagnosi_and_esami = df_diagnosi_and_esami.groupby(["idana", "idcentro"]).agg(
    {"data": ["min", "max"]}
)


groups_diagnosi_and_esami["data_min"] = groups_diagnosi_and_esami["data"]["min"]
groups_diagnosi_and_esami["data_max"] = groups_diagnosi_and_esami["data"]["max"]
# print(groups_diagnosi_and_esami.head(30))
print("groups_diagnosi_and_esami")

groups_diagnosi_and_esami["diff"] = (
    groups_diagnosi_and_esami["data_max"] - groups_diagnosi_and_esami["data_min"]
)

# print(groups_diagnosi_and_esami.head(30))
print(
    "numero di pazienti con tutte le date in un unico giorno: ",
    len(
        groups_diagnosi_and_esami[
            groups_diagnosi_and_esami["diff"] == pd.Timedelta("0 days")
        ]
    ),
)
print(
    "numero di pazienti con tutte le date in un unico mese: ",
    len(
        groups_diagnosi_and_esami[
            groups_diagnosi_and_esami["diff"] < pd.Timedelta("31 days")
        ]
    ),
)

groups_diagnosi_and_esami = groups_diagnosi_and_esami[
    groups_diagnosi_and_esami["diff"] >= pd.Timedelta("31 days")
]
groups_diagnosi_and_esami = groups_diagnosi_and_esami.sort_values(by=["diff"])
print("paziente traiettoria minima: ", groups_diagnosi_and_esami.head(2))
print("paziente traiettoria massima: ", groups_diagnosi_and_esami.tail(4))
print(
    "numero pazienti fine punto 3: ",
    len(groups_diagnosi_and_esami),
)

# print(groups_diagnosi_and_esami)
# print(groups_diagnosi_and_esami.info())
# select only idana and idcentro from groups_diagnosi_and_esami

groups_diagnosi_and_esami_keys = (
    groups_diagnosi_and_esami.stack()
    .reset_index()[["idana", "idcentro"]]
    .drop_duplicates()
)

del groups_diagnosi_and_esami
# print(groups_diagnosi_and_esami_keys.head())
# print(groups_diagnosi_and_esami_keys.info())
print(len(groups_diagnosi_and_esami_keys))

### Punto 4
print("############## POINT 4 START ##############")

# wanted_amd_par = ["AMD004", "AMD005", "AMD006", "AMD007", "AMD008", "AMD009", "AMD111"]
# wanted_stitch_par = ["STITCH001", "STITCH002", "STITCH003", "STITCH004", "STITCH005"]

print("prima update: ")
amd004 = df_esami_par[df_esami_par["codiceamd"] == "AMD004"]["valore"]
print("numero AMD004 minori di 40: ", len(amd004[amd004.astype(float) < 40]))
print("numero AMD004 maggiori di 200: ", len(amd004[amd004.astype(float) > 200]))

# df_esami_par_copy = df_esami_par.copy()
mask = df_esami_par["codiceamd"] == "AMD004"
df_esami_par.loc[mask, "valore"] = df_esami_par.loc[mask, "valore"].clip(40, 200)
# would like to use this single line but from documentation it seems that it can cause problems
# so we must use this in two lines with a precomputation of a mask
# df_esami_par["valore"].update(
#     df_esami_par[df_esami_par["codiceamd"] == "AMD004"]["valore"].clip(40, 200)
# )

mask = df_esami_par["codiceamd"] == "AMD005"
df_esami_par.loc[mask, "valore"] = df_esami_par.loc[mask, "valore"].clip(40, 130)

mask = df_esami_par["codiceamd"] == "AMD007"
df_esami_par.loc[mask, "valore"] = df_esami_par.loc[mask, "valore"].clip(50, 500)

mask = df_esami_par["codiceamd"] == "AMD008"
df_esami_par.loc[mask, "valore"] = df_esami_par.loc[mask, "valore"].clip(5, 15)

print("dopo update: ")
amd004_dopo = df_esami_par[df_esami_par["codiceamd"] == "AMD004"]["valore"]

print("numero AMD004 minori di 40 dopo filtro: ", len(amd004_dopo[amd004_dopo < 40]))
print(
    "numero AMD004 maggiori di 200 dopo filtro: ",
    len(amd004_dopo[amd004_dopo.astype(float) > 200]),
)

print("prima update: ")

stitch002 = df_esami_par_cal[df_esami_par_cal["codicestitch"] == "STITCH002"]["valore"]
print("numero STITCH001 minori di 30: ", len(stitch002[stitch002.astype(float) < 30]))
print(
    "numero STITCH001 maggiori di 300: ", len(stitch002[stitch002.astype(float) > 300])
)

mask = df_esami_par_cal["codicestitch"] == "STITCH002"
df_esami_par_cal.loc[mask, "valore"] = df_esami_par_cal.loc[mask, "valore"].clip(
    30, 300
)

mask = df_esami_par_cal["codicestitch"] == "STITCH003"
df_esami_par_cal.loc[mask, "valore"] = df_esami_par_cal.loc[mask, "valore"].clip(
    60, 330
)

stitch002_dopo = df_esami_par_cal[df_esami_par_cal["codicestitch"] == "STITCH002"][
    "valore"
]

print("dopo update: ")
print(
    "numero STITCH001 minori di 30 dopo filtro: ",
    len(stitch002_dopo[stitch002_dopo < 30]),
)
print(
    "numero STITCH001 maggiori di 300 dopo filtro: ",
    len(stitch002_dopo[stitch002_dopo.astype(float) > 300]),
)

### Punto 5
print("############## POINT 5 START ##############")

aa_prob_cuore_filtered_keys = (
    aa_prob_cuore[["idana", "idcentro"]]
    .drop_duplicates()
    .merge(
        groups_diagnosi_and_esami_keys,
        on=["idana", "idcentro"],
        how="inner",
    )
)

del groups_diagnosi_and_esami_keys

print(
    "numero pazienti inizio punto 5: ",
    len(aa_prob_cuore_filtered_keys[["idana", "idcentro"]].drop_duplicates()),
)

df_diagnosi_and_esami = df_diagnosi_and_esami.merge(
    aa_prob_cuore_filtered_keys,
    on=["idana", "idcentro"],
    how="inner",
)
print("df_diagnosi_and_esami merged")

# TODO: a questo punto dato che per il punto 3 non abbiamo usato le prescrizioni, non dobbiamo usarle nemmeno qui le prescrizioni
#  in quanto no ritengo che siano eventi significativi, quindi qui vanno rivisti i filtri
# df_prescrizioni_diabete_farmaci = df_prescrizioni_diabete_farmaci.merge(
#     aa_prob_cuore_filtered_keys,
#     on=["idana", "idcentro"],
#     how="inner",
# )
# print("df_prescrizioni_diabete_farmaci merged")
# df_prescirizioni_non_diabete = df_prescirizioni_non_diabete.merge(
#     aa_prob_cuore_filtered_keys,
#     on=["idana", "idcentro"],
#     how="inner",
# )
# print("df_prescirizioni_non_diabete merged")
# df_prescrizioni_diabete_non_farmaci = df_prescrizioni_diabete_non_farmaci.merge(
#     aa_prob_cuore_filtered_keys,
#     on=["idana", "idcentro"],
#     how="inner",
# )
# print("df_prescrizioni_diabete_non_farmaci merged")

# df_diagnosi_and_esami_and_prescrioni = pd.concat(
#     objs=(
#         idf.set_index(["idana", "idcentro"])
#         for idf in [
#             df_diagnosi_and_esami[["idcentro", "idana", "data"]],
#             df_prescrizioni_diabete_farmaci[["idcentro", "idana", "data"]],
#             df_prescirizioni_non_diabete[["idcentro", "idana", "data"]],
#             df_prescrizioni_diabete_non_farmaci[["idcentro", "idana", "data"]],
#         ]
#     ),
#     join="inner",
# ).reset_index()

print("df_diagnosi_and_esami_and_prescrioni concatenated")
cont = (
    # df_diagnosi_and_esami_and_prescrioni[["idana", "idcentro"]]
    df_diagnosi_and_esami[["idana", "idcentro"]]
    .groupby(["idana", "idcentro"])
    .size()
    .reset_index(name="count")
)

print("paziente con minimo numero eventi", cont.sort_values(by=["count"]).head(1))
print("paziente con massimo numero eventi", cont.sort_values(by=["count"]).tail(1))
print("cont grouped")
cont_filtered = cont[cont["count"] >= 2]

select_all_events = df_diagnosi_and_esami.merge(
    # df_diagnosi_and_esami_and_prescrioni.merge(
    cont_filtered.reset_index()[["idana", "idcentro"]],
    on=["idana", "idcentro"],
    how="inner",
)

# print(select_all_events)
# select_all_events["data"] = pd.to_datetime(select_all_events["data"], format="%Y-%m-%d")

last_event = select_all_events.groupby(["idana", "idcentro"], group_keys=True)[
    "data"
].max()

# print("last event:\n", last_event)
# print(last_event.info())

print(
    "num pazienti in all_events: ",
    len(select_all_events[["idana", "idcentro"]].drop_duplicates()),
)

print(
    "df_problemi_cuore: ",
    len(aa_prob_cuore_filtered_keys[["idana", "idcentro"]].drop_duplicates()),
)

aa_prob_cuore_filtered = aa_prob_cuore_filtered_keys.merge(
    aa_prob_cuore[["idana", "idcentro", "data"]],
    on=["idana", "idcentro"],
    how="inner",
)

aa_prob_cuore_filtered["data"] = pd.to_datetime(
    aa_prob_cuore_filtered["data"], format="%Y-%m-%d"
)

last_problem = aa_prob_cuore_filtered.groupby(["idana", "idcentro"], group_keys=True)[
    "data"
].max()

# print("last problem:\n", last_problem)
# print(last_problem.info())

# this only to empty memory and make work the other code on laptop
del (
    aa_prob_cuore_filtered,
    cont,
    cont_filtered,
    # df_diagnosi_and_esami_and_prescrioni,
    aa_prob_cuore_filtered_keys,
    df_diagnosi_and_esami,
)

wanted_patient = select_all_events.join(
    (last_problem.ge(last_event - pd.DateOffset(months=6))).rename("label"),
    on=["idana", "idcentro"],
)

del last_problem, select_all_events, last_event

# TODO: delete wanted_patient with trajectory less than 6 months
# I don't understand here someting fishy is going on because I see only 127 days of difference
# between min and max date but the specification says 6 months
wanted_patient_6_months = wanted_patient.groupby(["idana", "idcentro"]).agg(
    {"data": ["min", "max"]}
)


wanted_patient_6_months["data_min"] = wanted_patient_6_months["data"]["min"]
wanted_patient_6_months["data_max"] = wanted_patient_6_months["data"]["max"]

wanted_patient_6_months["diff"] = (
    wanted_patient_6_months["data_max"] - wanted_patient_6_months["data_min"]
)

wanted_patient_6_months = wanted_patient_6_months[
    wanted_patient_6_months["diff"] >= pd.Timedelta("183 days")
]
wanted_patient_6_months = wanted_patient_6_months.sort_values(by=["diff"])
print("paziente traiettoria minima: ", wanted_patient_6_months.head(1))
print("paziente traiettoria massima: ", wanted_patient_6_months.tail(1))

wanted_patient_6_months_keys = (
    wanted_patient_6_months.stack()
    .reset_index()[["idana", "idcentro"]]
    .drop_duplicates()
)

print("RISULATI PUNTO 1.5")
# print(wanted_patient[["idana", "idcentro", "data", "label"]])
wanted_patient = wanted_patient.merge(
    wanted_patient_6_months_keys,
    on=["idana", "idcentro"],
    how="inner",
)

wanted_patient_keys = wanted_patient[["idana", "idcentro"]].drop_duplicates()

print(
    "pazienti fine punto 5: ",
    len(wanted_patient_keys),
)
wanted_patient1 = wanted_patient[wanted_patient["label"] == True]
unwanted_patient = wanted_patient[wanted_patient["label"] == False]
# print(wanted_patient1)
print("True rows patients: ", len(wanted_patient1))
print("False rows patients: ", len(unwanted_patient))
print("True patients: ", len(wanted_patient1[["idana", "idcentro"]].drop_duplicates()))
print(
    "False patients: ", len(unwanted_patient[["idana", "idcentro"]].drop_duplicates())
)

### TODO: Punto 6
# some things for point 6 are done in point 2 and 3 to speed up computations
print("############## POINT 6 START ##############")

# TODO: controllare anagrafica e diagnosi
print("patients labels: ")
print(wanted_patient.isna().sum())
# qui tutto ok

print("anagrafica: ")
df_anagrafica_attivi = df_anagrafica_attivi.merge(
    wanted_patient_keys, on=["idana", "idcentro"], how="inner"
)
print(df_anagrafica_attivi.isna().sum())
# qui ci sono 350 righe con annodiagnosi diabete a nan, 6,5k righe con annoprimoaccesso a nan
# e le informazioni demografiche sono spesso mancanti ma possono essere tenute usando un [UNK] in seguito
# la colonna origine è quasi sempre a nan quindi non è utile e andrebbe eliminata

print("diagnosi: ")
df_diagnosi = df_diagnosi.merge(
    wanted_patient_keys, on=["idana", "idcentro"], how="inner"
)
print(df_diagnosi.isna().sum())
# qui ci sono 33k righe con valore a nan
# non sono interessanti poichè non sono tra le diagnosi per noi interessanti,
# quindi non vanno riempite ma le terrei per avere comunque altre informazioni sui pazienti
df_diagnosi_nan = (
    df_diagnosi[df_diagnosi["valore"].isna()]
    .groupby(["codiceamd"])
    .size()
    .sort_values(ascending=False)
)
print(df_diagnosi_nan)
# TODO: qui dai grafici in plot si capisce che c'è molto da pulire sulla colonna valore che spesso
# è sparca presentando lo stesso valore ma con numeri interi/decimali con diverse cifre.

print("esami lab parametri: ")
df_esami_par = df_esami_par.merge(
    wanted_patient_keys, on=["idana", "idcentro"], how="inner"
)

print(df_esami_par.isna().sum())
# qui ci sono 30k righe con valore a nan
df_esami_par_nan = (
    df_esami_par[df_esami_par["valore"].isna()]
    .groupby(["codiceamd"])
    .size()
    .sort_values(ascending=False)
)
print(df_esami_par_nan)
# i seguenti codice amd hanno i rispettivi nan:
# codiceamd
# AMD009    28776
# AMD001     1599
# TODO qui si possono fillare i 1,6k valori nan di amd001 con il valore medio in quanto rappresenta l'altezza

print("esami lab parametri calcolati: ")
df_esami_par_cal = df_esami_par_cal.merge(
    wanted_patient_keys, on=["idana", "idcentro"], how="inner"
)
print(df_esami_par_cal.isna().sum())
# qui ci sono 900k righe con codiceamd nan

print(df_esami_par_cal.groupby(["codiceamd"]).size())
print(df_esami_par_cal.groupby(["codicestitch"]).size())
# una parte dei codiciamd mancanti possono essere fillati in base al valore del codice stitch
# quindi va fatta un analisi raggruppando per codice stitch e poi per codice amd in modo da
# vedere quali sono le caratterisitche per il fill dei codici amd mancanti
# qui sotto si vede che gli 900k codici amd mancanti hanno tutti codici stitch 003 e 004
print(
    df_esami_par_cal[df_esami_par_cal["codiceamd"].isna()]["codicestitch"]
    .isin(["STITCH003", "STITCH004"])
    .sum()
)

# raggruppa per codice stitch e poi per codice amd
# da qui si vede proprio che i codici stitch e gli amd sono legati da:
# codicestitch  codiceamd
# STITCH001     AMD927       969012
# STITCH002     AMD013       342914
# STITCH005     AMD304       525186
# quindi non è possibile fare un fill dei codici amd mancanti in base al codice stitch
# poiche non ci sono relazioni tra gli stich 003 e 004 e gli amd
# praticamente gli stitch 003 e 004 sono l'unica informazione utilizzabile piuttosto di amd e stitch insieme
print(df_esami_par_cal.groupby(["codicestitch", "codiceamd"]).size())


df_esami_stru = df_esami_stru.merge(
    wanted_patient_keys, on=["idana", "idcentro"], how="inner"
)
print("esami strumentali: ")
print(df_esami_stru.isna().sum())
# qui ci sono 21k righe con valore a nan

print(df_esami_stru.groupby(["codiceamd"]).size())
# alcuni codici amd sono presenti in proporzioni molto maggiori rispetto ad altri

# ragruppando i codici amd per quantità di nan in valore
# si vede che i codici amd con valori nan sono solo amd125 con 21k righe e amd126 solo 4
# quindi si potrebbe fare un fill dei valori nan in base al valore più presente nel caso del codice amd126 che è N
# mentre nel caso del codice amd125 non si può fare un fill in quanto si tratta di fillare metà delle righe e quindi
# potrebbe portare pi problemi che benefici
df_esami_stru_nan = (
    df_esami_stru[df_esami_stru["valore"].isna()]
    .groupby(["codiceamd"])
    .size()
    .sort_values(ascending=False)
)
print(df_esami_stru_nan)

print("prescrizioni diabete farmaci: ")
df_prescrizioni_diabete_farmaci = df_prescrizioni_diabete_farmaci.merge(
    wanted_patient_keys, on=["idana", "idcentro"], how="inner"
)
print(df_prescrizioni_diabete_farmaci.isna().sum())
# qui ci sono 38 righe con codice atc nan da gestire
print(df_prescrizioni_diabete_farmaci.groupby(["codiceatc"]).size())
# print(
#     df_prescrizioni_diabete_farmaci.groupby(["codiceatc", "descrizionefarmaco"]).size()
# )
df_prescrizioni_diabete_farmaci_nan = (
    df_prescrizioni_diabete_farmaci[df_prescrizioni_diabete_farmaci["codiceatc"].isna()]
    .groupby(["descrizionefarmaco"])
    .size()
    .sort_values(ascending=False)
)
print(df_prescrizioni_diabete_farmaci_nan)

print("prescrizioni diabete non farmaci: ")
df_prescrizioni_diabete_non_farmaci = df_prescrizioni_diabete_non_farmaci.merge(
    wanted_patient_keys, on=["idana", "idcentro"], how="inner"
)

print(df_prescrizioni_diabete_non_farmaci.isna().sum())
# qui ci sono 15k righe con valore nan
print(df_prescrizioni_diabete_non_farmaci.groupby(["codiceamd"]).size())
# qui abbiamo un codice amd096 che è presente in sole 32 righe e quindi completamente
# sbilanciato rispetto agli altri codici amd presenti in grandi quantità, quindi lo scarterei
# poi due codici amd086 e amd152 riportano la stessa descrizione e quindi li unirei in un unico codice,
# l'unico problema è che a quel punto la maggioranza dei codici sarebbero l'unione di questo 120k
# e i rimanenti 25k sarebbero altri due codici, quindi non so se sia il caso di unirli

# dal seguente si vede che gli unici amd con valori nan sono amd096 e amd152,
# quindi si potrebbe fare un fill dei valori nan in base al valore più presente nel caso del codice amd152 prendendoli
# dal codice amd086 che riporta la stessa descrizione e quindi valutare se si potrebbe unire amd086 e amd152 in un unico codice
# mentre anche per questo motivo scarterei amd096
df_prescrizioni_diabete_non_farmaci_nan = (
    df_prescrizioni_diabete_non_farmaci[
        df_prescrizioni_diabete_non_farmaci["valore"].isna()
    ]
    .groupby(["codiceamd"])
    .size()
    .sort_values(ascending=False)
)
print(df_prescrizioni_diabete_non_farmaci_nan)

drop_mask = (
    df_prescrizioni_diabete_non_farmaci["codiceamd"] == "AMD096"
) & df_prescrizioni_diabete_non_farmaci["valore"].isna()

df_prescrizioni_diabete_non_farmaci = df_prescrizioni_diabete_non_farmaci.drop(
    df_prescrizioni_diabete_non_farmaci[drop_mask].index
)

print("prescrizioni non diabete: ")
print("no nan")
# qui non ci sono nan
df_prescirizioni_non_diabete = df_prescirizioni_non_diabete.merge(
    wanted_patient_keys, on=["idana", "idcentro"], how="inner"
)

# print(df_prescirizioni_non_diabete.isna().sum())
