import pandas as pd
import datetime as dt
import numpy as np
import concurrent.futures as futures
import multiprocessing

seed = 0
rng = np.random.default_rng(seed)

read_data_path = "clean_data"

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
        df_list[str(name)] = executor.submit(read_csv, f"{read_data_path}/{name}.csv")

print("Loading data...")
### Load dataset and parse dates columns to datetime64[ns] ###
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


for col in ["annonascita", "annoprimoaccesso", "annodecesso"]:
    df_anagrafica[col] = cast_to_datetime(df_anagrafica, col, format="%Y-%m-%d")

## Cast string to datetime
for df in list_of_df:
    df["data"] = cast_to_datetime(df, "data", format="%Y-%m-%d")

### Point 2.1 ####
print("Point 2.1")
# print(df_anagrafica.head())
print(df_anagrafica.label.value_counts())

df_anagrafica_label_0 = df_anagrafica[df_anagrafica.label == 0]
df_anagrafica_label_1 = df_anagrafica[df_anagrafica.label == 1]

# df_diagnosi_label_0 = pd.merge(
#     df_diagnosi, df_anagrafica_label_0, on=["idcentro", "idana"]
# )[df_diagnosi.columns]
# if False:
#     df_esami_lab_par_label_0 = pd.merge(
#         df_esami_lab_par, df_anagrafica_label_0, on=["idcentro", "idana"]
#     )[df_esami_lab_par.columns]
#     df_esami_lab_par_calc_label_0 = pd.merge(
#         df_esami_lab_par_calc, df_anagrafica_label_0, on=["idcentro", "idana"]
#     )[df_esami_lab_par_calc.columns]
#     df_esami_stru_label_0 = pd.merge(
#         df_esami_stru, df_anagrafica_label_0, on=["idcentro", "idana"]
#     )[df_esami_stru.columns]
#     df_pre_diab_farm_label_0 = pd.merge(
#         df_pre_diab_farm, df_anagrafica_label_0, on=["idcentro", "idana"]
#     )[df_pre_diab_farm.columns]
#     df_pre_diab_no_farm_label_0 = pd.merge(
#         df_pre_diab_no_farm, df_anagrafica_label_0, on=["idcentro", "idana"]
#     )[df_pre_diab_no_farm.columns]
#     df_pre_no_diab_label_0 = pd.merge(
#         df_pre_no_diab, df_anagrafica_label_0, on=["idcentro", "idana"]
#     )[df_pre_no_diab.columns]

#     df_diagnosi_label_1 = pd.merge(
#         df_diagnosi, df_anagrafica_label_1, on=["idcentro", "idana"]
#     )[df_diagnosi.columns]
#     df_esami_lab_par_label_1 = pd.merge(
#         df_esami_lab_par, df_anagrafica_label_1, on=["idcentro", "idana"]
#     )[df_esami_lab_par.columns]
#     df_esami_lab_par_calc_label_1 = pd.merge(
#         df_esami_lab_par_calc, df_anagrafica_label_1, on=["idcentro", "idana"]
#     )[df_esami_lab_par_calc.columns]
#     df_esami_stru_label_1 = pd.merge(
#         df_esami_stru, df_anagrafica_label_1, on=["idcentro", "idana"]
#     )[df_esami_stru.columns]
#     df_pre_diab_farm_label_1 = pd.merge(
#         df_pre_diab_farm, df_anagrafica_label_1, on=["idcentro", "idana"]
#     )[df_pre_diab_farm.columns]
#     df_pre_diab_no_farm_label_1 = pd.merge(
#         df_pre_diab_no_farm, df_anagrafica_label_1, on=["idcentro", "idana"]
#     )[df_pre_diab_no_farm.columns]
#     df_pre_no_diab_label_1 = pd.merge(
#         df_pre_no_diab, df_anagrafica_label_1, on=["idcentro", "idana"]
#     )[df_pre_no_diab.columns]

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
    balanced_aa_keys = balanced_aa[["idana", "idcentro"]]
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
    print(
        duplicated_df_anagrafica_label_1.sort_values(by=["idana", "idcentro"]).head(30)
    )

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
        #  in the 6 months after last event, to ensure we are not creating other False . maybe do also here a dropLastSixMonths?

        # the idana is negative for th duplicate to easly distinguish it from the original
        # the 10000 is a number greather than the max value of number of patient in each idcentro
        new_dup_record["idana"] = -(
            new_dup_record["idana"].astype("int")
            + 10000 * new_dup_record["duplicate_identifier"].astype("int")
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
        + 10000 * new_dup_record["duplicate_identifier"].astype("int")
    )
    new_dup_record["label"] = False
    new_dup_record = new_dup_record.drop(["duplicate_identifier", "duplicated"], axis=1)
    df_anagrafica = pd.concat([df_anagrafica, new_dup_record], ignore_index=True)
    print("After balance: ", len(df_anagrafica))
    print(df_anagrafica.label.value_counts())

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


# # Remove events in the last 6 months
# ## FIXME: this is not working
# def dropLastSixMonths(df: pd.DataFrame, label: bool) -> pd.DataFrame:
#     if label:
#         patients = df_anagrafica_label_1
#     else:
#         patients = df_anagrafica_label_0
#     newDF = pd.DataFrame(columns=df.columns)
#     print(df.head())
#     newDF = df.groupby(["idana", "idcentro"])
#     ei = newDF.agg({"data": ["max"]})
#     print("qui")
#     print(newDF.head())
#     print(ei.head())
#     input("heheh")
#     df = df["data" < "max"]

#     return df


# aux = df_diagnosi_label_0.shape[0]
# df_diagnosi_label_0 = dropLastSixMonths(df_diagnosi_label_0, True)
# print(f"df_diagnosi_label_0: {aux} => {df_diagnosi_label_0.shape[0]}")
# exit()
# aux = df_esami_lab_par_label_0.shape[0]
# df_esami_lab_par_label_0 = dropLastSixMonths(df_esami_lab_par_label_0, True)
# print(f"df_esami_lab_par_label_0: {aux} => {df_esami_lab_par_label_0.shape[0]}")

# aux = df_esami_lab_par_calc_label_0.shape[0]
# df_esami_lab_par_calc_label_0 = dropLastSixMonths(df_esami_lab_par_calc_label_0, True)
# print(
#     f"df_esami_lab_par_calc_label_0: {aux} => {df_esami_lab_par_calc_label_0.shape[0]}"
# )

# aux = df_esami_stru_label_0.shape[0]
# df_esami_stru_label_0 = dropLastSixMonths(df_esami_stru_label_0, True)
# print(f"Pdf_class0: {aux} => {df_esami_stru_label_0.shape[0]}")

# aux = df_pre_diab_farm_label_0.shape[0]
# df_pre_diab_farm_label_0 = dropLastSixMonths(df_pre_diab_farm_label_0, True)
# print(f"Pdnf_class0: {aux} => {df_pre_diab_farm_label_0.shape[0]}")

# aux = df_pre_diab_no_farm_label_0.shape[0]
# df_pre_diab_no_farm_label_0 = dropLastSixMonths(df_pre_diab_no_farm_label_0, True)
# print(f"Pnd_class0: {aux} => {df_pre_diab_no_farm_label_0.shape[0]}")

# aux = df_pre_no_diab_label_0.shape[0]
# df_pre_no_diab_label_0 = dropLastSixMonths(df_pre_no_diab_label_0, True)
# print(f"Pnd2_class0: {aux} => {df_pre_no_diab_label_0.shape[0]}")
