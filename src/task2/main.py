import pandas as pd
import datetime as dt

read_data_path = "clean_data"
print("Loading data...")
### Load dataset parsing datatime to datetime64[ns] ###
df_anagrafica = pd.read_csv(
    read_data_path + "/anagraficapazientiattivi_c.csv",
    header=0,
    # names=[
    #     "idcentro",
    #     "idana",
    #     "sesso",
    #     "annodiagnosidiabete",
    #     "annonascita",
    #     "annoprimoaccesso",
    #     "annodecesso",
    #     "label",
    # ],
    parse_dates=["annonascita", "annoprimoaccesso", "annodecesso"],
)
df_diagnosi = pd.read_csv(
    read_data_path + "/diagnosi_c.csv",
    header=0,
    # names=["idcentro", "idana", "data", "codiceamd", "valore"],
    parse_dates=["data"],
)
df_esami_par = pd.read_csv(
    read_data_path + "/esamilaboratorioparametri_c.csv",
    header=0,
    # names=["idcentro", "idana", "data", "codiceamd", "valore"],
    parse_dates=["data"],
)
df_esami_par_cal = pd.read_csv(
    read_data_path + "/esamilaboratorioparametricalcolati_c.csv",
    header=0,
    # names=["idcentro", "idana", "data", "codiceamd", "valore", "codicestitch"],
    parse_dates=["data"],
)
df_esami_stru = pd.read_csv(
    read_data_path + "/esamistrumentali_c.csv",
    header=0,
    names=["idcentro", "idana", "data", "codiceamd", "valore"],
    parse_dates=["data"],
)
df_pre_diab_farm = pd.read_csv(
    read_data_path + "/prescrizionidiabetefarmaci_c.csv",
    header=0,
    # names=[
    #     "idcentro",
    #     "idana",
    #     "data",
    #     "codiceatc",
    #     "quantita",
    #     "idpasto",
    #     "descrizionefarmaco",
    # ],
    parse_dates=["data"],
)
df_pre_diab_no_farm = pd.read_csv(
    read_data_path + "/prescrizionidiabetenonfarmaci_c.csv",
    header=0,
    # names=["idcentro", "idana", "data", "codiceamd", "valore"],
    parse_dates=["data"],
)
df_pre_no_diab = pd.read_csv(
    read_data_path + "/prescrizioninondiabete_c.csv",
    header=0,
    # names=["idcentro", "idana", "data", "codiceamd", "valore"],
    parse_dates=["data"],
)

### Point 2.1 ####
print("Point 2.1")
print(df_anagrafica.head())
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
    df = df_last_event_label_1[temp]
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


# wanted_patient = select_all_events.join(
#     (last_problem.ge(last_event - pd.DateOffset(months=6))).rename("label"),
#     on=["idana", "idcentro"],
# )


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
