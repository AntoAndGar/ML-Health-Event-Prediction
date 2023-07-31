import pandas as pd
import datetime as dt

read_data_path  = "clean_data"
print("Loading data...")
### Load dataset parsing datatime to datetime64[ns] ###
df_anagrafica   = pd.read_csv(read_data_path+'/anagraficapazientiattivi_c.csv', header=0 ,names=['idcentro','idana','sesso','annodiagnosidiabete','annonascita','annoprimoaccesso','annodecesso','label'], parse_dates=['annonascita', 'annoprimoaccesso', 'annodecesso'])
df_diagnosi = pd.read_csv(read_data_path+'/diagnosi_c.csv', header=0 ,names=['idcentro','idana','data','codiceamd','valore'], parse_dates=['data'])
df_esami_lab_par  = pd.read_csv(read_data_path+'/esamilaboratorioparametri_c.csv', header=0 ,names=['idcentro','idana','data','codiceamd','valore'], parse_dates=['data'])
df_esami_lab_par_calc  = pd.read_csv(read_data_path+'/esamilaboratorioparametricalcolati_c.csv', header=0 ,names=['idcentro','idana','data','codiceamd','valore','codicestitch'], parse_dates=['data'])
df_esami_stru   = pd.read_csv(read_data_path+'/esamistrumentali_c.csv', header=0 ,names=['idcentro','idana','data','codiceamd','valore'], parse_dates=['data'])
df_pre_diab_farm  = pd.read_csv(read_data_path+'/prescrizionidiabetefarmaci_c.csv', header=0 ,names=['idcentro','idana','data','codiceatc','quantita','idpasto','descrizionefarmaco'], parse_dates=['data'])
df_pre_diab_no_farm = pd.read_csv(read_data_path+'/prescrizionidiabetenonfarmaci_c.csv', header=0 ,names=['idcentro','idana','data','codiceamd','valore'], parse_dates=['data'])
df_pre_no_diab  = pd.read_csv(read_data_path+'/prescrizioninondiabete_c.csv', header=0 ,names=['idcentro','idana','data','codiceamd','valore'], parse_dates=['data'])

### Point 2.1 ####
print("Point 2.1")
print(df_anagrafica.head())
print(df_anagrafica.label.value_counts())

df_anagrafica_label_0 = df_anagrafica[df_anagrafica.label == 0]
df_anagrafica_label_1 = df_anagrafica[df_anagrafica.label == 1]

df_diagnosi_label_0 = pd.merge(df_diagnosi, df_anagrafica_label_0, on=['idcentro','idana'])[df_diagnosi.columns]
if False:
    df_esami_lab_par_label_0  = pd.merge(df_esami_lab_par, df_anagrafica_label_0, on=['idcentro','idana'])[df_esami_lab_par.columns]
    df_esami_lab_par_calc_label_0 = pd.merge(df_esami_lab_par_calc, df_anagrafica_label_0, on=['idcentro','idana'])[df_esami_lab_par_calc.columns]
    df_esami_stru_label_0  = pd.merge(df_esami_stru, df_anagrafica_label_0, on=['idcentro','idana'])[df_esami_stru.columns]
    df_pre_diab_farm_label_0 = pd.merge(df_pre_diab_farm, df_anagrafica_label_0, on=['idcentro','idana'])[df_pre_diab_farm.columns]
    df_pre_diab_no_farm_label_0 = pd.merge(df_pre_diab_no_farm, df_anagrafica_label_0, on=['idcentro','idana'])[df_pre_diab_no_farm.columns]
    df_pre_no_diab_label_0  = pd.merge(df_pre_no_diab, df_anagrafica_label_0, on=['idcentro','idana'])[df_pre_no_diab.columns]

    df_diagnosi_label_1 = pd.merge(df_diagnosi, df_anagrafica_label_1, on=['idcentro','idana'])[df_diagnosi.columns]
    df_esami_lab_par_label_1  = pd.merge(df_esami_lab_par, df_anagrafica_label_1, on=['idcentro','idana'])[df_esami_lab_par.columns]
    df_esami_lab_par_calc_label_1 = pd.merge(df_esami_lab_par_calc, df_anagrafica_label_1, on=['idcentro','idana'])[df_esami_lab_par_calc.columns]
    df_esami_stru_label_1  = pd.merge(df_esami_stru, df_anagrafica_label_1, on=['idcentro','idana'])[df_esami_stru.columns]
    df_pre_diab_farm_label_1 = pd.merge(df_pre_diab_farm, df_anagrafica_label_1, on=['idcentro','idana'])[df_pre_diab_farm.columns]
    df_pre_diab_no_farm_label_1 = pd.merge(df_pre_diab_no_farm, df_anagrafica_label_1, on=['idcentro','idana'])[df_pre_diab_no_farm.columns]
    df_pre_no_diab_label_1  = pd.merge(df_pre_no_diab, df_anagrafica_label_1, on=['idcentro','idana'])[df_pre_no_diab.columns]

# Remove events in the last 6 months
## FIXME: this is not working
def dropLastSixMonths(df:pd.DataFrame, label: bool) -> pd.DataFrame:
    if label:
        patients = df_anagrafica_label_1
    else:
        patients = df_anagrafica_label_0
    newDF = pd.DataFrame(columns=df.columns)
    print(df.head())
    newDF = df.groupby(["idana", "idcentro"])
    ei = newDF.agg({"data": ["max"]})
    print("qui")
    print(newDF.head())
    print(ei.head())
    input("heheh")
    df = df['data' < 'max']

    return df

aux = df_diagnosi_label_0.shape[0]
df_diagnosi_label_0 = dropLastSixMonths(df_diagnosi_label_0, True)
print(f"df_diagnosi_label_0: {aux} => {df_diagnosi_label_0.shape[0]}")
exit()
aux = df_esami_lab_par_label_0.shape[0]
df_esami_lab_par_label_0 = dropLastSixMonths(df_esami_lab_par_label_0, True)
print(f"df_esami_lab_par_label_0: {aux} => {df_esami_lab_par_label_0.shape[0]}")

aux = df_esami_lab_par_calc_label_0.shape[0]
df_esami_lab_par_calc_label_0 = dropLastSixMonths(df_esami_lab_par_calc_label_0, True)
print(f"df_esami_lab_par_calc_label_0: {aux} => {df_esami_lab_par_calc_label_0.shape[0]}")

aux = df_esami_stru_label_0.shape[0]
df_esami_stru_label_0 = dropLastSixMonths(df_esami_stru_label_0, True)
print(f"Pdf_class0: {aux} => {df_esami_stru_label_0.shape[0]}")

aux = df_pre_diab_farm_label_0.shape[0]
df_pre_diab_farm_label_0 = dropLastSixMonths(df_pre_diab_farm_label_0, True)
print(f"Pdnf_class0: {aux} => {df_pre_diab_farm_label_0.shape[0]}")

aux = df_pre_diab_no_farm_label_0.shape[0]
df_pre_diab_no_farm_label_0 = dropLastSixMonths(df_pre_diab_no_farm_label_0, True)
print(f"Pnd_class0: {aux} => {df_pre_diab_no_farm_label_0.shape[0]}")

aux = df_pre_no_diab_label_0.shape[0]
df_pre_no_diab_label_0 = dropLastSixMonths(df_pre_no_diab_label_0, True)
print(f"Pnd2_class0: {aux} => {df_pre_no_diab_label_0.shape[0]}")
