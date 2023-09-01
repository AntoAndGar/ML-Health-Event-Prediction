import concurrent.futures
import multiprocessing
import pandas as pd

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

# Read all the dataset concurrently and store them in a dictionary with the name of the file as key
with concurrent.futures.ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
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
df_pres_diab_farm = df_pres_diab_farm.drop(columns="data")
df_pres_diab_no_farm = df_pres_diab_no_farm.drop(columns="data")
df_pres_no_diab = df_pres_no_diab.drop(colums="data")
