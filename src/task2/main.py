import pandas as pd
import numpy as np
import datetime as dt
import random

read_data_path  = "clean_data"

df_anagrafica   = pd.read_csv(read_data_path+'/anagraficapazientiattivi_c.csv', header=0 ,names=['idana','idcentro','data','label'])
#df_diagnosi = pd.read_csv(read_data_path+'/diagnosi_c.csv', header=0 ,names=['idcentro','idana','data','codiceamd','valore'])
#df_esami_lab_par  = pd.read_csv(read_data_path+'/esamilaboratorioparametri_c.csv', header=0 ,names=['idcentro','idana','data','codiceamd','valore'])
#df_esami_lab_par_calc  = pd.read_csv(read_data_path+'/esamilaboratorioparametricalcolati_c.csv', header=0 ,names=['idcentro','idana','data','codiceamd','valore'])
#df_esami_stru   = pd.read_csv(read_data_path+'/esamistrumentali_c.csv', header=0 ,names=['idcentro','idana','data','codiceamd','valore'])
#df_pre_diab_farm  = pd.read_csv(read_data_path+'/prescrizionidiabetefarmaci_c.csv', header=0 ,names=['idcentro','idana','data','codiceatc','quantita','idpasto','descrizionefarmaco'])
#df_pre_diab_no_farm = pd.read_csv(read_data_path+'/prescrizionidiabetenonfarmaci_c.csv', header=0 ,names=['idcentro','idana','data','codiceamd','valore'])
#df_pre_no_diab  = pd.read_csv(read_data_path+'/prescrizioninondiabete_c.csv', header=0 ,names=['idcentro','idana','data','codiceamd','valore'])

### Point 2.1 ####
print(df_anagrafica.label.value_counts())