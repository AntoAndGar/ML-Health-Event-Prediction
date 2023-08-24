class DatatimeConverter:
    def __init__(self, df_anagrafica, to="float"):
        self.__converted_df_anagrafica = self.convert_anagrafica_data_to_float(df_anagrafica)
        self.df_anno_nascita = df_anagrafica[['idana', 'idcentro', 'annodinascita']]

    def convert_event_datatime_to_float(self, df_to_convert):
       # df_data_nascita = self.df_anno_nascita[['idana', 'idcentro', 'annonascita']]
        df_to_convert = df_to_convert.merge(self.df_anno_nascita, on=['idana', 'idcentro'], how='inner')
        df_to_convert['appo_data'] = df_to_convert['data'] - df_to_convert['annonascita']

        print(df_to_convert.head(15))


    def convert(self, df_to_convert):
        df_to_convert = self.convert_event_datatime_to_float(df_to_convert)
        df_to_convert = self.convert_anagrafica_data_to_float(df_to_convert)
        df_to_convert = self.add_life_length(df_to_convert)

        return df_to_convert
    
    def convert_anagrafica_data_to_float(df_anagrafica):
        #TODO: gestire i valori nulli
        df_anagrafica['annodecesso'] = df_anagrafica['annodcesso'] - df_anagrafica['annodinascita']
        df_anagrafica['annoprimoaccesso'] = df_anagrafica['annoprimoaccesso'] - df_anagrafica['annodinascita']
        df_anagrafica['annodiagnosidiabete'] = df_anagrafica['annodiagnosidiabete'] - df_anagrafica['annodinascita']
        return df_anagrafica
    
    def getConvertedAnagrafica(self):
        return self.__converted_df_anagrafica





def add_life_length(df_anagrafica):
    df_anagrafica['life_length'] = df_anagrafica['annodinascita'] - df_anagrafica['annodecesso']
    return df_anagrafica

