
from app.base import imports

class ConvertHelpers:

    def __init__(self):
        pass

    def convert_to_categorical(self, dataframe,column_name):
        column = imports.pd.Categorical(dataframe[column_name])
        
        #remove old column_name column
        dataframe.drop(columns=[column_name])

        #create new column_name column
        dataframe[column_name] = column

        return dataframe