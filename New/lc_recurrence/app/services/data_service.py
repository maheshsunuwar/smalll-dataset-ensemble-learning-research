import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from lc_recurrence.app.abstracts.data_abstract import DataAbstract


class DataService(DataAbstract):
    def __init__(self, data_url):
        """ Initiate class with url of the data """
        super().__init__(data_url)
        self.testing = None

    def preprocessing(self):
        self.data = self.data[50:].reset_index(drop=True)

        self.data.drop(columns=['CaseID', 'Patientaffiliation'], inplace=True)

        # dropping not collected recurrence data
        # self.data.drop(index=[48], inplace=True)

        # replace Not Collected with NaN
        self.data.replace(['Not Collected', 'Not collected', 'Not Recorded In Database'], np.nan, inplace=True)
        self.data['Weightlbs'] = pd.to_numeric(self.data['Weightlbs'])

        # replace nan with mean value
        mean_value = self.data['Weightlbs'].mean()
        self.data['Weightlbs'].replace(np.nan, mean_value, inplace=True)

        # convert to categorical
        self.convert_to_category(
            [
                'Gender', 'Smokingstatus', 'Ethnicity', 'GG', 'Histology',
                'PathologicalTstage', 'PathologicalMstage', 'PathologicalNstage',
                'HistopathologicalGrade', 'Lymphovascularinvasion',
                'Pleuralinvasionelasticvisceralorparietal',
                'ALKtranslocationstatus',
                'EGFRmutationstatus', 'KRASmutationstatus',
                # 'Recurrence Location'
            ])

        # replace checked and unchecked with 0 and 1
        self.data.replace(['Checked', 'yes', 'Yes'], 1, inplace=True)
        self.data.replace(['Unchecked', 'no', 'No'], 0, inplace=True)

        # remove columns 'Date of Recurrence','Date of Last Known Alive', 'Survival Status', 'Date of Death',

        self.data.drop(columns=[
            'DateofRecurrence', 'DateofLastKnownAlive',
            'SurvivalStatus', 'DateofDeath', 'TimetoDeathdays',
            'RecurrenceLocation', 'PETDate', 'CTDate', 'QuitSmokingYear'
        ], inplace=True)

        # male to 0, female to 1
        self.data['Gender'] = np.where(self.data['Gender'] == 'Female', 1, 0)

        # for non-smokers, pack years is 0
        self.data['PackYears'] = np.where(self.data['Smokingstatus'] == 'Nonsmoker', 0, self.data['PackYears'])
        self.convert_to_numeric(['PackYears'])
        self.data['PackYears'].replace(np.nan, np.floor(self.data['PackYears'].mean()), inplace=True)

        # # 8 null values in Lympovascular invasion
        # # take rows where Lymphovascular invasion is not null
        # self.data = self.data[self.data['Lymphovascular invasion'].notnull()]
        #
        # # 2 null values in EGFR mutation status
        # # take rows where EGFR mutation status is not null
        # self.data = self.data[self.data['EGFR mutation status'].notnull()]

        # imputing missing values with mode
        self.impute_categorical(['Histology',
                                 'PathologicalTstage',
                                 'PathologicalMstage',
                                 'PathologicalNstage',
                                 'Lymphovascularinvasion',
                                 'ALKtranslocationstatus',
                                 'EGFRmutationstatus',
                                 'KRASmutationstatus'])

        # converting categorical to numeric
        self.encode_categorical_column()

        # # taking random 25 rows
        self.testing = self.data.sample(n=55, random_state=1).reset_index(drop=True)
        #

        # Commenting out the code below to use all the data - without bootstrap sampling
        # # data that does not contain testing
        # self.data = self.data[~self.data.index.isin(self.testing.index)]
        #
        # # # bootstrap sampling to balance the data
        # self.data = self.data.sample(n=300, replace=True, random_state=1)
        # self.data = self.data.reset_index(drop=True)

        # sampling only training data

    # feature selection using pca
    def feature_selection(self):
        '''
            Feature selection using PCA
        '''

        # separating features and target
        x = self.data.drop(columns=['Recurrence'])
        y = self.data['Recurrence']
        print(x)
        # standardizing the features
        x = StandardScaler().fit_transform(x)

        # perform pca
        pca = PCA(n_components=3)

        # fit and transform the data and get dataset with reduced features
        x_pca = pca.fit_transform(x)

        # Get the absolute loadings for each feature in the first principal component
        loadings = np.abs(pca.components_[0])

        # Get the indices of the top two features in the first principal component
        selected_features = np.argsort(loadings)[-20:]
        print(self.data.columns[selected_features])
        # self.data = self.data[self.data.columns[selected_features]]

    # function to convert categorical to numeric using one hot encoding
    def encode_categorical_column(self):
        # encoding categorical data
        encoder = OneHotEncoder(sparse_output=False)

        # loop through categorical columns
        for column in self.data.select_dtypes(include=['category']).columns:

            # fit and transform
            encoded = encoder.fit_transform(self.data[column].values.reshape(-1, 1))
            # convert to dataframe
            encoded = pd.DataFrame(encoded, columns=[column + '_' + str(i) for i in range(encoded.shape[1])])
            encoded = encoded.reset_index(drop=True)
            # # drop original column
            self.data.drop(columns=[column], axis=1, inplace=True)
            # concat with original dataframe
            self.data = pd.concat([self.data, encoded], axis=1)
