from abc import ABC, abstractmethod

from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import re


class DataAbstract(ABC):

    def __init__(self, data_url):
        """
        Initiate class with url of the data
        :param data_url: url of the data
        """
        self.data = pd.read_csv(data_url, delimiter=',').rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None

    def view(self, row_numbers):
        """
        Shows top 5 rows of the data if row_numbers is 5
        :param row_numbers: numbers of row to show
        :return:
        """
        return self.data.head(row_numbers)

    def get_columns(self):
        """
        :return: list of column names
        """
        return self.data.columns

    def get_train_test_data(self, selected_columns=None):
        """
        make sure (in the data) the last column is the feature you want to predict/classify
        :selected_columns: pass a list of columns to consider for the training process
        :return: X_train, X_test, y_train, y_test
        """
        x_columns, y_column = self.get_x_y_columns(selected_columns)

        x = self.data[x_columns]
        y = self.data[y_column]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

        return self.X_train, self.X_test, self.y_train, self.y_test

    def get_x_y_columns(self, selected_columns):
        columns = self.get_columns()

        y_column = 'Recurrence'
        # all columns except recurrence
        x_columns = [col for col in columns if col != 'Recurrence']

        return x_columns, y_column

    def convert_to_category(self, list_of_column_names):
        for column_name in list_of_column_names:
            self.data[f'{column_name}'] = pd.Categorical(self.data[f'{column_name}'])

        return

    def convert_to_numeric(self, list_of_column_names):
        for column_name in list_of_column_names:
            self.data[f'{column_name}'] = pd.to_numeric(self.data[f'{column_name}'])

        return

    def check_null(self):
        print(self.data.isnull().sum())

    def get_correlation(self, column_name):
        correlation = self.data.corr()
        print(f'Correlation of {column_name} with other columns is: ')
        print(correlation[f'{column_name}'].sort_values(ascending=False))

    def impute_categorical(self, column_names):
        for column_name in column_names:
            self.data[column_name].fillna(self.data[column_name].mode()[0], inplace=True)
    @abstractmethod
    def preprocessing(self):
        pass
