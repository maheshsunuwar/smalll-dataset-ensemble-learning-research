from abc import ABC, abstractmethod, abstractclassmethod

import click
from scipy.stats import kendalltau, wilcoxon, mannwhitneyu
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, roc_auc_score, f1_score, \
    roc_curve, cohen_kappa_score, matthews_corrcoef


class ModelAbstract(ABC):

    def __init__(self, dataService):
        self.dataService = dataService
        self.model = None
        self.confusion_matrix, self.accuracy, self.precision, self.recall = None, None, None, None
        self.cohen_kappa, self.kendall_tau, self.wilcoxon, self.mannwhitneyu = None, None, None, None
        self.mathews_corrcoef = None
        self.f1, self.roc_auc, self.fpr, self.tpr, self.thresholds = None, None, None, None, None
    @abstractmethod
    def load_algorithm(self, algorithm):
        pass

    def train(self):
        with click.progressbar(length=100, ) as bar:
            self.model.fit(self.dataService.X_train, self.dataService.y_train.values.ravel())
        return

    def evaluation(self):
        print(f"Accuracy of the trained model is: {self.model.score(self.dataService.X_test, self.dataService.y_test)}")

        predictions = self.predict(self.dataService.X_test)
        self.confusion_matrix = confusion_matrix(self.dataService.y_test, predictions)
        self.accuracy = accuracy_score(self.dataService.y_test, predictions)
        self.recall = recall_score(self.dataService.y_test, predictions)
        self.precision = precision_score(self.dataService.y_test, predictions)
        self.f1 = f1_score(self.dataService.y_test, predictions)
        self.roc_auc = roc_auc_score(self.dataService.y_test, predictions)
        self.fpr, self.tpr, self.thresholds = roc_curve(self.dataService.y_test, predictions)

        self.cohen_kappa = cohen_kappa_score(self.dataService.y_test, predictions)
        # self.kendall_tau = kendalltau(self.dataService.y_test, predictions)
        # self.wilcoxon = wilcoxon(self.dataService.y_test, predictions)
        # self.mannwhitneyu = mannwhitneyu(self.dataService.y_test, predictions)
        self.mathews_corrcoef = matthews_corrcoef(self.dataService.y_test, predictions)

        print(f'Confusion Matrix: {self.confusion_matrix}')
        print(f'Accuracy: {self.accuracy}')
        # print(f'Recall: {self.recall}')
        # print(f'Precision: {self.precision}')
        print(f'cohen_kappa: {self.cohen_kappa}')
        print(f'mathews_corrcoef: {self.mathews_corrcoef}')

        return self.confusion_matrix, self.accuracy, self.recall, self.precision, self.cohen_kappa, self.mathews_corrcoef

    def predict(self, features):
        return self.model.predict(features)

    @abstractmethod
    def visualize(self):
        pass
