import catboost as catboost
import lightgbm as lightgbm
import xgboost as xgboost
from sklearn import svm, naive_bayes, tree, ensemble, linear_model
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

from app.abstracts.model_abtract import ModelAbstract


class ModelService(ModelAbstract):

    def __init__(self, data_service):
        super().__init__(data_service)
        self.model_name = 'No algorithm specified'

    def load_algorithm(self, algorithm):
        """
        loads an algorithm based on the parameter
        :param algorithm: naive_bayes, decision_tree, svm, rf, ada_boost, gradient_boost, extra_tree, voting_ensemble, stacking_ensemble
        :return:
        """
        self.model_name = algorithm
        if algorithm == 'lr':
            self.model = self.lr_model()
        elif algorithm == 'dt':
            self.model = self.dt_model()
        elif algorithm == 'rf':
            self.model = self.rf_model()
        elif algorithm == 'nb':
            self.model = self.nb_model()
        elif algorithm == 'bagging_ensemble':
            self.model = self.bagging_model()
        elif algorithm == 'knn':
            self.model = self.knn_model(10)
        elif algorithm == 'svm':
            self.model = self.svm_model()
        elif algorithm == 'gb':
            self.model = self.gb_model()
        elif algorithm == 'ada_boost':
            self.model = self.ada_model()
        elif algorithm == 'xgboost':
            self.model = self.xgboost_model()
        elif algorithm == 'catboost':
            self.model = self.catboost_model()
        elif algorithm == 'lightgbm':
            self.model = self.lightgbm_model()
        elif algorithm == 'lda':
            self.model = self.lda_model()
        elif algorithm == 'qda':
            self.model = self.qda_model()
        elif algorithm == 'gaussian_process':
            self.model = self.gaussian_process_model()

        elif algorithm == 'extra_tree':
            self.model = self.et_model()
        elif algorithm == 'voting_ensemble':
            self.model = self.voting_model()
        elif algorithm == 'stacking_ensemble':
            self.model = self.stacking_model()


        else:
            raise Exception('Please specify a supported algorithm')
            return

        return self.model

    # different algorithms
    def lr_model(self):
        return linear_model.LogisticRegression(random_state=42, max_iter=1000)

    def nb_model(self):
        return naive_bayes.BernoulliNB()

    def dt_model(self):
        return tree.DecisionTreeClassifier(criterion='gini')

    def svm_model(self):
        return svm.SVC(gamma='auto', max_iter=400)

    def rf_model(self):
        return RandomForestClassifier(n_estimators=1000, max_depth=5, random_state=0)

    def ada_model(self):
        return ensemble.AdaBoostClassifier(n_estimators=200, learning_rate=0.1, random_state=0)

    def gb_model(self):
        return ensemble.GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, random_state=0)

    def et_model(self):
        return ensemble.ExtraTreesClassifier(n_estimators=200, random_state=0)

    def voting_model(self):
        return ensemble.VotingClassifier(estimators=[
                ('rf', self.rf_model()),
                # ('lr', self.lr_model()),
                ('nb', self.nb_model()),
                ('svm', self.svm_model()),
                ('dt', self.dt_model())
        ],
            voting='hard', verbose=1, weights=[1, 1, 1, 1]
        )

    def stacking_model(self):
        return ensemble.StackingClassifier(estimators=[
                ('rf', self.rf_model()),
                ('nb', self.nb_model()),
                ('svm', self.svm_model())],
            final_estimator=self.svm_model(), cv=5, passthrough=True, n_jobs=-1,
            verbose=1, stack_method='auto'
        )

    def bagging_model(self):
        return ensemble.BaggingClassifier()

    def xgboost_model(self):
        return xgboost.XGBClassifier()

    def catboost_model(self):
        return catboost.CatBoostClassifier()

    def lightgbm_model(self):
        return lightgbm.LGBMClassifier()

    def knn_model(self, number_of_neighbors=5):
        return KNeighborsClassifier(n_neighbors=number_of_neighbors)

    def lda_model(self):
        return LinearDiscriminantAnalysis()

    def qda_model(self):
        return QuadraticDiscriminantAnalysis()

    def gaussian_process_model(self):
        return GaussianProcessClassifier(random_state=42)

    def rfe_model(self):
        """
        model for feature reselction
        :return: a model
        """

        rfe = RFE(RandomForestClassifier(n_estimators=200), n_features_to_select=30)

        return rfe

    def visualize(self):
        pass

