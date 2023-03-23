import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, accuracy_score, matthews_corrcoef
 # Import DataService and ModelService classes
from app.services.data_service import DataService
from app.services.model_service import ModelService
if __name__ == '__main__':
    # Create a DataService instance
    data_service = DataService(f"{os.getcwd()}/data/nsclc_radiogenomic_data.csv")
     # Preprocess the data
    data_service.preprocessing()
     # Perform feature selection
    # data_service.feature_selection()
     # Split the data into train and test sets
    data_service.get_train_test_data()
     # Initialize a dictionary to store the models
    models = {}
     # List of model names
    model_names = ['lr', 'dt', 'rf', 'nb', 'knn', 'svm', 'gb', 'ada_boost',
                   'xgboost', 'catboost', 'lda', 'gaussian_process',
                   'extra_tree', 'voting_ensemble', 'stacking_ensemble']
     # Iterate over the list of model names
    for model_name in model_names:
        # Create a ModelService instance
        model = ModelService(data_service)
         # Load the algorithm
        model.load_algorithm(model_name)
         # Add the model to the models dictionary
        models[model_name] = model

     # Initialize lists to store the model evaluation metrics
    model_accuracies = []
    model_precisions = []
    model_recalls = []
    model_cohenskappas = []
    model_evaluations=[model_accuracies, model_precisions, model_recalls]
     # Iterate over the models
    for model_name in models:
        # Print the name of the model being trained
        print("Training model: ", model_name)
         # Train the model
        models[model_name].train()
         # Evaluate the model
        models[model_name].evaluation()
         # Append the model evaluation metrics to the corresponding lists
        model_accuracies.append(models[model_name].accuracy)
        model_precisions.append(models[model_name].precision)
        model_recalls.append(models[model_name].recall)
        model_cohenskappas.append(models[model_name].cohen_kappa)

     # Get the testing labels
    testing_y = np.array(data_service.testing['Recurrence'])
     # Get the testing features
    testing_x = data_service.testing[[col for col in data_service.testing.columns if col != 'Recurrence']]
     # Iterate over the models
    for model_name in models:
        # Make predictions using the model
        # predict = models[model_name].predict(testing_x)
         # Print the model evaluation metrics
        # print(f'{model_name} '
        #       f'accuracy: {accuracy_score(testing_y, predict)} '
        #       f'cohen_kappa: {cohen_kappa_score(testing_y, predict)}')
        # without bootstraping
        predict = models[model_name].predict(data_service.X_test)
        print(f'{model_name} '
              f'accuracy: {accuracy_score(data_service.y_test, predict)} '
              f'cohen_kappa: {cohen_kappa_score(data_service.y_test, predict)}'
              )


        # print(f'{model_name} accuracy: {np.sum(predict == testing_y) / len(testing_y)}')
        # print confusion matrix
        # print(f'{model.model_name} confusion matrix: {pd.crosstab(testing_y, predict, rownames=["Actual"], colnames=["Predicted"])}')
        # print classification report
        # print(f'{model.model_name} classification report: {pd.crosstab(testing_y, predict, rownames=["Actual"], colnames=["Predicted"])}')

