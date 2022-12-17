# CS6316_Machine_Learning_Final_Project
 
 In this project, we implement seven shallow model, i.e., SVM, Decision Tree, Random Forest, Adaboost, Logistic Regression, and KNN. To train our model, we use 2 different dataset. The dataset can be found inside the "Data" folder. Dataset 1 has 30 attributes and Dataset 2 has 9 attributes. There exists 569 and 462 samples. We split this dataset into 80:20 for training and testing. We further use 10-fold cross validation on training dataset to tune the hyperparameters. 
 
 To use this project files, please install the following library in your local machine:
 
 numpy 
 sklearn  
 mlxtend 
 pandas 
 warnings
 
 We also implement a Neural Network model which can be found inside "Neural Network_MNIST dataset.ipynb". We use popular MNIST dataset to train and test our model. We also use a dataloader to load the data.
 
 To run the models do the following steps.
 1. To run all of the ML Classifiers except Neural Network run ML_Classifiers.ipynb
 2. To run the neural Neural Network model run Neural Network_MNIST dataset.ipynb
 3. To run these file mnist_data_loader.py and performance_metrics.py files are requird
