from enum import auto
import dwdatareader as dw
import os
import math 
import numpy as np
import pandas as pd
import csv
import scipy
from scipy.io import wavfile
from scipy import signal
import matplotlib.pyplot as plt 
from scipy.fft import fft, fftfreq, fftshift, rfft, rfftfreq
import pywt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import scikitplot as skplt
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import IsolationForest
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
import scikitplot as skplt
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import IsolationForest
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import validation_curve
from sklearn.linear_model import Ridge
from sklearn.model_selection import learning_curve
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn import mixture
from sklearn.svm import SVC
import more_itertools as mit
import seaborn as sns
import math
import os
import csv
from sklearn.model_selection import StratifiedShuffleSplit
from imblearn.under_sampling import RandomUnderSampler
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree

class machine:
    def standardize(self,data):
        std_scaler = StandardScaler()
        scaled = std_scaler.fit_transform(data)
        return scaled

    def normalized(self,data):
        norm_data=preprocessing.normalize(data)
        return norm_data

    def partition(self,data,label):
        x_train, x_test, y_train, y_test = train_test_split(data, label,stratify=label, random_state=0, train_size = .80,test_size=0.20,shuffle=True)
        return x_train,x_test,y_train,y_test

    # Generating synthetic data and perform upsampling 
    def handle_imbalance_using_upsampling(self,data,label):
        smote = SMOTE()

        # # fit predictor and target variable
        smote_data, smote_label = smote.fit_resample(data, label)
        return smote_data,smote_label

    def handle_imbalance_using_downsampling(self,data,label):
        rus = RandomUnderSampler()
        X_rus,y_rus=rus.fit_resample(data,label)
        
        return X_rus,y_rus
        
    def PCA(self,data,components):
        pca = PCA(n_components=components)
        principalComponents = pca.fit_transform(data)

        print("Explained variance",pca.explained_variance_)
        print("Explained variance ratio",pca.explained_variance_ratio_)
        print("Explained variance cumulative sum", pca.explained_variance_ratio_.cumsum())
        print("Explained variance for each PCA component: ",pca.explained_variance_ratio_)
        print("Sum explained variance of PCS components: {:2.2%}".format(sum(pca.explained_variance_ratio_)))
        return principalComponents

    def evaluate_classifier(self,data, label,model,sampling,func):
        unique, counts = np.unique(label, return_counts=True)
        d=dict(zip(unique, counts))
        print(f"Numbers of class instances (Raw): {d}")
        print(f"Numbers of class instances (Raw): {np.bincount(label)}")
        # normalized data
        if func=='standardize':
            scaled_data=self.standardize(data)
        elif func=='normalize':
            scaled_data=self.normalized(data)

        # partition data
        x_train,x_test,y_train,y_test=self.partition(scaled_data,label)

        print(f"Numbers of train class instances by class (before splitting): {np.bincount(y_train)}")
        print(f"Numbers of test class instances by class (before splitting): {np.bincount(y_test)}")
        print(sampling)
        # Perform upsampling for imbalance using SMOTE
        if sampling=='up':
            smote_data,smote_label=self.handle_imbalance_using_upsampling(x_train,y_train)
            print(f"Numbers of class instances by class (after upsampling): {np.bincount(smote_label)}")
        elif sampling=='down':
            smote_data,smote_label=self.handle_imbalance_using_downsampling(x_train,y_train)
            print(f"Numbers of class instances by class (after downsampling): {np.bincount(smote_label)}")
        elif sampling==None:
            smote_data=data
            smote_label=label

        print(f"Numbers of train class instances by class (after splitting): {np.bincount(y_train)}")
        print(f"Numbers of test class instances by class (after splitting): {np.bincount(y_test)}")

        model.fit(smote_data,smote_label)
        
        cM=metrics.confusion_matrix(y_test, model.predict(x_test))
        print("Confusion matrix \n",cM)
        
        cR=metrics.classification_report(y_test, model.predict(x_test))
        print("ClassificationReport \n",cR)

        scores = cross_val_score(model, smote_data, smote_label, cv=5)
        print('Scores',scores)

        accuracy= metrics.accuracy_score(y_train, model.predict(x_train))
        print("Train Accuracy=",accuracy)

        accuracy= metrics.accuracy_score(y_test, model.predict(x_test))
        print("Test Accuracy=",accuracy)

        # print("Confusion matrix \n")
        # ConfusionMatrixDisplay.from_predictions(y_test,model.predict(x_test))

    def evaluate_classifier_new(self,data, label,model,sampling):
        unique, counts = np.unique(label, return_counts=True)
        d=dict(zip(unique, counts))
        print(f"Numbers of class instances (Raw): {d}")
                
        # partition data
        x_train,x_test,y_train,y_test=self.partition(data,label)

        print(f"Numbers of train class instances by class (before splitting): {np.bincount(y_train)}")
        print(f"Numbers of test class instances by class (before splitting): {np.bincount(y_test)}")
        print(sampling)
        # Perform upsampling for imbalance using SMOTE
        if sampling=='up':
            smote_data,smote_label=self.handle_imbalance_using_upsampling(x_train,y_train)
            print(f"Numbers of train class instances by class (after upsampling): {np.bincount(smote_data)}")
            print(f"Numbers of test class instances by class (after upsampling): {np.bincount(y_test)}")
        elif sampling=='down':
            smote_data,smote_label=self.handle_imbalance_using_downsampling(x_train,y_train)
            print(f"Numbers of class instances by class (after downsampling): {np.bincount(smote_label)}")
            print(f"Numbers of test class instances by class (after downsampling): {np.bincount(y_test)}")
        elif sampling==None:
            smote_data=data
            smote_label=label
            print(f"Numbers of class instances by class (no sampling): {np.bincount(smote_label)}")
            print(f"Numbers of test class instances by class (no sampling): {np.bincount(y_test)}")

        model.fit(smote_data,smote_label)
        
        cM=metrics.confusion_matrix(y_test, model.predict(x_test))
        print("Confusion matrix \n",cM)
        
        scores = cross_val_score(model, smote_data, smote_label, cv=5)
        print('Scores',scores)

        accuracy= metrics.accuracy_score(y_train, model.predict(x_train))
        print("Train Accuracy=",accuracy)

        accuracy= metrics.accuracy_score(y_test, model.predict(x_test))
        print("Test Accuracy=",accuracy)

        cR=metrics.classification_report(y_test, model.predict(x_test))
        print("ClassificationReport \n",cR)

        print("Confusion matrix \n")
        ConfusionMatrixDisplay.from_predictions(y_test,model.predict(x_test))
        
   