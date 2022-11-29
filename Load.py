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

class load():
    def read_data(self,filepath):
        print("use torchtime to raed all data")
    
    def standardize(self, data):
        std_scaler = StandardScaler()
        scaled = std_scaler.fit_transform(data)
        return scaled

    def replace_null_values(self,data):
        print("Write something")
    
    def preprocess(self,data):
        print("Perform preprocessing like")
        print(" replace nan/ none values")
        print("Check size of each column")
    
    def loading(self,):
        print("call other functions")
    