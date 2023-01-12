# R-n-D-project
# Title: Anomaly detection with acoustic data in friction stir welding

# Description:
The aim is to investigate airborne acoustic signals generated during the in-line friction stir welding process and determine whether the airborne acoustic emissions are abnormla or normal. Perform windowing operations on acoustic emissions to generate additional data, because the data set is imbalanced. Extract the properties of acoustic emissions in three different time, frequency, and time-frequency domains to analyze the basic properties of acoustic emissions. Train the Random Forest, Support Vector Machine, and Convolutional Autoencoder (CAE) techniques on these collected features to classify acoustic emissions as abnormal or not. Automated anomaly detection
techniques decrease unproductive process times and manufacturing costs by detecting anomalies early on.

# Source location:
https://cloud.tu-ilmenau.de/s/JP26fPb2darxTr8?path=%2F

# Files information:
save_file.py : converts .dxd files to .csv files. Other process parameters considered are rotationla speed and force. 
time_features.py: defines time domain features extracted.
frequency_features.py: defines frequency domain features , i.e., Power spectral density and MFCC 
time_feature_analysis.ipynb: visulization of various time domain features
frequency_feature_analysis.ipynb: visualization of frequency domain features.
Parameter_tuning.ipynb: hyperparameter tuning for support vector machine. It is performed on all tool types.
Parameter_tuning_random.ipynb: hyperparameter tuning for random forest. It is performed on all tool types.
WKZ_150600_207005_classify.ipynb: Classification results of tool 1 for all sections II, III, and IV
WKZ_150600_207006_classify.ipynb: Classification results of tool 2 for all sections II, III and IV
WKZ_150600_207011_classify.ipynb: Classification results of tool 3 for all sections II, III and IV
ML_time.ipynb: ML (random forest, SVM and autoencoder) results for grouped sections for all tools. 




