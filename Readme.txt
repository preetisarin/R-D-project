Anomaly detection using acoustic data in friction stir welding

Acoustic samples can be viewed at: https://cloud.tu-ilmenau.de/s/JP26fPb2darxTr8?path=%2Fdata

# File information
This repository consist of following files:

1. <strong>save_file.py </strong> : in this we convert .dxd files into .csv files. We define sections and annotations for each section  

2. <strong>time_features.py </strong>: Define extracted time domain features function 

3. <strong>Frequency_features.py </strong>: define extracted frequency and time-frequency domain feature functions

4. <strong>Time_feature_analysis.ipynb </strong>: perform time domain feature visualizations 

5. <strong>Frequency_feature_analysis </strong>: perform frequency and time-frequency domain feature visualization 

6. <strong>Parameter_tuning </strong>: perform parameter tuning for SVM

7. <strong>Parameter_tuning_random </strong>: perform parameter tuning for random

8. <strong>WKZ_150600_207005_classify </strong>: perform quality assurance using random forest, SVM for tool 1 data

9. <strong>WKZ_150600_207006_classify </strong>: perform quality assurance using random forest, SVM for tool 2 data

10. <strong>WKZ_150600_207011_classify </strong>: perform quality assurance using random forest, SVM for tool 3 data

11. <strong>ML_time </strong>: perform quality assurance using random forest, SVM and autoencoder for combine data i.e., for tool 1,2 and 3. 



