from scipy.fft import fft, fftfreq, fftshift, rfft, rfftfreq
from scipy.signal import find_peaks
from scipy.signal import butter,filtfilt
import pandas as pd
import numpy as np

def compute_highest_peaks(data,sampling_rate,time):
    data_mean=np.mean(data)
    data-=data_mean
    y_fourier=rfft(data)/len(data)
    x_fourier=rfftfreq(len(data),1/sampling_rate)
    
    # height_threshold=np.max(data)*(0.0375)
    # print("max",np.max(np.abs(y_fourier)))
    height_threshold=np.max(np.abs(y_fourier))*0.8
    # print("threshold",height_threshold)
    peaks_index, properties = find_peaks(np.abs(y_fourier), height=height_threshold)
    # print(peaks_index)
    # Notes: 
    # 1) peaks_index does not contain the frequency values but indices
    # 2) In this case, properties will contain only one property: 'peak_heights'
    #    for each element in peaks_index (See help(find_peaks) )
    # Let's first output the result to the terminal window:
    max_amplitude_index=properties['peak_heights'].argmax()
    max_amplitude=max(properties['peak_heights'])
    
    peak_frequency=x_fourier[peaks_index[max_amplitude_index]]
    # print("values",max_amplitude_index,max_amplitude, peak_frequency)
    
    return peak_frequency
    

def butter_lowpass_filter(data, cutoff, fs, order):
    # print("Cutoff freq " + str(cutoff))
    nyq = 0.5 * fs # Nyquist Frequency
    normal_cutoff = cutoff / nyq
    print(nyq, normal_cutoff)
    # Get the filter coefficients 
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a,data)
    return y

def butter_highpass_filter(data, cutoff, fs, order):
    # print("Cutoff freq " + str(cutoff))
    nyq = 0.5 * fs # Nyquist Frequency
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients 
    print(nyq, normal_cutoff)
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    y = filtfilt(b, a,data)
    return y

def filtering(data,sampling_rate):
    # perform low pass butter worth filtering, by default order=6 and critical frequencies=100
    
    # b1,a1=signal.iirfilter(N=order,Wn=frequencies,btype="low",ftype='butter',output='ba',fs=sampling_rate)
    # data['SE8']=signal.filtfilt(b1,a1,data['SE8'])
    # return data
    duration=len(data)/sampling_rate
    cutoff=compute_highest_peaks(data, sampling_rate, duration)
    print("Cutoff",cutoff)
    order=3
    filtered_signal = butter_lowpass_filter(data, cutoff, sampling_rate, order)
    return filtered_signal

def filtering_1(data,sampling_rate):
    # perform low pass butter worth filtering, by default order=6 and critical frequencies=100
    
    # b1,a1=signal.iirfilter(N=order,Wn=frequencies,btype="low",ftype='butter',output='ba',fs=sampling_rate)
    # data['SE8']=signal.filtfilt(b1,a1,data['SE8'])
    # return data
    duration=len(data)/sampling_rate
    cutoff=compute_highest_peaks(data, sampling_rate, duration)
    print("Cutoff",cutoff)
    order=3
    filtered_signal = butter_highpass_filter(data, cutoff, sampling_rate, order)
    return filtered_signal
# Reference: https://nehajirafe.medium.com/using-fft-to-analyse-and-cleanse-time-series-data-d0c793bb82e3