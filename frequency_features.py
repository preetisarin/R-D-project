from scipy import signal
from scipy.fft import fft, fftfreq, fftshift, rfft, rfftfreq
import librosa as l
import librosa.display as ld
import pywt
import acoustics.signal as S
from scipy.signal import fftconvolve
import csv
import numpy as np
import pandas as pd
import PyOctaveBand
import os
import re
import more_itertools as mit
from filter_data import filtering
import matplotlib.pyplot as plt
from scipy.fftpack import dct
class Freq_feature():
    def fast_fourier(self,data,sampling_rate):
        '''
        ---------------FILL IN THIS FUNCTION----------------- 
        '''
        # number_of_samples=len(data)
        # or another way samples= sampling_rate*duration, but for our case, 
        # I have already multiplied sampling rate to time to get equal number of values for time and data
        
        y_fourier=rfft(data)/len(data)
        x_fourier=rfftfreq(len(data),1/sampling_rate)
        # peak,signal_amplitude=self.compute_freq_max(y_fourier,x_fourier,sampling_rate,'fft')
        # result=[]
        # for i in range(len(peak)):
        #     result.append(peak[i])
        #     result.append(signal_amplitude[i])
        # return result
        return np.abs(y_fourier)
        

    def power_spectrum(self,data,sampling_rate):
        freq,psd=signal.welch(data,fs=sampling_rate)
        # print("Frequency",freq)
        # peak,signal_amplitude=self.compute_freq_max(psd,freq,sampling_rate,'power')
        # # result=[]
        # # for i in range(len(peak)):
        # #     result.append(peak[i])
        # #     result.append(signal_amplitude[i])
        # # return result
        return psd
        
    # 'fft_freq_1','fft_amp_1','fft_freq_2','fft_amp_2','fft_freq_3','fft_amp_3','fft_freq_4','fft_amp_4','fft_freq_5','fft_amp_5'
    # 'psd_freq_1','psd_amp_1','psd_freq_2','psd_amp_2','psd_freq_3','psd_amp_3','psd_freq_4','psd_amp_4','psd_freq_5','psd_amp_5'
    # 'corr_freq_1','corr_amp_1','corr_freq_2','corr_amp_2','corr_freq_3','corr_amp_3','corr_freq_4','corr_amp_4','corr_freq_5','corr_amp_5'
    # 'mfcc_1','mfcc_2','mfcc_3','mfcc_4','mfcc_5','mfcc_6','mfcc_7','mfcc_8','mfcc_9','mfcc_10'
    # 'mfcc_11','mfcc_12','mfcc_13','mfcc_14','mfcc_15','mfcc_16','mfcc_17','mfcc_18','mfcc_19','mfcc_20'
    # 'oct_freq1','oct_val1','oct_freq2','oct_val2','oct_freq3','oct_val3','oct_freq4','oct_val4','oct_freq5','oct_val5','oct_freq6','oct_val6'
    # 'oct_freq7','oct_val7','oct_freq8','oct_val8','oct_freq9','oct_val9','oct_freq10','oct_val10','oct_freq11','oct_val11','oct_freq12','oct_val12'
    # 'oct_freq13','oct_val13','oct_freq14','oct_val14','oct_freq15','oct_val15','oct_freq16','oct_val16','oct_freq17','oct_val17','oct_freq18','oct_val18'
    # 'oct_freq19','oct_val19','oct_freq20','oct_val20','oct_freq21','oct_val21','oct_freq22','oct_val22','oct_freq23','oct_val23','oct_freq24','oct_val24'
    # 'oct_freq25','oct_val25','oct_freq26','oct_val26','oct_freq27','oct_val27','oct_freq28','oct_val28','oct_freq29','oct_val29','oct_freq30','oct_val30'
    # 'oct_freq31','oct_val31','oct_freq32','oct_val32','oct_freq33','oct_val33','oct_freq34','oct_val34','oct_freq35','oct_val35','oct_freq36','oct_val36'
    # 'oct_freq37','oct_val37','oct_freq38','oct_val39',
    def MFCC(self,data,sampling_rate):
        Mfcc=l.feature.mfcc(y=data, sr=sampling_rate,n_mfcc=5)
        Mfcc_combine=[]
        for i in range(len(Mfcc)):
            for v in Mfcc[i]:
                Mfcc_combine.append(v)
            # print(len(Mfcc))
        return Mfcc_combine

    def octave_band(self,data, sampling_rate):
        freq_range=[1,20000]
        
        f_min = freq_range[0]
        f_max = freq_range[1]
                                                
        spl, freq= PyOctaveBand.octavefilter(data, fs=sampling_rate, fraction=3, order=6, limits=[f_min, f_max], show=0)
        
        # # print(len(spl),len(freq))
        # result=[]
        # for i,val in enumerate(spl):
        #     result.append(freq[i]) 
        #     result.append(val)
        # # print(result)
        # return result
        return spl

    def MFCC_1(self,signal,sample_rate):
        frame_size=0.025
        frame_stride=0.05
        sample_rate=50000
        frame_length, frame_step = frame_size * sample_rate, frame_stride * sample_rate  # Convert from seconds to samples
        signal_length = len(signal)
        frame_length = int(round(frame_length))
        frame_step = int(round(frame_step))
        num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))  # Make sure that we have at least 1 frame
        print(signal_length,num_frames)
        pad_signal_length = num_frames * frame_step + frame_length
        z = np.zeros((pad_signal_length - signal_length))
        pad_signal = np.append(signal, z) # Pad Signal to make sure that all frames have equal number of samples without truncating any samples from the original signal

        indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
        frames = pad_signal[indices.astype(np.int32, copy=False)]
        print(frames.shape)
        frames *= np.hamming(frame_length)
        NFFT=512
        mag_frames = np.absolute(np.fft.rfft(frames, NFFT))  # Magnitude of the FFT
        pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))
        nfilt=40
        low_freq_mel = 0
        high_freq_mel = (2595 * np.log10(1 + (sample_rate / 2) / 700))  # Convert Hz to Mel
        mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # Equally spaced in Mel scale
        hz_points = (700 * (10**(mel_points / 2595) - 1))  # Convert Mel to Hz
        bin = np.floor((NFFT + 1) * hz_points / sample_rate)

        fbank = np.zeros((nfilt, int(np.floor(NFFT / 2 + 1))))
        for m in range(1, nfilt + 1):
            f_m_minus = int(bin[m - 1])   # left
            f_m = int(bin[m])             # center
            f_m_plus = int(bin[m + 1])    # right

            for k in range(f_m_minus, f_m):
                fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
            for k in range(f_m, f_m_plus):
                fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
        filter_banks = np.dot(pow_frames, fbank.T)
        filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  # Numerical Stability
        filter_banks = 20 * np.log10(filter_banks)  # dB
        num_ceps=20
        mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')[:, 1 : (num_ceps + 1)] # Keep 2-13
        cep_lifter=22
        (nframes, ncoeff) = mfcc.shape
        n = np.arange(ncoeff)
        lift = 1 + (cep_lifter / 2) * np.sin(np.pi * n / cep_lifter)
        mfcc *= lift
        filter_banks -= (np.mean(filter_banks, axis=0) + 1e-8)
        mfcc -= (np.mean(mfcc, axis=0) + 1e-8)
        print(type(mfcc),mfcc.shape)
        return mfcc

    def correlate(self,data,sampling_rate):
        corr_result = np.correlate(data, data, mode='full')
        result=self.fast_fourier(corr_result,sampling_rate)
        return result

    # def compute_freq_max(self,data, freq_data,sampling_rate,type):
    #     signal_amplitude=[]
    #     peak_frequency=[]
    #     freq_range=[[0,5000],[5000,10000],[10000,15000],[15000,20000],[20000,24999]]
    #     for val in freq_range:
    #         # print(val[0],val[1])

    #         indices=np.where(np.logical_and(freq_data>val[0] , freq_data<=val[1]))

    #         new_data=data[indices]
    #         new_freq_data=freq_data[indices]
            
    #         amp_indices=np.argpartition(new_data,-5)[-5:]
    #         signal_amplitude.extend(np.abs(new_data[amp_indices]))
    #         peak_frequency.extend(new_freq_data[amp_indices])
    #         # print(signal_amplitude,peak_frequency)
    #     return peak_frequency,signal_amplitude

    def compute_freq(self,data,sampling_rate):

        func_list={"m1": self.fast_fourier, "m2": self.power_spectrum,"m3":self.correlate,"m4":self.octave_band}
        freq_features=[]
        for key, value in func_list.items():
            # print(key)
            result=value(data,sampling_rate)
            # print("R",result[1])
            if type(result)==list:
                freq_features.extend(result)
            else:
                freq_features.extend(result)
        return freq_features

    def compute_filepath(self,directory,frame_length,overlap,func_name,storing_path='./frequency/'):
        for root, dirs, files in os.walk(directory):
            # print(root,dirs,files)
                for file in files:
                    print(file)
                    if file.endswith('.csv'):
                        filepath=os.path.join(root,file)
                        self.extract_freq(filepath,frame_length,overlap,func_name,storing_path)
    
    def extract_freq(self,filepath,frame_length, overlap,func_name,storing_path='./frequency/'):
        
        print(filepath)
        df=pd.read_csv(filepath)
        
        # if storing_path=='./frequency/':
        #     filename=re.sub('.dxd', '', filepath)
        # else:
        #     filename=re.sub('.dxd', '', filepath[8:])
        
        sampling_rate=50000

        window_size=int(frame_length*sampling_rate)
        stride=int(((overlap/100)*frame_length)*sampling_rate)

        filtered_data=filtering(np.array(df['SE8']),sampling_rate)
        
        subsection_name=[1,2,3,4,5,6,7,8,9,10]
        sub_name=['I_1','II_1','III','I_2','II_2','IV_1','II_3','IV_2','II_4','IV_3']
        for i,val in enumerate(subsection_name):
            storing_filename=re.sub('_\d','',sub_name[i])+'.csv'
                            
            indices=np.where(df['Subsection']==val)
            
            se8_subdata=filtered_data[indices[0][0]:indices[0][-1]]
            label_subdata=df['label'].iloc[indices[0][0]:indices[0][-1]]
            
            # print(len(se8_subdata),len(label_subdata),len(rotational_subdata),len(force_subdata))
            # print(val)
            for i in range(0,len(se8_subdata)-(window_size)+1,stride):
            # for i in range(0,len(se8_subdata),1):
                # print(i)
                f=[]
                start=i
                end=i+window_size
                            
                # if window_size>len(se8_subdata[start:end]):
                #     print("Inside")
                #     end=df.shape[0]
                #     data=np.array([0 if i is None else i for i in mit.take(window_size, mit.padnone(se8_subdata[start:end]))])
                # else:
                data=se8_subdata[start:end]
                # print(len(data))
                # check if all labels are same or not. 
                # if labels are not same then, take maximum value as label
                
                unique, counts = np.unique(label_subdata[start:end], return_counts=True)
                d=dict(zip(unique, counts))
                # print(filepath,start,end,d)
                if len(d)>1 and (unique[0]==unique[1] or unique[0]>unique[1] or unique[1]>unique[0]):
                    max_value=1
                else:
                    max_value=unique[0]

                # print(len(df['SE8'].iloc[start:end]),max_value)
                # add features here
                # if func_name=='freq':
                #     f=list(self.compute_freq(data,sampling_rate))
                # # elif func_name=='octave':
                # #     f=self.octave_band(data,sampling_rate)
                # elif func_name=='mfcc':
                #     f=self.MFCC(data,sampling_rate)
                if func_name=='fft':
                    f=list(self.fast_fourier(data,sampling_rate))
                elif func_name=='power':
                    f=list(self.power_spectrum(data,sampling_rate))
                elif func_name=='octave':
                    f=list(self.octave_band(data,sampling_rate))
                elif func_name=='mfcc':
                    f=list(self.MFCC(data,sampling_rate))

                f.append(max_value)

                # print(storing_path+storing_filename)
                file = open(storing_path+storing_filename, 'a+', newline ='')
                
                with file:
                    write=csv.writer(file)
                    write.writerow(f)