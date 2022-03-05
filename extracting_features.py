from random import sample
from pydub import AudioSegment
import wave
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import librosa as l
import librosa.display as l1
import scipy
import sklearn
import math

def generic_features(audio_file):
    '''
        EXTRACT AUDIO GENERIC FEATURES using pydub
            Input: file path
            Output:  
            Channels: number of channels: 1 for mono, 2 for stereo audio
            Sample width: number of bytes per sample; 1 means 8 bits, 2 means 16 bit
            Frame rate/ sample rate: frequency of sample used (in hertz)
            Frame width: number of bytes for each frame. one fram e contains a sample of each channel
            length: audio file length in millisecond
            frame count: number of frames from samples
            Intensity: loudness in dBFS (dB relative to maximum possible loudness. )
    ''' 
    audio_segment = AudioSegment.from_file(audio_file)
    l=[audio_segment.channels, audio_segment.sample_width, audio_segment.frame_rate,
        audio_segment.frame_width,len(audio_segment),audio_segment.frame_count(),audio_segment.dBFS]
    return l

def librosa_features(signal,samplerate):
    #convert timestamps into STFT frames
    # print("frame",l.time_to_frames(times=signal,sr=samplerate))
    #short fourier transformation
    stft=l.stft(y=signal)
    return np.std(stft)

def zero_crossing_rate(signal):
    '''
        Zero crossing rate aims to study the rate in which a signal amplitude changes sign from positive to negative or back
        Parameter:
            signal: audio time series
            samplerate: audio sample rate
        Output:
            sum of zero_crossing_rate 

    '''
    zcr=l.feature.zero_crossing_rate(signal)
    return zcr

# Normalising the spectral centroid for visualisation
def normalize(x, axis=0):
    return sklearn.preprocessing.minmax_scale(x, axis=axis)


def spectral_centroid(signal):
    '''
        indicates where the centre of mass for a sound is located and is calculated as the weighted mean of the frequencies present in the sound.
        If the frequencies in music are same throughout then spectral centroid would be around a centre and if there are high frequencies at the end of sound then the centroid would be towards its end.
        Parameter:
            signal: audio time series
            samplerate: audio sample rate
        Output:
            spec_centroid an ndarray 
    '''
    sc=l.feature.spectral_centroid(y=signal)[0]  
    return sc

def root_mean_square_energy(signal):
    '''
        The root-mean-square here refers to the total magnitude of the signal, which in layman terms can be interpreted as the loudness or energy parameter of the audio file.
    '''
    #S,phase=l.magphase(l.stft(signal))
    #rms=l.feature.rms(signal,512,4)
    S, phase = l.magphase(l.stft(y=signal))
    rms = l.feature.rms(S=S)
    return rms

def spectral_bandwidth(signal):
    sb=l.feature.spectral_bandwidth(y=signal)
    return sb

def spectral_rolloff(signal):
    sr=l.feature.spectral_rolloff(y=signal)
    return sr

def melfrequency(signal):
    mfcc=l.feature.mfcc(y=signal)
    return mfcc

if __name__=="__main__":

    audio_data="Data/0_1_2021_06_17_090407_Sound_Pa_SF0000002,0000.wav"
    audio_list=['0_1_2021_06_17_090407_Sound_Pa_SF0000002,0000.wav', '0_2_2021_06_17_091517_Sound_Pa_SF0000002,0000.wav',
                '1_1_2021_07_08_095523_Sound_Pa_SF0000002,0000.wav', '1_2_2021_07_08_101934_Sound_Pa_SF0000002,0000.wav',
                '2_1_2021_08_05_120829_Sound_Pa_SF0000002,0000.wav', '2_2_2021_08_05_123145_Sound_Pa_SF0000002,0000.wav',
                '3_1_2021_08_12_115910_Sound_Pa_SF0000002,0000.wav', '3_2_2021_08_12_125423_Sound_Pa_SF0000002,0000.wav']

    combined_list=[]
    for audio in audio_list:
        audio_path=''
        audio_path='Data/'+audio 
        feature_list=generic_features(audio_path)
        
        #load wave file as floating point time series. 
        # automatically resample audio at given sampling rate(sr) i.e. sr=22050
        # to preserve native sampling rate use sr=None
        # returns y, sr
        # y: audio time series. 
        # sr: sampling rate of y
        y,sr=l.load(audio_path)# with sr=None: y=8280000 sr=100000 and when sr=22050 y=1825740
        
        # plotting waveform  
        fig, axs = plt.subplots(figsize=(10,15),nrows=6, sharex=True)
        plt.suptitle(audio)
        plt.subplots_adjust(hspace = 1)
        l.display.waveshow(y,sr=sr,ax=axs[0])
        axs[0].set(title='Waveform')
        
        # Calculating Spectral centroid
        sc=spectral_centroid(y)
        #calculating mean and std of spectral centroid 
        feature_list.append(np.mean(sc))
        feature_list.append(np.std(sc))
        frames=range(len(sc))
        t=l.frames_to_time(frames)
        #l.display.waveshow(y,sr=sr,ax=axs[1])
        axs[1].plot(t,normalize(sc),color='r')
        axs[1].set_xlabel('Time')
        axs[1].set_ylabel('Hz')
        axs[1].set(title="spectral centroid")
        
        # calculating root mean square energy
        rms=root_mean_square_energy(y)
        #calculating mean and std of root mean square energy
        feature_list.append(np.mean(rms))
        feature_list.append(np.std(rms))
        t=l.times_like(rms)
        #l.display.waveshow(y,sr=sr,ax=axs[2])
        axs[2].plot(t,rms[0],color='r')
        axs[2].set_xlabel('Time')
        axs[2].set_ylabel('Hz')
        axs[2].set(title="root mean square")
        

        # Calculating spectral bandwidth 
        sb=spectral_bandwidth(y)
        #calculating mean and std of spectral bandwidth
        feature_list.append(np.mean(sb))
        feature_list.append(np.std(sb))
        times = l.times_like(sb)
        #l.display.waveshow(y,sr=sr,ax=axs[3])
        axs[3].plot(times,sb[0],color='r')
        axs[3].set_xlabel('Time')
        axs[3].set_ylabel('Hz')
        axs[3].set(title="spectral bandwidth")

        # Calculating spectral roll-off
        spr=spectral_rolloff(y)
        #calculating mean and std of spectral roll-off
        feature_list.append(np.mean(spr))
        feature_list.append(np.std(spr))
        times = l.times_like(spr)
        #l.display.waveshow(y,sr=sr,ax=axs[4])
        axs[4].plot(times,spr[0],color='r')
        axs[4].set_xlabel('Time')
        axs[4].set_ylabel('Hz')
        axs[4].set(title="spectral rolloff")
        
        
        # Calculating zero crossing rate
        zcr=zero_crossing_rate(y)
        #calculating mean and std of zero-crossing rate
        feature_list.append(np.mean(zcr))
        feature_list.append(np.std(zcr))
        times = l.times_like(spr)
        #l.display.waveshow(y,sr=sr,ax=axs[5])
        axs[5].plot(times,zcr.T,color='r')
        axs[5].set_xlabel('Time')
        axs[5].set(title="Zero crossing rate")
        plt.show()

        
        #calculating Mel-spectrogramm
        mfcc=melfrequency(y)
        #calculating mean and std of MFCC
        feature_list.append(np.mean(mfcc))
        feature_list.append(np.std(mfcc))
        fig,ax=plt.subplots()
        img=l1.specshow(mfcc)
        ax.set(title="MFCC"+audio)
        ax.set_xlabel('Time')
        ax.set_ylabel('Hz')
        fig.colorbar(img)
        plt.show()

        std_stft=librosa_features(y,sr)
        feature_list.append(std_stft)
        combined_list.append(feature_list)
    col_name=["Channels", "Sample width", "Sample rate", "Frame width", "Length (ms)", "Frame count", "Intensity",
              "Mean_spectral centroid","std_spectral_centroid", "Mean_RMS","STD_RMS",
              "Mean_spectral bandwidth","std_spectral_bandwidth","Mean_spectral roll-off","std_spectral_roll-off",
              "Mean_spectral zero-crossing rate","Std_zero_crossing_rate","Mean_MFCC","std_MFCC", "Std_STFT"]

    df=pd.DataFrame(combined_list,columns=col_name)
    print(df)
    df.to_csv('CSV/feature.csv')
    
    
    
    
    
