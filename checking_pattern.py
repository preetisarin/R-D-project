from random import sample
from pydub import AudioSegment
import wave
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import librosa as l
import librosa.display as l1
import scipy as sp
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
    # Returns the loudness of the AudioSegment in dBFS (db relative to the maximum possible loudness). 
    print("dBFS", audio_segment.dBFS)
    # Number of channels in this audio segment (1 means mono, 2 means stereo)
    print("Channels", audio_segment.channels)
    # Number of bytes in each sample (1 means 8 bit, 2 means 16 bit, etc)
    print("sample_width (no of bytes each sample",audio_segment.sample_width)
    # Frame_rate
    print("Frame_rate", audio_segment.frame_rate)
    # root mean square: A measure of loudness.
    print("RMS",audio_segment.rms)
    # Max: The highest amplitude of any sample in the AudioSegment.
    print("Max", audio_segment.max)
    # Max_amplitude: The highest amplitude of any sample in the AudioSegment
    print("Max_amplitude", audio_segment.max_dBFS)
    # duration in seconds
    print("Duration", audio_segment.duration_seconds)
    # frame_count: Returns the number of frames in the AudioSegment
    print("Frame_count", audio_segment.frame_count())
    # frame width
    print("frame width", audio_segment.frame_width)
    #length
    print("len", len(audio_segment))
    l=[audio_segment.channels, audio_segment.sample_width, audio_segment.frame_rate,
        audio_segment.frame_width,len(audio_segment),audio_segment.frame_count(),audio_segment.dBFS]
    return l

def amplitude_envelope(signal,framesize,hop_length):
    amplitude_envelope=[]
    for i in range(0,len(signal),hop_length):
        current_frame_ae=max(signal[i:i+framesize])
        amplitude_envelope.append(current_frame_ae)
    return np.array(amplitude_envelope)

def rms(signal, frame_size, hop_length):
    rms=[]
    for i in range(0,len(signal),hop_length):
        current_frame_rms=np.sqrt(np.sum(signal[i:i+frame_size]**2/frame_size))
        rms.append(current_frame_rms)
    return np.array(rms)
def plot_waveform(y0,y1,y2,y3):
    
    fig, axs = plt.subplots(figsize=(10,15),nrows=4, sharex=True)
    plt.suptitle("Waveform")
    l.display.waveshow(y0,ax=axs[0])
    axs[0].set(title="File: 00")

    l.display.waveshow(y1,ax=axs[1])
    axs[1].set(title="File: 01")

    l.display.waveshow(y2,ax=axs[2])
    axs[2].set(title="File: 02")

    l.display.waveshow(y3,ax=axs[3])
    axs[3].set(title="File: 03")
    plt.subplots_adjust(0.1)
    plt.show()
    plt.ylim(-1,1)

def plot_AE(y0,AE_0,y1,AE_1,y2,AE_2,y3,AE_3,hop_length):
    frames_0=range(0,AE_0.size)
    t_0=l.frames_to_time(frames_0,hop_length=hop_length)

    frames_1=range(0,AE_1.size)
    t_1=l.frames_to_time(frames_1,hop_length=hop_length)

    frames_2=range(0,AE_2.size)
    t_2=l.frames_to_time(frames_2,hop_length=hop_length)

    frames_3=range(0,AE_3.size)
    t_3=l.frames_to_time(frames_3,hop_length=hop_length)

    fig, axs = plt.subplots(figsize=(10,15),nrows=4, sharex=True)
    plt.suptitle("Amplitude Envelope")
    l.display.waveshow(y0,ax=axs[0])
    axs[0].plot(t_0,AE_0,color='r')
    axs[0].set(title="File: 00")

    l.display.waveshow(y1,ax=axs[1])
    axs[1].plot(t_1,AE_1,color='r')
    axs[1].set(title="File: 01")

    l.display.waveshow(y2,ax=axs[2])
    axs[2].plot(t_2,AE_2,color='r')
    axs[2].set(title="File: 02")

    l.display.waveshow(y3,ax=axs[3])
    axs[3].plot(t_3,AE_3,color='r')
    axs[3].set(title="File: 03")
    plt.subplots_adjust(0.1)
    plt.show()
    plt.ylim(-1,1)

def plot_rms(y0,rms_0,y1,rms_1,y2,rms_2,y3,rms_3,hop_length):
    frames_0=range(0,rms_0.size)
    t_0=l.frames_to_time(frames_0,hop_length=hop_length)

    frames_1=range(0,rms_1.size)
    t_1=l.frames_to_time(frames_1,hop_length=hop_length)

    frames_2=range(0,rms_2.size)
    t_2=l.frames_to_time(frames_2,hop_length=hop_length)

    frames_3=range(0,rms_3.size)
    t_3=l.frames_to_time(frames_3,hop_length=hop_length)

    fig, axs = plt.subplots(figsize=(10,15),nrows=4, sharex=True)
    plt.suptitle("Root mean square energy")
    l.display.waveshow(y0,ax=axs[0])
    axs[0].plot(t_0,rms_0,color='r')
    axs[0].set(title="File: 00")

    l.display.waveshow(y1,ax=axs[1])
    axs[1].plot(t_1,rms_1,color='r')
    axs[1].set(title="File: 01")

    l.display.waveshow(y2,ax=axs[2])
    axs[2].plot(t_2,rms_2,color='r')
    axs[2].set(title="File: 02")

    l.display.waveshow(y3,ax=axs[3])
    axs[3].plot(t_3,rms_3,color='r')
    axs[3].set(title="File: 03")
    plt.show()
    plt.ylim(-1,1)

def plot_zcr(zcr_0,zcr_1,zcr_2,zcr_3,frame_size,hop_length):
    frames_0=range(0,zcr_0.size)
    t_0=l.frames_to_time(frames_0,hop_length=hop_length)

    frames_1=range(0,zcr_1.size)
    t_1=l.frames_to_time(frames_1,hop_length=hop_length)

    frames_2=range(0,zcr_2.size)
    t_2=l.frames_to_time(frames_2,hop_length=hop_length)

    frames_3=range(0,zcr_3.size)
    t_3=l.frames_to_time(frames_3,hop_length=hop_length)

    fig, axs = plt.subplots(figsize=(10,15),nrows=4, sharex=True)
    plt.suptitle("Zero crossing rate")
    axs[0].plot(t_0,zcr_0*frame_size,color='r')
    axs[0].set(title="File: 00")
    axs[0].set_xlabel('Time')
    axs[1].plot(t_1,zcr_1*frame_size,color='r')
    axs[1].set(title="File: 01")
    axs[2].plot(t_2,zcr_2*frame_size,color='r')
    axs[2].set(title="File: 02")
    axs[3].plot(t_3,zcr_3*frame_size,color='r')
    axs[3].set(title="File: 03")
    plt.subplots_adjust(bottom=0.1)
    plt.show()

def plot_fft(ft_0,ft_1,ft_2,ft_3,sr0,sr1,sr2,sr3,frame_size):
    magnitude_0=np.absolute(ft_0)
    frequency_0=np.linspace(0,sr0,len(magnitude_0))

    magnitude_1=np.absolute(ft_1)
    frequency_1=np.linspace(0,sr1,len(magnitude_1))

    magnitude_2=np.absolute(ft_2)
    frequency_2=np.linspace(0,sr2,len(magnitude_2))

    magnitude_3=np.absolute(ft_3)
    frequency_3=np.linspace(0,sr3,len(magnitude_3))
    # for closer look
    #num_frequency_bins=int(len(frequency)*0.001) #fratio=1
    fig, axs = plt.subplots(figsize=(10,15),nrows=4, sharex=True)
    plt.suptitle("Fast Fourier Transformation")
    axs[0].plot(frequency_0,magnitude_0*frame_size,color='r')
    axs[0].set(title="File: 00")
    axs[1].plot(frequency_1,magnitude_1*frame_size,color='r')
    axs[1].set(title="File: 01")
    axs[2].plot(frequency_2,magnitude_2*frame_size,color='r')
    axs[2].set(title="File: 02")
    axs[3].plot(frequency_3,magnitude_3*frame_size,color='r')
    axs[3].set(title="File: 03")
    plt.show()

def plot_sc(sc_0,sc_1,sc_2,sc_3,hop_length):
    frames_0=range(0,sc_0.size)
    t_0=l.frames_to_time(frames_0,hop_length=hop_length)

    frames_1=range(0,sc_1.size)
    t_1=l.frames_to_time(frames_1,hop_length=hop_length)

    frames_2=range(0,sc_2.size)
    t_2=l.frames_to_time(frames_2,hop_length=hop_length)

    frames_3=range(0,sc_3.size)
    t_3=l.frames_to_time(frames_3,hop_length=hop_length)

    fig, axs = plt.subplots(figsize=(10,15),nrows=4, sharex=True)
    plt.suptitle("Spectral centroid")
    axs[0].plot(t_0,sc_0,color='r')
    axs[0].set(title="File: 00")
    axs[1].plot(t_0,sc_0,color='r')
    axs[1].set(title="File: 01")
    axs[2].plot(t_0,sc_0,color='r')
    axs[2].set(title="File: 02")
    axs[3].plot(t_0,sc_0,color='r')
    axs[3].set(title="File: 03")
    plt.show()

def plot_sb(sb_0,sb_1,sb_2,sb_3,hop_length):
    frames_0=range(0,sb_0.size)
    t_0=l.frames_to_time(frames_0,hop_length=hop_length)

    frames_1=range(0,sb_1.size)
    t_1=l.frames_to_time(frames_1,hop_length=hop_length)

    frames_2=range(0,sb_2.size)
    t_2=l.frames_to_time(frames_2,hop_length=hop_length)

    frames_3=range(0,sb_3.size)
    t_3=l.frames_to_time(frames_3,hop_length=hop_length)

    fig, axs = plt.subplots(figsize=(10,15),nrows=4, sharex=True)
    plt.suptitle("Spectral bandwidth")
    axs[0].plot(t_0,sb_0,color='r')
    axs[0].set(title="File: 00")
    axs[1].plot(t_0,sb_0,color='r')
    axs[1].set(title="File: 01")
    axs[2].plot(t_0,sb_0,color='r')
    axs[2].set(title="File: 02")
    axs[3].plot(t_0,sb_0,color='r')
    axs[3].set(title="File: 03")
    plt.show()

def plot_mel_spectrogram(log_mel_spectrogram_0,sr0,log_mel_spectrogram_1,sr1,log_mel_spectrogram_2,sr2,log_mel_spectrogram_3,sr3):
    
    fig, axs = plt.subplots(figsize=(10,15),nrows=4, sharex=True)
    plt.suptitle("MEL Spectrogram")
    img_0=l1.specshow(log_mel_spectrogram_0,sr=sr0,ax=axs[0])
    axs[0].set(title="File: 00")
    fig.colorbar(img_0, ax=axs[0])
    img_1=l1.specshow(log_mel_spectrogram_1,sr=sr1,ax=axs[1])
    axs[1].set(title="File: 01")
    fig.colorbar(img_1, ax=axs[1])
    img_2=l1.specshow(log_mel_spectrogram_2,sr=sr2,ax=axs[2])
    axs[2].set(title="File: 02")
    fig.colorbar(img_2, ax=axs[2])
    img_3=l1.specshow(log_mel_spectrogram_3,sr=sr3,ax=axs[3])
    fig.colorbar(img_3, ax=axs[3])
    axs[3].set(title="File: 03")
    plt.show()

def plot_mfcc(mfcc_0,mfcc_1,mfcc_2,mfcc_3):
    fig, axs = plt.subplots(figsize=(10,15),nrows=4, sharex=True)
    plt.suptitle("MFCC")
    img_0=l1.specshow(mfcc_0,ax=axs[0])
    axs[0].set(title="File: 00")
    fig.colorbar(img_0, ax=axs[0])
    img_1=l1.specshow(mfcc_1,ax=axs[1])
    axs[1].set(title="File: 01")
    fig.colorbar(img_1,ax=axs[1])
    img_2=l1.specshow(mfcc_2,ax=axs[2])
    axs[2].set(title="File: 02")
    fig.colorbar(img_2,ax=axs[2])
    img_3=l1.specshow(mfcc_3,ax=axs[3])
    axs[3].set(title="File: 03")
    fig.colorbar(img_3,ax=axs[3])
    plt.show()

if __name__=="__main__":

    # combined_list=[]
    # col_name=["Channels", "Sample width", "Sample rate", "Frame width", "Length (ms)", "Frame count", "Intensity",
    #           "Mean_spectral centroid","std_spectral_centroid", "Mean_RMS","STD_RMS",
    #           "Mean_spectral bandwidth","std_spectral_bandwidth","Mean_spectral roll-off","std_spectral_roll-off",
    #           "Mean_spectral zero-crossing rate","Std_zero_crossing_rate","Mean_MFCC","std_MFCC", "Std_STFT"]
    frame_size=1024
    hop_length=512

    # audio_data_00='Data/0_1_2021_06_17_090407_Sound_Pa_SF0000002,0000.wav'
    # audio_data_01='Data/1_1_2021_07_08_095523_Sound_Pa_SF0000002,0000.wav'
    # audio_data_02='Data/2_1_2021_08_05_120829_Sound_Pa_SF0000002,0000.wav'
    # audio_data_03='Data/3_1_2021_08_12_115910_Sound_Pa_SF0000002,0000.wav'

    audio_data=['0_1_2021_06_17_090407_Sound_Pa_SF0000002,0000.wav',
                '0_2_2021_06_17_091517_Sound_Pa_SF0000002,0000.wav',
                '0_3_2021_06_17_092721_Sound_Pa_SF0000002,0000.wav',
                '0_4_2021_06_17_094050_Sound_Pa_SF0000002,0000.wav',
                '0_5_2021_06_17_095438_Sound_Pa_SF0000002,0000.wav',
                '0_6_2021_06_17_100733_Sound_Pa_SF0000002,0000.wav',
                '0_7_2021_06_17_102035_Sound_Pa_SF0000002,0000.wav',
                '0_8_2021_06_17_104047_Sound_Pa_SF0000002,0000.wav',
                '0_9_2021_06_17_105536_Sound_Pa_SF0000002,0000.wav',
                '0_10_2021_06_17_110845_Sound_Pa_SF0000002,0000.wav',
                '0_11_2021_06_17_112933_Sound_Pa_SF0000002,0000.wav',
                '0_12_2021_06_17_114920_Sound_Pa_SF0000002,0000.wav',
                '0_13_2021_06_17_134557_Sound_Pa_SF0000002,0000.wav',
                '0_14_2021_06_17_140043_Sound_Pa_SF0000002,0000.wav',
                '0_15_2021_06_17_141528_Sound_Pa_SF0000002,0000.wav',
                '0_16_2021_06_17_143005_Sound_Pa_SF0000002,0000.wav',
                '0_17_2021_06_17_144821_Sound_Pa_SF0000002,0000.wav',
                '0_18_2021_06_17_150224_Sound_Pa_SF0000002,0000.wav',
                '0_19_2021_06_17_152121_Sound_Pa_SF0000002,0000.wav',
                '0_20_2021_06_17_153606_Sound_Pa_SF0000002,0000.wav',
                '0_21_2021_06_17_155609_Sound_Pa_SF0000002,0000.wav']

    audio_data_1=['1_1_2021_07_08_095523_Sound_Pa_SF0000002,0000.wav',
                '1_2_2021_07_08_101934_Sound_Pa_SF0000002,0000.wav',
                '1_3_2021_07_08_103512_Sound_Pa_SF0000002,0000.wav',
                '1_4_2021_07_08_105101_Sound_Pa_SF0000002,0000.wav',
                '1_5_2021_07_08_110551_Sound_Pa_SF0000002,0000.wav',
                '1_6_2021_07_08_112530_Sound_Pa_SF0000002,0000.wav',
                '1_7_2021_07_08_114025_Sound_Pa_SF0000002,0000.wav',
                '1_8_2021_07_08_115537_Sound_Pa_SF0000002,0000.wav',
                '1_9_2021_07_08_123951_Sound_Pa_SF0000002,0000.wav',
                '1_10_2021_07_08_125653_Sound_Pa_SF0000002,0000.wav',
                '1_11_2021_07_08_131059_Sound_Pa_SF0000002,0000.wav',
                '1_12_2021_07_08_133004_Sound_Pa_SF0000002,0000.wav',
                '1_13_2021_07_08_135216_Sound_Pa_SF0000002,0000.wav',
                '1_14_2021_07_08_142154_Sound_Pa_SF0000002,0000.wav',
                '1_15_2021_07_08_162545_Sound_Pa_SF0000002,0000.wav',
                '1_16_2021_07_08_164415_Sound_Pa_SF0000002,0000.wav',
                '1_17_2021_07_08_165931_Sound_Pa_SF0000002,0000.wav',
                '1_18_2021_07_08_171513_Sound_Pa_SF0000002,0000.wav',
                '1_19_2021_07_08_173507_Sound_Pa_SF0000002,0000.wav',
                '1_20_2021_07_08_175732_Sound_Pa_SF0000002,0000.wav']
    audio_data_2=['2_1_2021_08_05_120829_Sound_Pa_SF0000002,0000.wav',
                '2_2_2021_08_05_123145_Sound_Pa_SF0000002,0000.wav',
                '2_3_2021_08_05_124736_Sound_Pa_SF0000002,0000.wav',
                '2_4_2021_08_05_130406_Sound_Pa_SF0000002,0000.wav',
                '2_5_2021_08_05_132001_Sound_Pa_SF0000002,0000.wav',
                '2_6_2021_08_05_133527_Sound_Pa_SF0000002,0000.wav',
                '2_7_2021_08_05_135737_Sound_Pa_SF0000002,0000.wav',
                '2_8_2021_08_05_142807_Sound_Pa_SF0000002,0000.wav',
                '2_9_2021_08_05_144359_Sound_Pa_SF0000002,0000.wav',
                '2_10_2021_08_05_171531_Sound_Pa_SF0000002,0000.wav',
                '2_11_2021_08_05_173121_Sound_Pa_SF0000004,0000.wav',
                '2_12_2021_08_19_122626_Sound_Pa_SF0000002,0000.wav',
                '2_13_2021_08_19_124757_Sound_Pa_SF0000002,0000.wav',
                '2_14_2021_08_19_130234_Sound_Pa_SF0000002,0000.wav',
                '2_15_2021_08_19_131846_Sound_Pa_SF0000002,0000.wav',
                '2_16_2021_08_19_133359_Sound_Pa_SF0000002,0000.wav',
                '2_17_2021_08_19_135158_Sound_Pa_SF0000002,0000.wav',
                '2_18_2021_08_19_143917_Sound_Pa_SF0000002,0000.wav',
                '2_19_2021_08_19_152241_Sound_Pa_SF0000002,0000.wav',
                '2_20_2021_08_19_153903_Sound_Pa_SF0000002,0000.wav',
                '2_21_2021_08_19_155502_Sound_Pa_SF0000002,0000.wav']
    audio_data_3=['3_1_2021_08_12_115910_Sound_Pa_SF0000002,0000.wav',
                '3_2_2021_08_12_125423_Sound_Pa_SF0000002,0000.wav',
                '3_3_2021_08_12_131721_Sound_Pa_SF0000002,0000.wav',
                '3_4_2021_08_12_133432_Sound_Pa_SF0000002,0000.wav',
                '3_5_2021_08_12_134820_Sound_Pa_SF0000002,0000.wav',
                '3_6_2021_08_12_140730_Sound_Pa_SF0000002,0000.wav',
                '3_7_2021_08_12_141936_Sound_Pa_SF0000002,0000.wav',
                '3_8_2021_08_12_143435_Sound_Pa_SF0000002,0000.wav',
                '3_9_2021_08_12_145047_Sound_Pa_SF0000002,0000.wav',
                '3_10_2021_08_12_161050_Sound_Pa_SF0000002,0000.wav',
                '3_14_2021_08_20_082149_Sound_Pa_SF0000002,0000.wav',
                '3_15_2021_08_20_083423_Sound_Pa_SF0000002,0000.wav',
                '3_16_2021_08_20_084849_Sound_Pa_SF0000002,0000.wav',
                '3_17_2021_08_20_090422_Sound_Pa_SF0000002,0000.wav',
                '3_18_2021_08_20_091712_Sound_Pa_SF0000002,0000.wav',
                '3_19_2021_08_20_094229_Sound_Pa_SF0000002,0000.wav',
                '3_20_2021_08_20_095638_Sound_Pa_SF0000002,0000.wav',
                '3_21_2021_08_20_101033_Sound_Pa_SF0000002,0000.wav',
                '3_22_2021_08_20_102553_Sound_Pa_SF0000002,0000.wav',
                '3_23_2021_08_20_104304_Sound_Pa_SF0000002,0000.wav']
    y_combined={}
    sr_combined={}
    for i,file in enumerate(audio_data):
        path='Data/'+file
        y_file,sr_file=l.load(path)
        y_combined[i]=y_file
        sr_combined[i]=sr_file

    print(len(y_combined),y_combined)
    fig, axs = plt.subplots(figsize=(10,15),nrows=2, sharex=True)
    plt.suptitle("Waveform")
    l.display.waveshow(y_combined[0],ax=axs[0])
    axs[0].set(title="File: "+str(1))

    l.display.waveshow(y_combined[1],ax=axs[1])
    axs[1].set(title="File: "+str(2))

    # l.display.waveshow(y_combined[2],ax=axs[2])
    # axs[2].set(title="File: "+str(3))

    # l.display.waveshow(y_combined[3],ax=axs[3])
    # axs[3].set(title="File: "+str(4))

    # l.display.waveshow(y_combined[4],ax=axs[4])
    # axs[4].set(title="File: "+str(5))

    # l.display.waveshow(y_combined[5],ax=axs[5])
    # axs[5].set(title="File: "+str(6))

    # l.display.waveshow(y_combined[6],ax=axs[6])
    # axs[6].set(title="File: "+str(7))

    # l.display.waveshow(y_combined[7],ax=axs[7])
    # axs[7].set(title="File: "+str(8))

    # l.display.waveshow(y_combined[8],ax=axs[8])
    # axs[8].set(title="File: "+str(9))

    # l.display.waveshow(y_combined[9],ax=axs[9])
    # axs[9].set(title="File: "+str(10))

    # l.display.waveshow(y_combined[10],ax=axs[10])
    # axs[10].set(title="File: "+str(11))

    # l.display.waveshow(y_combined[11],ax=axs[11])
    # axs[11].set(title="File: "+str(12))

    # l.display.waveshow(y_combined[12],ax=axs[12])
    # axs[12].set(title="File: "+str(13))

    # l.display.waveshow(y_combined[13],ax=axs[13])
    # axs[13].set(title="File: "+str(14))

    # l.display.waveshow(y_combined[14],ax=axs[14])
    # axs[14].set(title="File: "+str(15))

    # l.display.waveshow(y_combined[15],ax=axs[15])
    # axs[15].set(title="File: "+str(16))

    # l.display.waveshow(y_combined[16],ax=axs[16])
    # axs[16].set(title="File: "+str(17))

    # l.display.waveshow(y_combined[17],ax=axs[17])
    # axs[17].set(title="File: "+str(18))

    # l.display.waveshow(y_combined[18],ax=axs[18])
    # axs[18].set(title="File: "+str(19))

    # l.display.waveshow(y_combined[19],ax=axs[19])
    # axs[19].set(title="File: "+str(20))

    # l.display.waveshow(y_combined[20],ax=axs[2])
    # axs[2].set(title="File: "+str(21))

    # l.display.waveshow(y_combined[21],ax=axs[21])
    # axs[21].set(title="File: "+str(21))

    plt.show()

    # # Amplitude envelope
    # AE_0=amplitude_envelope(y0,frame_size,hop_length)
    # AE_1=amplitude_envelope(y1,frame_size,hop_length)
    # AE_2=amplitude_envelope(y2,frame_size,hop_length)
    # AE_3=amplitude_envelope(y3,frame_size,hop_length)

    # plot_AE(y0,AE_0,y1,AE_1,y2,AE_2,y3,AE_3,hop_length)

    # # Root mean square energy
    # # The root-mean-square here refers to the total magnitude of the signal, which in layman terms can be interpreted as the loudness or energy parameter of the audio file.
    # rms_0=l.feature.rms(y=y0,frame_length=frame_size,hop_length=hop_length)[0]
    # rms_1=l.feature.rms(y=y1,frame_length=frame_size,hop_length=hop_length)[0]
    # rms_2=l.feature.rms(y=y2,frame_length=frame_size,hop_length=hop_length)[0]
    # rms_3=l.feature.rms(y=y3,frame_length=frame_size,hop_length=hop_length)[0]
    # # rms=rms(y,frame_size, hop_length)
    # # print(rms.shape)
    # plot_rms(y0,rms_0,y1,rms_1,y2,rms_2,y3,rms_3,hop_length)

    # # zero-crossing-rate
    # # Zero crossing rate aims to study the rate in which a signal amplitude changes sign from positive to negative or back
    # zcr_0=l.feature.zero_crossing_rate(y=y0,frame_length=frame_size,hop_length=hop_length)[0]
    # zcr_1=l.feature.zero_crossing_rate(y=y1,frame_length=frame_size,hop_length=hop_length)[0]
    # zcr_2=l.feature.zero_crossing_rate(y=y2,frame_length=frame_size,hop_length=hop_length)[0]
    # zcr_3=l.feature.zero_crossing_rate(y=y3,frame_length=frame_size,hop_length=hop_length)[0]
    # # print(zcr.shape)
    # plot_zcr(zcr_0,zcr_1,zcr_2,zcr_3,frame_size,hop_length)

    # # Calculating Spectral centroid
    # # indicates where the centre of mass for a sound is located and is calculated as the weighted mean of the frequencies present in the sound.
    # # If the frequencies in music are same throughout then spectral centroid would be around a centre and if there are high frequencies at the end of sound then the centroid would be towards its end.
    # sc_0=l.feature.spectral_centroid(y=y0, sr=sr0,n_fft=frame_size,hop_length=hop_length)[0]  
    # sc_1=l.feature.spectral_centroid(y=y1, sr=sr1,n_fft=frame_size,hop_length=hop_length)[0]  
    # sc_2=l.feature.spectral_centroid(y=y2, sr=sr2,n_fft=frame_size,hop_length=hop_length)[0]  
    # sc_3=l.feature.spectral_centroid(y=y3, sr=sr3,n_fft=frame_size,hop_length=hop_length)[0]     
    # plot_sc(sc_0,sc_1,sc_2,sc_3,hop_length)
    
    # # Calculating Spectral bandwidth
    # sb_0=l.feature.spectral_bandwidth(y=y0, sr=sr0,n_fft=frame_size,hop_length=hop_length)[0]     
    # sb_1=l.feature.spectral_bandwidth(y=y1, sr=sr1,n_fft=frame_size,hop_length=hop_length)[0]    
    # sb_2=l.feature.spectral_bandwidth(y=y2, sr=sr2,n_fft=frame_size,hop_length=hop_length)[0]    
    # sb_3=l.feature.spectral_bandwidth(y=y3, sr=sr3,n_fft=frame_size,hop_length=hop_length)[0]    
    
    # plot_sb(sb_0,sb_1,sb_2,sb_3,hop_length)

    # # Calculating MFCC
    # mfcc_0=l.feature.mfcc(y=y0,n_mfcc=13,sr=sr0)
    # mfcc_1=l.feature.mfcc(y=y1,n_mfcc=13,sr=sr1)
    # mfcc_2=l.feature.mfcc(y=y2,n_mfcc=13,sr=sr2)
    # mfcc_3=l.feature.mfcc(y=y3,n_mfcc=13,sr=sr3)

    # plot_mfcc(mfcc_0,mfcc_1,mfcc_2,mfcc_3)

    # # Mel Spectrogram 
    # mel_spectrogram_0=l.feature.melspectrogram(y=y0,sr=sr0,n_fft=frame_size,hop_length=hop_length,n_mels=20)
    # log_mel_spectrogram_0=l.power_to_db(mel_spectrogram_0)

    # mel_spectrogram_1=l.feature.melspectrogram(y=y1,sr=sr1,n_fft=frame_size,hop_length=hop_length,n_mels=20)
    # log_mel_spectrogram_1=l.power_to_db(mel_spectrogram_1)

    # mel_spectrogram_2=l.feature.melspectrogram(y=y2,sr=sr2,n_fft=frame_size,hop_length=hop_length,n_mels=20)
    # log_mel_spectrogram_2=l.power_to_db(mel_spectrogram_2)
    
    # mel_spectrogram_3=l.feature.melspectrogram(y=y3,sr=sr3,n_fft=frame_size,hop_length=hop_length,n_mels=20)
    # log_mel_spectrogram_3=l.power_to_db(mel_spectrogram_3)
    
    # plot_mel_spectrogram(log_mel_spectrogram_0,sr0,log_mel_spectrogram_1,sr1,log_mel_spectrogram_2,sr2,log_mel_spectrogram_3,sr3)

    # # fast fourier transform
    # ft_0=np.fft.fft(y0)
    # ft_1=np.fft.fft(y1)
    # ft_2=np.fft.fft(y2)
    # ft_3=np.fft.fft(y3)

    # plot_fft(ft_0,ft_1,ft_2,ft_3,sr0,sr1,sr2,sr3,frame_size)