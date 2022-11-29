import dwdatareader as dw
import matplotlib.pyplot as plt 
import noisereduce as nr
from IPython.display import Audio
from scipy import signal
from scipy.fft import fft, fftfreq, fftshift, rfft, rfftfreq
from scipy.signal import butter,filtfilt
import numpy as np
import acoustics.signal as S
import pandas as pd
import re
import os
import math

def compute_endpoints(filepath):
    '''
    get L and M curve endpoints in seconds, eg. L ends at 25.10 second and M ends at 81.23 
    '''
    
    dewefile_handle = dw.open(filepath)

    z_pos = dewefile_handle["Rob_Pos_Z"].series().to_numpy(dtype='float')
    time_z_pos = dewefile_handle["Rob_Pos_Z"].series().index
    time_z_pos -= time_z_pos[0]

    fs_z_pos = dewefile_handle["Rob_Pos_Z"].number_of_samples / time_z_pos[-1]
    
    endpoint_L, endpoint_Maeander = get_seam_endpoints(z_pos, fs_z_pos,time_z_pos)

    # print("{:.2f}, {:.2f}".format(endpoint_L, endpoint_Maeander))
    return np.round(endpoint_L,5),np.round(endpoint_Maeander,5)
    
def get_seam_endpoints(z_pos, fs,time_z_pos):
    """Function that gets the endpoints of the different weld seams. The end is defined as the moment as the tool is moved
    up and released out of the metal sheet. For this purpose the signal is filtered by a FIR Filter too smooth the
    stair shaped signal of the roboter position in z. After this the second derivate of the position is calculated
    to get the first point of steppping out of the sheet.

    The signals are plotted and the result can be validated by visual control.

    The functions returns the times of the end of welds."""
    # creating and applying the fir filter
    taps = 100
    fir_filt = signal.firwin(taps, 1, fs=fs)
    filtered = signal.lfilter(fir_filt, 1.0, z_pos)

    # fir filter phase shifts the signal. delay determines the shift in seconds
    delay = 0.5 * (taps - 1) / fs

    # defining start point to look for the peak which correspondends to the out stepping of the tool
    start_offset = int(5 * fs)

    # getting the peaks
    peaks, properties = signal.find_peaks(np.gradient(np.gradient(filtered[start_offset:])), height=[0.002, 0.006],
                                          prominence=.004)

    peaks += start_offset
    
    # check if exactly two end points were found. If not exactly twi points were found an exception is raised
    if len(peaks) > 2:
        print("more than two maximums found. Getting endpoints aborted")
        raise
    elif len(peaks) < 2:
        print("not enough maximums found")
        raise
    
    L_cut_time=time_z_pos[peaks[0]]
    M_cut_time=time_z_pos[peaks[1]]
    # return time_z_pos[peaks[0]], time_z_pos[peaks[1]]
    return L_cut_time,M_cut_time

def round_down(n, decimals=0):
    multiplier = 10 ** decimals
    return math.floor(n * multiplier) / multiplier

def round_up(n, decimals=0):
    multiplier = 10 ** decimals
    return math.ceil(n * multiplier) / multiplier

def get_subsection_indices(L_cut,M_cut,feed_rate,read_subsection,fs):
    
    read_subsection['start_index']=[int(round_down(x/fs)) for x in ((read_subsection['start']/feed_rate).values.tolist())]
    read_subsection['end_index']=[int(round_up(x/fs)) for x in ((read_subsection['end']/feed_rate).values.tolist())]
    return read_subsection

def get_start(L_cut,M_cut,feed_rate,read_subsection):
        
    read_subsection['duration']=np.round((read_subsection['end']-read_subsection['start'])/feed_rate,4)
    
    total_L_duration=sum(read_subsection['duration'].iloc[0:3])
    total_M_duration=sum(read_subsection['duration'].iloc[3:10])

    print(total_L_duration,total_M_duration)

    start_L_time=L_cut-total_L_duration
    start_M_time=M_cut-total_M_duration

    print(start_L_time,start_M_time)
    
    return start_L_time, start_M_time


def get_annotation(filename,feed_rate,fs,annotation,L_start,M_start):
    '''    
    get all cavities start and end cavities time 
    eg. L has 3 cavities 
    First cavity start at time 0 second and last until 0.23 seconds   
    second cavity starts at 0.5 seconds and lasts until 1.45 seconds and so on. 
    '''

    # read annotated files
    R=pd.read_csv(r'..\Data\All_Xray.csv')

    # convert cavity start and end (in mm) to sec and append columns to R dataframe [
    R['cavity_start_index']=[int(x/fs) for x in ((R['cavity_start[mm]:']/feed_rate).values.tolist())]
    R['cavity_end_index']=[int(x/fs) for x in ((R['cavity_end[mm]:']/feed_rate).values.tolist())]


    L_indices=np.asarray(np.where(((R['sample_id'] == filename)&(R['track'] == 'L-curve'))))
    M_indices=np.asarray(np.where(((R['sample_id'] == filename)&(R['track'] == 'maeander-curve'))))

    sum=0
    for I in L_indices[0]:
        s=int(round_down(R['cavity_start_index'][I]))+L_start
        e=int(round_up(R['cavity_end_index'][I]))+L_start
        np.put(annotation,range(s,e+1),1)
    for I in M_indices[0]:
        s=int(round_down(R['cavity_start_index'][I]))+M_start
        e=int(round_up(R['cavity_end_index'][I]))+M_start
        np.put(annotation,range(s,e+1),1)
    
    print("Count of zeros",np.count_nonzero(annotation),len(annotation)-np.count_nonzero(annotation))
    return annotation

def loading(filepath, store_filepath='./data/'):
    dw_handle=dw.open(filepath)
    
    if store_filepath=='./data':
        filename=re.sub('.dxd', '', filepath)
    else:
        filename=re.sub('.dxd', '', filepath[8:])

    # load SE8
    se8_data=dw_handle['SE8'].series().to_numpy(dtype='float64')
    se8_time= dw_handle['SE8'].series().index.to_numpy()
    se8_time-=se8_time[0]

    sampling_rate= len(se8_data)/se8_time[-1]
    fs=1/sampling_rate

    # get cuttime
    L_cut,M_cut=compute_endpoints(filepath)

    # # get cut time indices (why???) 
    # L_cut_indices=round_up(L_cut/fs)
    # M_cut_indices=round_up(M_cut/fs)

    feed_rate=16.67

    read_subsection=pd.read_csv('subsection_end_point.csv')

    L_start,M_start=get_start(L_cut,M_cut,feed_rate,read_subsection)
    L_start_index=int(round_down(L_start/fs))
    M_start_index=int(round_down(M_start/fs))

    subsection_indices=get_subsection_indices(L_cut,M_cut,feed_rate,read_subsection,fs)
    
    annotation=np.zeros(len(se8_data))
    annotation=get_annotation(filename,feed_rate,fs,annotation,L_start_index,M_start_index)

    rotational_speed=np.zeros(len(se8_data))
    force=np.zeros(len(se8_data))

    L_subsection_name=['I_1','II_1','III']
    M_subsection_name=['I_2','II_2','IV_1','II_3','IV_2','II_4','IV_3']

    subsection_name=np.zeros(len(se8_data))

    for v in L_subsection_name:
        index=np.asarray(np.where(((subsection_indices['curve']=='L_curve')&(subsection_indices['section']==v))))
        # print(index)
        # index=subsection_indices[subsection_indices['section']=='I_1'].index.values
        s=int(subsection_indices['start_index'][index[0]]+L_start_index)
        e=int(subsection_indices['end_index'][index[0]]+L_start_index)
        
        if v=='I_1':
            np.put(subsection_name, range(s,e),1)
            np.put(rotational_speed, range(s,e),1250)
            np.put(force, range(s,e),8000)
        if v=='II_1':
            np.put(subsection_name, range(s,e),2)
            np.put(rotational_speed, range(s,e),1500)
            np.put(force, range(s,e),8500)
        if v=='III':
            np.put(subsection_name, range(s,e),3)
            np.put(rotational_speed, range(s,e),2000)
            np.put(force, range(s,e),8500)
        
    for v in M_subsection_name:
        index=np.asarray(np.where(((subsection_indices['curve']=='Maeander')&(subsection_indices['section']==v))))
        # index=subsection_indices[subsection_indices['section']=='I_1'].index.values
        s=int(subsection_indices['start_index'][index[0]]+M_start_index)
        e=int(subsection_indices['end_index'][index[0]]+M_start_index)
        
        if v=='I_2':
            np.put(subsection_name, range(s,e),4)
            np.put(rotational_speed, range(s,e),1250)
            np.put(force, range(s,e),8000)
        if v=='II_2':
            np.put(subsection_name, range(s,e),5)
            np.put(rotational_speed, range(s,e),1500)
            np.put(force, range(s,e),8500)
        if v=='IV_1':
            np.put(subsection_name, range(s,e),6)
            np.put(rotational_speed, range(s,e),1750)
            np.put(force, range(s,e),8500)
        if v=='II_3':
            np.put(subsection_name, range(s,e),7)
            np.put(rotational_speed, range(s,e),1500)
            np.put(force, range(s,e),8500)
        if v=='IV_2':
            np.put(subsection_name, range(s,e),8)
            np.put(rotational_speed, range(s,e),1750)
            np.put(force, range(s,e),8500)
        if v=='II_4':
            np.put(subsection_name, range(s,e),9)
            np.put(rotational_speed, range(s,e),1500)
            np.put(force, range(s,e),8500)
        if v=='IV_3':
            np.put(subsection_name, range(s,e),10)
            np.put(rotational_speed, range(s,e),1750)
            np.put(force, range(s,e),8500)
    
    dict={'Time':se8_time,'SE8':se8_data,'Rotational_speed':rotational_speed,'Force':force,'Subsection':subsection_name,'label':annotation}
    Combined_data=pd.DataFrame(dict)
    print(store_filepath+'/'+filename+'.csv')
    Combined_data.to_csv(store_filepath+'/'+filename+'.csv',index=False)

def compute_raw_csv(directory,store_path):
    for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith('.dxd'):
                    filepath=os.path.join(root,file)
                    print(filepath)
                    loading(filepath,store_path)

def compute_csv_file(filepath,store_path):
    return loading(filepath,store_path)
