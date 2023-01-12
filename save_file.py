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


def round_down(n, decimals=0):
    multiplier = 10 ** decimals
    return math.floor(n * multiplier) / multiplier

def round_up(n, decimals=0):
    multiplier = 10 ** decimals
    return math.ceil(n * multiplier) / multiplier

# def get_subsection_indices(L_cut,M_cut,feed_rate,read_subsection,fs):
#     L_area=[25,68,133]
#     M_area=[25,49,113,120,115,119,130]

#     read_subsection['start_index']=[int(round_down(x/fs)) for x in ((read_subsection['start']/feed_rate).values.tolist())]
#     read_subsection['end_index']=[int(round_up(x/fs)) for x in ((read_subsection['end']/feed_rate).values.tolist())]
#     return read_subsection
def truncate(value,n):
    return math.floor(value * 10 ** n) / 10 ** n
def subsection_duration(area,feed_rate):
    subsection_time=area/feed_rate
    return subsection_time
def get_L_total_duration(self,feed_rate):
    '''
        compute L curve total duration
    '''
    #L_area_name=['I_1','II_1','buffer_L1','III_K','III','buffer_L2']
    L_area=np.array([25,68,10,28,90,5]) # all dimensions in mm
    
    L_time=np.round(L_area/feed_rate,5)
    return sum(L_time)

    # get total length of M cut
def get_M_total_duration(self,feed_rate):
    '''
        compute Maeander curve total duration
    '''
    #M_area_name=['I_2','II_2','buffer_M1','IV_1_K','IV_1','buffer_M2','II_3_K',
    #        'II_3','buffer_M3','IV_2_K','IV_2','buffer_M4','II_4_K','II_4','buffer_M5','IV_3_K','buffer_M6','V','buffer_M7']
    M_area=np.array([25,49,10,41,62,10,48,62,10,43,62,10,48,61,10,34,10,71,5]) # all dimensions in mm

    M_time=np.round(M_area/feed_rate,5)
    return sum(M_time)

def get_start(L_cut,M_cut,feed_rate):
    '''
        get start location of L and maeander curve in seconds
        Input:
            L_cut: seconds when L finishes
            M_cut: seconds when Maeander finishes
            feed_rate: feedrate contant 16.67
            read_subsection: 
    '''    
    # read_subsection['duration']=np.round((read_subsection['end']-read_subsection['start'])/feed_rate,4)
    
    total_L_duration=get_L_total_duration(feed_rate)
    total_M_duration=get_M_total_duration(feed_rate)

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


def get_subsection_end_points(data,cut_time,subsection_time,filepath,subsection_name) :
    print(cut_time,subsection_time)
    file_indices=[]
    end=cut_time
    start=0
    s=0
    # L_start=cut_time-sum(subsection_time)
    # print("Cut and subsection",cut_time,subsection_time)
    L_section_list=['III','II_1','I_1']
    M_section_list=['IV_3','II_4','IV_2','II_3','IV_1','II_2','I_2']
    section_start=[]
    section_end=[]
    sum=0
    if subsection_name=='L_sub':
            
        end_l=cut_time
        for a in subsection_time:
            sum+=a
            section_start.append(cut_time-sum)
            section_end.append(end_l)
            end_l=cut_time-sum
    elif subsection_name=='M_sub':
        
        end_m=cut_time
        for a in subsection_time:
            sum+=a
            section_start.append(cut_time-sum)
            section_end.append(end_m)
            end_m=cut_time-sum

    print("Section",section_start,section_end)
    start_indices=[int(l1*50000) for l1 in section_start]
    end_indices=[int(l1*50000) for l1 in section_end]
    return list(reversed(start_indices)),list(reversed(end_indices))

def get_data(filepath):
    dw_handle=dw.open(filepath)
    se8_data=dw_handle['SE8'].series().to_numpy(dtype='float64')
    se8_time= dw_handle['SE8'].series().index.to_numpy()
    se8_time-=se8_time[0]

    df=pd.DataFrame({'Time':se8_time,'SE8':se8_data},columns=['Time','SE8'])
  
    drehzahl= dw_handle['Drehzahl'].series().to_numpy(dtype='float64')

    drehzahl_time= dw_handle['Drehzahl'].series().index.to_numpy()
    drehzahl_time-=drehzahl_time[0]
    # print(drehzahl,len(drehzahl),len(drehzahl_time))
    df1=pd.DataFrame({'Time':drehzahl_time},columns=['Time'])
    df1['Rotational_speed']=drehzahl
   

    f_z= dw_handle['F_Z'].series().to_numpy(dtype='float64')
    Fz_time= dw_handle['F_Z'].series().index.to_numpy()
    Fz_time-=Fz_time[0]

    df2=pd.DataFrame({'Time':Fz_time},columns=['Time'])
    df2['Force']=f_z
    
    df_merge=pd.merge(df1,df2,on='Time',how='left')
    
    merged=pd.merge(df,df_merge,on='Time',how='left')
    merged.fillna(method='ffill',inplace=True)
    
    sampling_rate= len(se8_data)/se8_time[-1]
    fs=1/sampling_rate
    return merged,sampling_rate,fs
def compute(filepath,store_filepath):
    data,sampling_rate,fs=get_data(filepath)
    
    df=pd.read_csv('end_seam_points.csv')
    indices=df[df['Filename']==filepath[8:]].index.values
    print(indices,len(indices),type(indices))
    L_cut=df['L_end_point'].loc[indices[0]]
    M_cut=df['Meander_end_point'].loc[indices[0]]

    feed_rate=16.667
    L_area=np.array([25,68,133])
    L_subsection_list=subsection_duration(L_area,feed_rate)
    
    M_area=np.array([25,49,113,120,115,119,130])
    M_subsection_list=subsection_duration(M_area,feed_rate)

    filename=re.sub('.dxd', '', filepath[8:])
    print(filename)

    print(L_subsection_list,M_subsection_list)
    print(np.sum(L_subsection_list),np.sum(M_subsection_list),L_cut,M_cut)
    
    L_start_index=int(np.around(L_cut-np.sum(L_subsection_list),6)*50000)
    M_start_index=int(np.around(M_cut-np.sum(M_subsection_list),6)*50000)
    print(np.around(L_cut-np.sum(L_subsection_list),6),L_start_index,np.around(M_cut-np.sum(M_subsection_list),6),M_start_index)
    L_subsection_list=L_subsection_list[::-1]
    M_subsection_list=M_subsection_list[::-1]
    
    L_start_indices,L_end_indices=get_subsection_end_points(np.around(data,6),truncate(L_cut,6),np.around(L_subsection_list,6),filepath,'L_sub')
    M_start_indices,M_end_indices=get_subsection_end_points(np.around(data,6),truncate(M_cut,6),np.around(M_subsection_list,6),filepath,'M_sub')
    
    subsection_list=np.zeros(data.shape[0])

    L_subsection_name=['I_1','II_1','III']
    M_subsection_name=['I_2','II_2','IV_1','II_3','IV_2','II_4','IV_3']

    subsection_name=np.zeros(data.shape[0])
    
    for i,d in enumerate(L_subsection_name):
        if i==0:
            np.put(subsection_list,range(L_start_indices[i],L_end_indices[i]+1),1)
        elif i==1:
            np.put(subsection_list,range(L_start_indices[i],L_end_indices[i]+1),2)
        elif i==3:
            np.put(subsection_list,range(L_start_indices[i],L_end_indices[i]+1),3)
    for i,d in enumerate(M_subsection_name):
        if i==0:
            np.put(subsection_list,range(M_start_indices[i],M_end_indices[i]+1),4)
        elif i==1:
            np.put(subsection_list,range(M_start_indices[i],M_end_indices[i]+1),5)
        elif i==3:
            np.put(subsection_list,range(M_start_indices[i],M_end_indices[i]+1),6)
        elif i==4:
            np.put(subsection_list,range(M_start_indices[i],M_end_indices[i]+1),7)
        elif i==5:
            np.put(subsection_list,range(M_start_indices[i],M_end_indices[i]+1),8)
        elif i==6:
            np.put(subsection_list,range(M_start_indices[i],M_end_indices[i]+1),9)
        elif i==7:
            np.put(subsection_list,range(M_start_indices[i],M_end_indices[i]+1),10)
    data['Subsection']=subsection_list
    
    annotation=np.zeros(data.shape[0])
    annotation=get_annotation(filename,feed_rate,fs,annotation,L_start_index,M_start_index)
    data['label']=annotation
    # dict={'Time':se8_time,'SE8':se8_data,'Rotational_speed':rotational_speed,'Force':force,'Subsection':subsection_name,'label':annotation}
    # Combined_data=pd.DataFrame(dict)
    print(annotation)
    print(store_filepath+'/'+filename+'.csv')
    data.to_csv(store_filepath+'/'+filename+'.csv',index=False)

def compute_raw_csv(directory,store_path):
    '''
        Load SE8, rotational and force data from dxd file
        Input:
            directory: location from where dxd files are read
            store_path: location where .csv files would be saved   
    '''
    for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith('.dxd'):
                    filepath=os.path.join(root,file)
                    print(filepath)
                    compute(filepath,store_path)

def compute_csv_file(filepath,store_path):
    '''
        Load SE8, rotational and force data from dxd file
        Input:
            filepath: read individual dxd file 
            store_path: location where .csv files would be saved   
    '''
    compute(filepath,store_path)

# def compute_raw_csv(directory,store_path):
#     '''
#         Load SE8, rotational and force data from dxd file
#         Input:
#             directory: location from where dxd files are read
#             store_path: location where .csv files would be saved   
#     '''
#     for root, dirs, files in os.walk(directory):
#             for file in files:
#                 if file.endswith('.dxd'):
#                     filepath=os.path.join(root,file)
#                     print(filepath)
#                     loading(filepath,store_path)

# def compute_csv_file(filepath,store_path):
#     '''
#         Load SE8, rotational and force data from dxd file
#         Input:
#             filepath: read individual dxd file 
#             store_path: location where .csv files would be saved   
#     '''
#     compute(filepath,store_path)
#     # return loading(filepath,store_path)

# def loading(filepath, store_filepath='./'):
#     '''
#         Load SE8, Force and rotational speed data from dxd files
#         Input:
#             filepath: file to read
#             store_filepath: path to store file
#         Stores data to csv file at store_filepath location
#     '''
#     dw_handle=dw.open(filepath)
    
#     if store_filepath=='./data':
#         filename=re.sub('.dxd', '', filepath)
#     else:
#         filename=re.sub('.dxd', '', filepath[8:])

#     # load SE8
#     se8_data=dw_handle['SE8'].series().to_numpy(dtype='float64')
#     se8_time= dw_handle['SE8'].series().index.to_numpy()
#     se8_time-=se8_time[0]

#     df=pd.DataFrame({'Time':se8_time,'SE8':se8_data},columns=['Time','SE8'])
  
#     drehzahl= dw_handle['Drehzahl'].series().to_numpy(dtype='float64')

#     drehzahl_time= dw_handle['Drehzahl'].series().index.to_numpy()
#     drehzahl_time-=drehzahl_time[0]
#     print(drehzahl,len(drehzahl),len(drehzahl_time))
#     df1=pd.DataFrame({'Time':drehzahl_time},columns=['Time'])
#     df1['Rotational_speed']=drehzahl
   

#     f_z= dw_handle['F_Z'].series().to_numpy(dtype='float64')
#     Fz_time= dw_handle['F_Z'].series().index.to_numpy()
#     Fz_time-=Fz_time[0]

#     df2=pd.DataFrame({'Time':Fz_time},columns=['Time'])
#     df2['Force']=f_z
    
#     df_merge=pd.merge(df1,df2,on='Time',how='left')
    
#     merged=pd.merge(df,df_merge,on='Time',how='left')
#     merged.fillna(method='ffill',inplace=True)

#     sampling_rate= len(se8_data)/se8_time[-1]
#     fs=1/sampling_rate

#     # get cuttime
#     L_cut,M_cut=compute_endpoints(filepath)

#     # # get cut time indices (why???) 
#     # L_cut_indices=round_up(L_cut/fs)
#     # M_cut_indices=round_up(M_cut/fs)

#     feed_rate=16.67

#     # read_subsection=pd.read_csv('subsection_end_point.csv')

#     L_start,M_start=get_start(L_cut,M_cut,feed_rate,read_subsection)
#     L_start_index=int(round_down(L_start/fs))
#     M_start_index=int(round_down(M_start/fs))

    
#     # subsection_indices=get_subsection_indices(L_cut,M_cut,feed_rate,read_subsection,fs)
    
#     annotation=np.zeros(len(se8_data))
#     annotation=get_annotation(filename,feed_rate,fs,annotation,L_start_index,M_start_index)

    
#     L_subsection_name=['I_1','II_1','III']
#     M_subsection_name=['I_2','II_2','IV_1','II_3','IV_2','II_4','IV_3']

#     subsection_name=np.zeros(len(se8_data))

#     for v in L_subsection_name:
#         index=np.asarray(np.where(((subsection_indices['curve']=='L_curve')&(subsection_indices['section']==v))))
#         # print(index)
#         # index=subsection_indices[subsection_indices['section']=='I_1'].index.values
#         s=int(subsection_indices['start_index'][index[0]]+L_start_index)
#         e=int(subsection_indices['end_index'][index[0]]+L_start_index)
        
#         if v=='I_1':
#             np.put(subsection_name, range(s,e),1)
#         if v=='II_1':
#             np.put(subsection_name, range(s,e),2)
#         if v=='III':
#             np.put(subsection_name, range(s,e),3)
        
#     for v in M_subsection_name:
#         index=np.asarray(np.where(((subsection_indices['curve']=='Maeander')&(subsection_indices['section']==v))))
#         # index=subsection_indices[subsection_indices['section']=='I_1'].index.values
#         s=int(subsection_indices['start_index'][index[0]]+M_start_index)
#         e=int(subsection_indices['end_index'][index[0]]+M_start_index)
        
#         if v=='I_2':
#             np.put(subsection_name, range(s,e),4)
#         if v=='II_2':
#             np.put(subsection_name, range(s,e),5)
#         if v=='IV_1':
#             np.put(subsection_name, range(s,e),6)
#         if v=='II_3':
#             np.put(subsection_name, range(s,e),7)
#         if v=='IV_2':
#             np.put(subsection_name, range(s,e),8)
#         if v=='II_4':
#             np.put(subsection_name, range(s,e),9)
#         if v=='IV_3':
#             np.put(subsection_name, range(s,e),10)
#     merged['Subsection']=subsection_name
#     merged['label']=annotation
#     # dict={'Time':se8_time,'SE8':se8_data,'Rotational_speed':rotational_speed,'Force':force,'Subsection':subsection_name,'label':annotation}
#     # Combined_data=pd.DataFrame(dict)
#     print(store_filepath+'/'+filename+'.csv')
#     merged.to_csv(store_filepath+'/'+filename+'.csv',index=False)

# def truncate(value,n):
#     return math.floor(value * 10 ** n) / 10 ** n
# def subsection_duration(area,feed_rate):
#     subsection_time=area/feed_rate
#     return subsection_time

# def compute_endpoints(filepath):
#     '''
#     get L and M curve endpoints in seconds, eg. L ends at 25.10 second and M ends at 81.23 
#     '''
    
#     dewefile_handle = dw.open(filepath)

#     z_pos = dewefile_handle["Rob_Pos_Z"].series().to_numpy(dtype='float')
#     time_z_pos = dewefile_handle["Rob_Pos_Z"].series().index
#     time_z_pos -= time_z_pos[0]

#     fs_z_pos = dewefile_handle["Rob_Pos_Z"].number_of_samples / time_z_pos[-1]
    
#     endpoint_L, endpoint_Maeander = get_seam_endpoints(z_pos, fs_z_pos,time_z_pos)

#     # print("{:.2f}, {:.2f}".format(endpoint_L, endpoint_Maeander))
#     return np.round(endpoint_L,5),np.round(endpoint_Maeander,5)
    
# def get_seam_endpoints(z_pos, fs,time_z_pos):
#     """Function that gets the endpoints of the different weld seams. The end is defined as the moment as the tool is moved
#     up and released out of the metal sheet. For this purpose the signal is filtered by a FIR Filter too smooth the
#     stair shaped signal of the roboter position in z. After this the second derivate of the position is calculated
#     to get the first point of steppping out of the sheet.

#     The signals are plotted and the result can be validated by visual control.

#     The functions returns the times of the end of welds."""
#     # creating and applying the fir filter
#     taps = 100
#     fir_filt = signal.firwin(taps, 1, fs=fs)
#     filtered = signal.lfilter(fir_filt, 1.0, z_pos)

#     # fir filter phase shifts the signal. delay determines the shift in seconds
#     delay = 0.5 * (taps - 1) / fs

#     # defining start point to look for the peak which correspondends to the out stepping of the tool
#     start_offset = int(5 * fs)

#     # getting the peaks
#     peaks, properties = signal.find_peaks(np.gradient(np.gradient(filtered[start_offset:])), height=[0.002, 0.006],
#                                           prominence=.004)

#     peaks += start_offset
    
#     # check if exactly two end points were found. If not exactly twi points were found an exception is raised
#     if len(peaks) > 2:
#         print("more than two maximums found. Getting endpoints aborted")
#         raise
#     elif len(peaks) < 2:
#         print("not enough maximums found")
#         raise
    
#     L_cut_time=time_z_pos[peaks[0]]
#     M_cut_time=time_z_pos[peaks[1]]
#     # return time_z_pos[peaks[0]], time_z_pos[peaks[1]]
#     return L_cut_time,M_cut_time
