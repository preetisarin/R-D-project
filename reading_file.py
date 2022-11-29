import dwdatareader as dw
import os
from scipy.io import wavfile
import math 
import numpy as np
import pandas as pd
import csv

class Load():

    def __init__(self):
        print("Load constructor")
    def load_file(self,filepath, type):
        if type=='wave':
            samplerate, data = wavfile.read(filepath)
            duration=len(data)/samplerate
            
            time=[i for i in np.arange(0,duration,0.00001)]
            return samplerate,data,time
        elif type=='dxd':
            dewefile_handle = dw.open(filepath)    
            se8 = dewefile_handle["SE8"].series().to_numpy(dtype='float16')
            time_se8 = dewefile_handle["SE8"].series().index.to_numpy()
            time_se8-=time_se8[0]
            return se8,time_se8
        

    def read_filenames(self,directory,type):
        filenames=[]
        for root, dirs, files in os.walk(directory):
            for file in files:
                if type=='wave':
                    if file.endswith('.wav'):
                        filepath=os.path.join(root,file)
                        filenames.append(filepath)
                elif type=='dxd':
                    if file.endswith('.dxd'):
                        filepath=os.path.join(root,file)
                        filenames.append(filepath)
                else:
                    print("wrong file extension")
                    return None
        return filenames

class Split():

    def __init__(self):
        print("all variables to be defined")

    def split(self,data, start,end):
        split_data=[]
        for i in range(start,end):
            split_data.append(data[i])
            
        #d=list(data)
        #split_data=d[start:end]
        return split_data
        
    def truncate(self,value,n):
        return math.floor(value * 10 ** n) / 10 ** n

    def get_index(self,data,start,end):
        start_index=0
        end_index=0
        for index,value in enumerate(list(data)):
            if self.truncate(value,4)==start:
                start_index=index
            if self.truncate(value,4)==end:
                end_index=index
        return start_index,end_index
        
    # get total length of L cut
    def L_total_duration(self,feed_rate):
        #L_area_name=['I_1','II_1','buffer_L1','III_K','III','buffer_L2']
        L_area=np.array([25,68,10,28,90,5]) # all dimensions in mm
        
        L_time=np.round(L_area/feed_rate,5)
        return sum(L_time)

    # get total length of M cut
    def M_total_duration(self,feed_rate):
        #M_area_name=['I_2','II_2','buffer_M1','IV_1_K','IV_1','buffer_M2','II_3_K',
        #        'II_3','buffer_M3','IV_2_K','IV_2','buffer_M4','II_4_K','II_4','buffer_M5','IV_3_K','buffer_M6','V','buffer_M7']
        M_area=np.array([25,49,10,41,62,10,48,62,10,43,62,10,48,61,10,34,10,71,5]) # all dimensions in mm

        M_time=np.round(M_area/feed_rate,5)
        return sum(M_time)


    def get_cut_time(self,filename, type):
        L_cut_time=0
        M_cut_time=0
        if type=='wave':
            with open('welding_cut_time.csv', newline='') as f:
                reader = csv.reader(f)
                wave_split_time = list(reader)
            wave_split_time.pop(0)
            
            for i,f in enumerate(wave_split_time):
                if filename[5:]==f[0]:
                    L_cut_time=self.truncate(float(f[1]),4)
                    M_cut_time=self.truncate(float(f[2]),4)
                    break
        elif type=='dxd':
            with open('CSV/end_seam_points.csv', newline='') as f:
                r = csv.reader(f)
                dxd_split_time = list(r)
            dxd_split_time.pop(0)

            for i,f in enumerate(dxd_split_time):
                if filename[89:]==f[0]:
                    # get L cut time
                    L_cut_time=self.truncate(float(f[1]),4)
                    # get M cut time
                    M_cut_time=self.truncate(float(f[2]),4)
                    break
        else:
            print("wrong file extension")
            return None
        return L_cut_time,M_cut_time

    def start_cut_duration(self,file,type):
        feed_rate=16.667
        L_duration=self.truncate(self.L_total_duration(feed_rate),4)
        M_duration=self.truncate(self.M_total_duration(feed_rate),4)
        
        L_cut,M_cut=self.get_cut_time(file,type)
        # print("L_cut",L_cut,"M_cut",M_cut,"L_duration",L_duration,"M_duration",M_duration)
        L_start=self.truncate(L_cut-L_duration,4)
        M_start=self.truncate(M_cut-M_duration,4)

        # print("L_start",L_start,"M_start",M_start,"L_cut",L_cut,"M_cut",M_cut)
        return L_start,L_cut,M_start,M_cut

        