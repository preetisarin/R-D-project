from copyreg import clear_extension_cache
import statistics
from scipy import signal
import acoustics.signal as S 
import numpy as np
import pandas as pd
import scipy
import csv
import os
import re
import more_itertools as mit
from filter_data import filtering,filtering_1
import scipy.signal
from scipy.stats import entropy

class Feature():
    def Peak(self,data):
        return max(np.abs(data))

    def energy(self,signal):
        # summation=0
        # # print("Signal length ",len(signal))
        # for s in signal:
        #     summation+=s**2
        # return summation
        return np.sum(np.abs(signal) ** 2)

    def impulse_factor(self,data):
        return max(np.abs(data))/np.average(data)

    def kurtosis_factor(self,data):
        denominator=np.power(self.energy(data)/len(data),2)
        me=np.mean(data)
        var=np.var(data)
        numerator=self.kurtosis(data)
        result=numerator/denominator
        return result

    def clearance_factor(self,data):
        numerator=self.Peak(data)
        den=0
        for d in data:
            den+=np.sqrt(np.abs(d))
        result=numerator/(den/len(data))
        return result

    def square_root_mean(self,data):
        sum=0
        for d in data:
            sum+=np.sqrt(np.abs(d))
        result=np.power(sum/len(data),2)
        return result

    def root_mean_square(self,data):
        # sum=0
        # for d in data:
        #     sum+=np.power(d,2)
        # result=np.sqrt(sum/len(data))
        # return result
        return np.sqrt(np.sum(np.array(data)**2)/len(data))

    def kurtosis(self,data):
        # numerator=0
        # me=np.mean(data)
        # var=np.var(data)
        # for d in data:
        #     numerator+=np.power((d-me)/var,4)
        # result=(numerator/len(data))
        # return result
        return scipy.stats.kurtosis(data)

    def margin_factor(self,data):
        denominator=self.square_root_mean(data)
        numerator=self.Peak(data)
        result=numerator/denominator
        return result

    def crest_factor(self,data):
        numerator=self.Peak(data)
        denominator=self.root_mean_square(data)
        result=numerator/denominator

        return result

    def skewness(self,data):
        # mean=np.mean(data)
        # var=np.var(data)
        # for d in data:
        #     sum+=np.power((d-mean)/var,3)
        # skewness=sum/len(data)
        # return skewness
        return scipy.stats.skew(data)

    def peak_to_peak_value(self,data):
        # ppv=np.max(data)-np.min(data)
        # return ppv
        return np.abs(np.max(data)-np.min(data))

    def shape_factor(self,data):
        numerator=self.root_mean_square(data)
        denominator=0
        for d in data:
            denominator+=np.abs(d)
        result=numerator/(denominator/len(data))
        return result

    def max(self,data):
        return np.max(data)
    
    # def min(self,data):
    #     return np.min(data)

    def mean(self,data):
        return np.mean(data)

    def variance(self,data):
        return np.var(data)
    
    def distance(self,data):
        diff_sig = np.diff(data).astype(float)
        return np.sum([np.sqrt(1 + diff_sig ** 2)])

    def zero_cross(self,data):
        return len(np.where(np.diff(np.sign(data)))[0])

        
    def compute_time_domain_feature(self,data):
        
        func_list={"m1":self.Peak,
                   "m2":self.energy,
                   "m3":self.impulse_factor,
                   "m4":self.kurtosis_factor,
                   "m5":self.clearance_factor,
                   "m6":self.square_root_mean,
                   "m7":self.root_mean_square,
                   "m8":self.kurtosis,
                   "m9":self.margin_factor,
                   "m10":self.crest_factor,
                   "m11":self.skewness,
                   "m12":self.peak_to_peak_value,
                   "m13":self.shape_factor,
                   "m14":self.mean,
                   "m15":self.variance,
                   "m16":self.distance,
                   "m17":self.zero_cross}
        # "m14":self.max,
        # func_list={"m1": self.Peak, "m2": self.energy,"m3": self.impulse_factor,"m4": self.root_mean_square,\
        #             "m5":self.kurtosis, "m6":self.skewness,"m7":self.peak_to_peak_value,"m8":self.max,"m9":self.min,\
        #             "m10":self.mean,"m11":self.variance,"m12":self.distance,"m13":self.zero_cross}

        features=[]
        for key, value in func_list.items():
            # print(key)
            result=value(data)
            if type(result)==list:
                features.extend(result)
            else:
                features.append(result)
        # print(len(features))
        return features

    def compute_filepath(self,directory,frame_length,overlap,storing_path='./time/'):
        for root, dirs, files in os.walk(directory):
            # print(root,dirs,files)
                for file in files:
                    if file.endswith('.csv'):
                        filepath=os.path.join(root,file)

                        # print(filepath)
                        self.extract_feature(filepath,frame_length,overlap,storing_path)
    
    def extract_feature(self,filepath,frame_length,overlap,storing_path='./time/'):
        feature=[]
        df=pd.read_csv(filepath)
                
        sampling_rate=int(len(df['SE8'])/df['Time'].iloc[-1])

        window_size=int(frame_length*sampling_rate)
        stride=int(((overlap/100)*frame_length)*sampling_rate)
        # print("stride",stride,window_size)

        print(filepath,len(df['SE8']))

        filtered_data=np.array(df['SE8'])
        
        subsection_name=[1,2,3,4,5,6,7,8,9,10]
        sub_name=['I_1','II_1','III','I_2','II_2','IV_1','II_3','IV_2','II_4','IV_3']
        for i,val in enumerate(subsection_name):
            storing_filename=re.sub('_\d','',sub_name[i])+'.csv'
            # print(storing_filename)
                                
            indices=np.where(df['Subsection']==val)
            
            se8_subdata=filtered_data[indices[0][0]:indices[0][-1]]
            label_subdata=df['label'].iloc[indices[0][0]:indices[0][-1]]
            rotational_subdata=df['Rotational_speed'].iloc[indices[0][0]:indices[0][-1]]
            force_subdata=df['Force'].iloc[indices[0][0]:indices[0][-1]]

            for i in range(0,len(se8_subdata)-(window_size)+1,stride):
                # print(i)
                f=[]
                start=i
                end=i+window_size
                # if window_size>len(se8_subdata[start:end]):
                #     end=df.shape[0]
                #     data=np.array([0 if i is None else i for i in mit.take(window_size, mit.padnone(se8_subdata[start:end]))])
                # else:
                data=se8_subdata[start:end]
                
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
                f=self.compute_time_domain_feature(data)
                f.append(np.mean(rotational_subdata[start:end]))
                f.append(np.mean(force_subdata[start:end]))
                f.append(max_value)
                # print(storing_path+storing_filename+'.csv')
                file = open(storing_path+storing_filename, 'a+', newline ='')
                header_added=False
                with file:
                    write=csv.writer(file)
                    write.writerow(f)

# def extract_feature(self,filepath,frame_length,overlap,storing_path='./time/'):
#         feature=[]
#         df=pd.read_csv(filepath)
#         print(filepath,len(df['SE8']))      
#         sampling_rate=int(len(df['SE8'])/df['Time'].iloc[-1])

#         window_size=int(frame_length*sampling_rate)
#         stride=int(((overlap/100)*frame_length)*sampling_rate)
#         # print("stride",stride,window_size)

#         filtered_data=filtering(np.array(df['SE8']),sampling_rate)
#         # filtered_data=filtering_1(np.array(df['SE8']),sampling_rate)
        
#         subsection_name=[1,2,3,4,5,6,7,8,9,10]
#         sub_name=['I_1','II_1','III','I_2','II_2','IV_1','II_3','IV_2','II_4','IV_3']
#         for i,val in enumerate(subsection_name):
#             storing_filename=re.sub('_\d','',sub_name[i])+'.csv'
#             # print(storing_filename)
                                
#             indices=np.where(df['Subsection']==val)
            
#             se8_subdata=filtered_data[indices[0][0]:indices[0][-1]]
#             label_subdata=df['label'].iloc[indices[0][0]:indices[0][-1]]
#             rotational_subdata=df['Rotational_speed'].iloc[indices[0][0]:indices[0][-1]]
#             force_subdata=df['Force'].iloc[indices[0][0]:indices[0][-1]]

#             for i in range(0,len(se8_subdata)-(window_size)+1,stride):
#                 # print(i)
#                 f=[]
#                 start=i
#                 end=i+window_size
#                 # if window_size>len(se8_subdata[start:end]):
#                 #     end=df.shape[0]
#                 #     data=np.array([0 if i is None else i for i in mit.take(window_size, mit.padnone(se8_subdata[start:end]))])
#                 # else:
#                 data=se8_subdata[start:end]
                
#                 # check if all labels are same or not. 
#                 # if labels are not same then, take maximum value as label
                
#                 unique, counts = np.unique(label_subdata[start:end], return_counts=True)
#                 d=dict(zip(unique, counts))
#                 # print(filepath,start,end,d)
#                 if len(d)>1 and (unique[0]==unique[1] or unique[0]>unique[1] or unique[1]>unique[0]):
#                     max_value=1
#                 else:
#                     max_value=unique[0]

#                 # print(len(df['SE8'].iloc[start:end]),max_value)
#                 # add features here
#                 f=self.compute_time_domain_feature(data)
#                 f.append(np.mean(rotational_subdata[start:end]))
#                 f.append(np.mean(force_subdata[start:end]))
#                 f.append(max_value)
#                 # print(storing_path+storing_filename+'.csv')
#                 file = open(storing_path+storing_filename, 'a+', newline ='')
#                 header_added=False
#                 with file:
#                     write=csv.writer(file)
# #                     write.writerow(f)
#     def compute_filepath1(self,directory,frame_length,overlap,storing_path='./time/'):
#         for root, dirs, files in os.walk(directory):
#             # print(root,dirs,files)
#                 for file in files:
#                     if file.endswith('.csv'):
#                         filepath=os.path.join(root,file)

#                         # print(filepath)
#                         self.extract_feature_without_filtering(filepath,frame_length,overlap,storing_path)