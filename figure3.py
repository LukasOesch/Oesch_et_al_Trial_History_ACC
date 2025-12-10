# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 23:05:13 2025

@author: Lukas Oesch
"""

from chiCa import * #Mke sure to be inside the chiCa folder
import chiCa
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from labdatatools import *
import os
import matplotlib.pyplot as plt
import matplotlib
from glob import glob

#%%-------First, define the sessions to fit.

# As always set your basis data directory
subject_data = dict()
base_dir = get_labdata_preferences()['paths'][0] #Retrieve the base directory
## Alternatively set the base directory manualy:
#base_dir = 'my base directory'

session_collection = ['LO028/20220616_145438',
            'LO032/20220830_121448',
            'LO032/20220905_123313',
            'LO032/20220907_150825',
            'LO032/20220909_144008',
            'LO032/20220923_135753',
            'LO051/20230322_161052',
            'LO051/20230329_161431',
            'LO051/20230427_163356',
            'LY008/20230405_172520',
            'LY008/20230425_105313',
            'LO068/20230831_132717',
            'LO068/20230905_115256',
            'LO068/20230906_140350',
            'LO068/20230911_154649',
            'LO069/20240102_125134',
            'LO069/20231212_135211',
            'LO069/20231227_105006',
            'LO074/20240731_165357',
            'LO074/20240808_120632',
            'LO067/20240209_105931',
            'LO073/20240814_105301',
            'LO073/20240815_110810']

sessions = [os.path.join(base_dir, x) for x in session_collection]

#%%----Re-generate the trial phase timestamps and load the data

#This is a slightly tedious way to recover the set 30 fps frame rate...
session_dir = sessions[0]
trial_alignment_file = glob(session_dir + '/analysis/*miniscope_data.npy')[0]
frame_rate = int(np.load(trial_alignment_file, allow_pickle = True).tolist()['frame_rate'])

#Generate the alignment timepoints 
aligned_to = ['outcome_presentation', 'outcome_end', 'PlayStimulus', 'DemonWaitForResponse']
time_frame = [np.array([round(0*frame_rate), round(1*frame_rate)+1], dtype=int),
                  np.array([round(-1*frame_rate), round(0.3*frame_rate)+1], dtype=int),
                  np.array([round(-0.5*frame_rate), round(1*frame_rate)+1], dtype=int),
                  np.array([round(-0.2*frame_rate), round(0.3*frame_rate)+1], dtype=int)]#Get cv rsq for single variable models
#Make a list containing the indices for the different trial phases
insertion_index = np.zeros(len(time_frame)+1)
insertion_index[1] = time_frame[0][1] - time_frame[0][0]
idx_list = [np.arange(insertion_index[1],dtype=int)]
for k in range(1,len(time_frame)):
        insertion_index[k+1] = int(insertion_index[k] + time_frame[k][1] - time_frame[k][0])
        idx_list.append(np.arange(insertion_index[k],insertion_index[k+1], dtype=int))
     
#Initialize lists and start loading the data
single_var = [] #Holds the entire model data
from_session = [] #Specifies the session the models are from
original_neuron_id = [] #Retain the original index of each neuron in their respective session
encoding_model_name = 'RidgeCV_encoding_models_ITI_partitioned_Fig3.npy'
counter = 0
for session_dir in sessions:
    enc = np.load(os.path.join(session_dir, 'analysis', encoding_model_name),allow_pickle = True).tolist()
    single_var.append(enc)
    from_session.append(np.zeros([enc['r_squared'][0].shape[0]])  + counter)
    counter = counter+1
    original_neuron_id.append(np.arange(enc['r_squared'][0].shape[0]))
    
original_neuron_id = np.hstack(original_neuron_id)
from_session = np.hstack(from_session)

#Do sanity check here: If some of the timecourse R2 values exceed 1 then this
#might indicate issues with the fitting or some activity outliers.
unlikely_r = [] #Session id, shuffle round id
for k in range(len(single_var)):
    for n in range(len(single_var[k]['r_squared_timecourse'])):
        if np.max(single_var[k]['r_squared_timecourse'][n]) > 1:
            unlikely_r.append(np.array([k,n]))

#Get the different R2 values
cvR_single = [] #The maximal explained variance
dR_single = [] #The unique explained variance
cvR_timecourse_single = [] #Maximal explained variance when evaluated at
#every timepoint. This is obtained by resorting the timestamps and then only 
#calculating the R2 at every timepoint.
dR_timecourse_single = []
cvCorr_timecourse_single = [] #The corr here indicates the squared correlation coefficient, as used in Musall & Kaufman et al., 2019
dCorr_timecourse_single = []

model_num = int((len(enc['r_squared']) - 2) / 2)
for k in range(2, model_num + 2):
    c = []
    d = []
    ct = []
    dt = []
    ccort = []
    dcort =[]

    for n in range(len(single_var)):
        #For all the maximal explained variances we subtract the time-regressor only
        #model (index 1) from the model in question. This means that we remove the variance explainable by across-trial
        #fluctuations that are identical for all trials.
        c.append(single_var[n]['r_squared'][k] - single_var[n]['r_squared'][1])
        ct.append(single_var[n]['r_squared_timecourse'][k] - single_var[n]['r_squared_timecourse'][1])
        ccort.append(single_var[n]['corr_timecourse'][k] - single_var[n]['corr_timecourse'][1])
        
        #For the unique variance we calculate the difference between the full model (index 0)
        #and the model in question to see how much variance cannot be accounted for
        #by any of the other included variables.
        d.append(single_var[n]['r_squared'][0] - single_var[n]['r_squared'][k + model_num])
        dt.append(single_var[n]['r_squared_timecourse'][0] - single_var[n]['r_squared_timecourse'][k + model_num])
        dcort.append(single_var[n]['corr_timecourse'][0] - single_var[n]['corr_timecourse'][k + model_num])

    cvR_single.append(np.hstack(c).T)
    dR_single.append(np.hstack(d).T)
    cvR_timecourse_single.append(np.hstack(ct))
    dR_timecourse_single.append(np.hstack(dt))
    cvCorr_timecourse_single.append(np.hstack(ccort))
    dCorr_timecourse_single.append(np.hstack(dcort))

# Also load the full model cvR2, dR2 and their timecourses
full_single = []
full_timecourse_single = []
full_corr_timecourse_single = []
betas = single_var[0]['betas'][0]
for k in range(len(single_var)):
    full_single.append(single_var[k]['r_squared'][0])
    full_timecourse_single.append(single_var[k]['r_squared_timecourse'][0])
    full_corr_timecourse_single.append(single_var[k]['corr_timecourse'][0])
    if k > 0:
        betas = np.vstack((betas,single_var[k]['betas'][0]))
full_single = np.hstack(full_single).T
full_timecourse_single = np.hstack(full_timecourse_single)
full_corr_timecourse_single = np.hstack(full_corr_timecourse_single)

labels = ['Choice', 'Outcome', 'Previous choice x outcome', 'Stimulus events',
          'Center poke', 'Left poke', 'Right poke',
          'Pitch', 'Roll', 'Yaw', 'Chest position', 'Video SVD', 'Video motion energy SVD']

#%%------Figure 3B and C: Make boxplots of average maximal- and unique explaiend variance----------

#Identify subjects in the sessions
subj_str = np.array([os.path.split(os.path.split(session_dir)[0])[1] for session_dir in sessions])
uni = np.unique(subj_str)
subj_ses = []
for k in uni:
    subj_ses.append(np.array([x for x in range(subj_str.shape[0]) if subj_str[x] == k]))

#Calculate mean and sem full mdodel accuracy and print average over all mice
ses_av = np.array([np.mean(x['r_squared'][0]) for x in single_var])
subj_av = np.array([np.mean(ses_av[subj_str==x]) for x in uni])
subj_sem = np.std(subj_av) / np.sqrt(uni.shape[0])
print(f'Mean +/- sem full model variance is: {np.mean(subj_av)} +/- {subj_sem}')

#Get the average dvR2 and dR2 per mouse for the different variables
mean_cvR = []
mean_dR = []
for k in range(len(uni)):
    tmp = []
    tmp_d = []
    for n in subj_ses[k]:
        tmp.append(np.mean(np.vstack(cvR_single)[:,from_session == n],axis=1))
        tmp_d.append(np.mean(np.vstack(dR_single)[:,from_session == n],axis=1))
    if len(tmp) > 1:
        mean_cvR.append(np.mean(np.squeeze(tmp),axis=0))
        mean_dR.append(np.mean(np.squeeze(tmp_d),axis=0))
    else:
        mean_cvR.append(np.squeeze(tmp))
        mean_dR.append(np.squeeze(tmp_d))
        
gray = '#858585' 
ax = plt.figure(figsize=[10, 4.8]).add_subplot()
widths = 0.7
fancy_boxplot(ax, np.squeeze(mean_cvR), None, widths = widths, labels = labels, y_label = 'cvR\u00b2')
ax.axhline(0, linestyle='--', color='k', linewidth=0.5)
jitter = (0.5 - np.random.rand(len(mean_cvR))) * (widths*0.8)
for k in range(len(mean_cvR)):
    ax.scatter(np.arange(1, model_num + 1) + jitter[k], np.squeeze(mean_cvR)[k,:],c=gray,s=10)
separate_axes(ax)
ax.spines['bottom'].set_bounds([0.5, len(cvR_single)+0.5])

ax = plt.figure(figsize=[10, 4.8]).add_subplot()
widths = 0.7
fancy_boxplot(ax, np.squeeze(mean_dR), None, widths = widths, labels = labels, y_label = 'cvR\u00b2')
ax.axhline(0, linestyle='--', color='k', linewidth=0.5)
jitter = (0.5 - np.random.rand(len(mean_cvR))) * (widths*0.8)
for k in range(len(mean_dR)):
    ax.scatter(np.arange(1, model_num + 1) + jitter[k], np.squeeze(mean_dR)[k,:],c=gray,s=10)
separate_axes(ax)
ax.spines['bottom'].set_bounds([0.5, len(cvR_single)+0.5])

##############################################################################
######-----Optional: Sort neurons by overall explained variance and plot cvR2
# #and dR2 for different variables -> see Musall and Kaufman et al., 2019, Fig. 7e
# sort_idx = np.flip(np.argsort(full_single))
# colors = ['#444444', '#e26f2e', '#9f89dd', '#2a99cd','#5a8c45']
# data = [full_single] + cvR_single
# line_labels = ['full model'] + labels
# ax = plt.figure(figsize=[10, 4.8]).add_subplot()
# model_list = [0,1,3,4,12,13]
# for k in range(len(model_list)):
#     ax.plot(data[model_list[k]][sort_idx], linewidth=1, alpha=0.8, label = line_labels[model_list[k]])
# ax.set_ylim([-0.05, 1])
# ax.set_xlabel('Neuron number')
# ax.set_ylabel('cvR\u00b2')
# separate_axes(ax)
# ax.spines['left'].set_bounds([0,1])
# plt.legend()

# sort_idx = np.flip(np.argsort(full_single))
# colors = ['#e26f2e', '#9f89dd', '#2a99cd','#5a8c45']
# data = dR_single
# ax = plt.figure(figsize=[10, 4.8]).add_subplot()
# model_list = [0,2,3,11,12]
# for k in range(len(model_list)):
#     ax.plot(data[model_list[k]][sort_idx], linewidth=1, alpha=0.8, label = labels[model_list[k]])
# ax.set_xlabel('Neuron number')
# ax.set_ylabel('\u0394R\u00b2')
# separate_axes(ax)
# plt.legend()


##Plot the same but only including the session with the most recorded neurons
# ses_idx = [0,1,7,9,14,17,18,20,22] #Include the session with most recorded neurons for each mouse
# use_neuron = np.array([x for x in range(from_session.shape[0]) if from_session[x] in ses_idx])
# sort_idx = np.flip(np.argsort(full_single[use_neuron]))

# colors = ['#e26f2e', '#9f89dd', '#2a99cd','#5a8c45']
# data = dR_single
# ax = plt.figure(figsize=[10, 4.8]).add_subplot()
# model_list = [2,0,10,11]
# for k in range(len(model_list)):
#     ax.plot(data[model_list[k]][use_neuron][sort_idx], linewidth=1, alpha=0.8, label = labels[model_list[k]])
# ax.set_xlabel('Neuron number')
# ax.set_ylabel('cvR\u00b2')
# separate_axes(ax)
# plt.legend()
##########################################################################

#%%-------Supplementary Figure 3D and E: Plot cvR2 and dR2 over time-----

time_cvR = []
time_dR = []
for k in range(len(uni)):
    tmp = []
    tmp_d = []
    for n in subj_ses[k]:
        tmp.append(np.mean(np.squeeze(cvR_timecourse_single)[:,:,from_session == n],axis=-1))
        tmp_d.append(np.mean(np.squeeze(dR_timecourse_single)[:,:,from_session == n],axis=-1))
    if len(tmp) > 1:
        time_cvR.append(np.mean(np.squeeze(tmp),axis=0))
        time_dR.append(np.mean(np.squeeze(tmp_d),axis=0))
    else:
        time_cvR.append(np.squeeze(tmp))
        time_dR.append(np.squeeze(tmp_d))
        
plot_models = [2,11,10,0]
cols = ['#765e9c', '#629bc6',  '#98446f', '#bc6238']
line_labels = ['trial history', 'video svd', 'chest position', 'choice']
for dat in [time_cvR, time_dR]:
    data = [np.squeeze(dat)[:,k,:].T for k in plot_models]
    ax = plt.figure(figsize=[10, 4.8]).add_subplot()
    plot_timecourse(ax, data, frame_rate, idx_list, spacer=4, colors=cols, line_labels = line_labels, x_label = 'Time (s)', y_label = 'cvR\u00b2')
    separate_axes(ax)
    
#%%-----Figure 3D, E and F: Scatter plots of unique trial history variance
#-------agains other variables

#Find number of cells per session
cell_num = []
for session_dir in sessions:
    trial_alignment_file = glob(session_dir + '/analysis/*miniscope_data.npy')[0]
    miniscope_data = np.load(trial_alignment_file, allow_pickle = True).tolist()
    cell_num.append(miniscope_data['F'].shape[0])

#Identify individual subejects and their sessions
subj_str = np.array([os.path.split(os.path.split(x)[0])[1] for x in sessions])
unique_subj = np.unique(subj_str)

#Finally for each subject find the session with most recorded neurons
ses_idx = []
for subj in unique_subj:
    tmp = np.where(subj == subj_str)[0]
    ses_idx.append(tmp[np.argmax(np.array(cell_num)[tmp])])
#ses_idx = [0,1,7,9,14,17,18,20,22] #Include the session with most recorded neurons for each mouse

#Find the indices of the neurons to show
use_neuron = np.array([x for x in range(from_session.shape[0]) if from_session[x] in ses_idx])

reg_idx = [11, 10, 0, 9] #Video SVD, Chest point tuning, choice, yaw tuning
cols = ['#629bc6',  '#98446f', '#bc6238', '#c8a91e']
ylims = [-0.01, np.max(np.squeeze(dR_single)[reg_idx,:][:,use_neuron]+ 0.01)] #The regresssor of interest
xlims = [-0.01, 0.35]
ax_list = []
for k in range(len(reg_idx)):
    #ax_list.append(plt.figure().add_subplot(111,aspect='equal'))
    ax_list.append(plt.figure(figsize=[5.5, 4.8]).add_subplot(111))
    ax_list[-1].scatter(dR_single[2][use_neuron], dR_single[reg_idx[k]][use_neuron], c=cols[k], s = 20, edgecolor='w', linewidth=0.5)
    ax_list[-1].set_xlim(xlims)
    ax_list[-1].set_ylim(ylims)
    ax_list[-1].set_title(f'{labels[reg_idx[k]]}')
    separate_axes(ax_list[-1])
    ax_list[-1].set_xlabel('Unique variance for trial history')
    ax_list[-1].set_ylabel(f'Unique variance for {labels[reg_idx[k]]}')
    
############################################################################
#####-----Optional label some neurons with mixed selectivity on the scatter plot    

# #We'll use session 18 to select example neurons
# example_n = np.array([69,146,195]) #These are the indices with respect to the session itself
# example_on_all = np.where(from_session==18)[0][0] + example_n

# #Now draw the circles
# for n in example_on_all:
#     for k in range(len(ax_list)):
#         circ = plt.Circle([dR_single[2][n], dR_single[reg_idx[k]][n]],0.005, fill=False, color ='k')
#         ax_list[k].add_patch(circ)
        
##############################################################################