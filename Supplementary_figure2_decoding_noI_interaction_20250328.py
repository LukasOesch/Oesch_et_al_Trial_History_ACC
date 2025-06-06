#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 27 23:48:53 2025

@author: loesch
"""


from chiCa import *
import chiCa
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon, pearsonr
from labdatatools import *
import os
import matplotlib.pyplot as plt
import matplotlib
from glob import glob
from scipy.ndimage import gaussian_filter1d
from scipy.stats import pearsonr, zscore
import multiprocessing as mp
from time import time
import sys
sys.path.append('C:/Users/Lukas Oesch/Documents/ChurchlandLab/fit_psychometric/fit_psychometric')
sys.path.append('/Users/loesch/Documents/Churchland_lab/chiCa')
from analysis import *
from sklearn.metrics import confusion_matrix

#%%

sessions = ['/Users/loesch/data/LO028/20220616_145438',
 '/Users/loesch/data/LO032/20220830_121448',
 '/Users/loesch/data/LO032/20220905_123313',
 '/Users/loesch/data/LO032/20220907_150825',
 '/Users/loesch/data/LO032/20220909_144008',
 '/Users/loesch/data/LO032/20220923_135753',
 '/Users/loesch/data/LO051/20230322_161052',
 '/Users/loesch/data/LO051/20230329_161431',
 '/Users/loesch/data/LO051/20230427_163356',
 '/Users/loesch/data/LY008/20230405_172520',
 '/Users/loesch/data/LY008/20230425_105313',
 '/Users/loesch/data/LO068/20230831_132717',
 '/Users/loesch/data/LO068/20230905_115256',
 '/Users/loesch/data/LO068/20230906_140350',
 '/Users/loesch/data/LO068/20230911_154649',
 '/Users/loesch/data/LO069/20240102_125134',
 '/Users/loesch/data/LO069/20231212_135211',
 '/Users/loesch/data/LO069/20231227_105006',
 '/Users/loesch/data/LO074/20240731_165357',
 '/Users/loesch/data/LO074/20240808_120632',
 '/Users/loesch/data/LO067/20240209_105931',
 '/Users/loesch/data/LO073/20240814_105301',
 '/Users/loesch/data/LO073/20240815_110810']

#%%-------Model fitting------


use_name = ['Supp_Fig2_decoding_choice_spr.npy' ,'Supp_Fig2_decoding_outcome_spr.npy']
signal_type = 'S'
balancing_mode = 'equal'
cover_all_samples = True
k_folds = 8 #Was 8 before, let's see Folds for cross-validation -> out of necessity!
subsampling_rounds = 20 #Re-drawing of samples from majority class
penalty = 'l2'
normalize = True
#reg = 0.
reg = 10**(np.linspace(-10,1,12))
model_params = {'penalty': penalty, 'solver': 'liblinear', 'inverse_regularization_strength': reg, 'fit_intercept': True, 'normalize': normalize} #Po

for session_dir in sessions:
    trial_alignment_file = glob(session_dir + '/analysis/*miniscope_data.npy')[0]
    miniscope_data = np.load(trial_alignment_file, allow_pickle = True).tolist()  
    frame_rate = miniscope_data['frame_rate']
    
    #Set the times up
    aligned_to = ['outcome_presentation', 'outcome_end', 'PlayStimulus', 'DemonWaitForResponse']
    time_frame = [ np.array([round(0*frame_rate), round(1*frame_rate)+1], dtype=int),
                  np.array([round(-1*frame_rate), round(0.3*frame_rate)+1], dtype=int),
                  np.array([round(-0.5*frame_rate), round(1*frame_rate)+1], dtype=int),
                  np.array([round(-0.2*frame_rate), round(0.3*frame_rate)+1], dtype=int)]
    
    trialdata = pd.read_hdf(glob(session_dir + '/chipmunk/*.h5')[0], '/Data')
    c_data = get_chipmunk_behavior(session_dir)
    
    #For one run the choice will be the labels and the outcome the secondary labels and then vice-versa
    prior_choice = np.array(c_data['prior_choice'])
    prior_outcome = np.array(c_data['prior_outcome'])
    
    valid_trials = np.where(c_data['valid_past'])[0] #Retrieve valid trials with a response before and after 
    #==========One session from LO067 has two unidentified response port out events
    if (os.path.split(session_dir)[1]=='20240209_105931') & (os.path.split(os.path.split(session_dir)[0])[1] == 'LO067'):
        valid_trials = np.delete(valid_trials, [113,115])
    #====================================================
    valid_trials_before = valid_trials-1
    val_trials =  [valid_trials_before]*2 + [valid_trials]*2 #Combine trial identities for the ITI before the new trial and once it has started
    
    if signal_type == 'S':
          signal = gaussian_filter1d(miniscope_data[signal_type].T,1, axis=0, mode='constant', cval=0, truncate=4)
          #Pad the first and last samples with zeros in this condition
    else:
          signal = miniscope_data[signal_type].T #Transpose the original signal so that rows are timepoints and columns cells
      
    #Determine which neurons to include in the analysis
    keep_neuron = np.arange(signal.shape[1])
    Y = []
    for k in range(len(aligned_to)):
            state_start_frame, state_time_covered = find_state_start_frame_imaging(aligned_to[k], trialdata, miniscope_data['frame_interval'], miniscope_data['trial_starts'], miniscope_data['trial_start_time_covered'])                                                                     
            zero_frame = np.array(state_start_frame[val_trials[k]] + time_frame[k][0], dtype=int) #The firts frame to consider
            
            for add_to in np.arange(time_frame[k][1] - time_frame[k][0]):
                Y.append(signal[zero_frame + add_to,:][:, keep_neuron])
    Y = np.squeeze(Y)
    
    #--Now define the labels and secondary labels
    for n in range(len(use_name)):
        if n == 0:
            labels = prior_choice
            secondary_labels = prior_outcome
        elif n == 1:
            labels = prior_outcome
            secondary_labels = prior_choice
            
        #Now the actual fitting
        start_parallel = time() #Measure the time
        par_pool = mp.Pool(mp.cpu_count())
        if secondary_labels is not None:
            output_models = par_pool.starmap(balanced_logistic_model_training,
                                         [(data, labels[valid_trials], k_folds, subsampling_rounds, secondary_labels[valid_trials], model_params, balancing_mode, cover_all_samples
                                           ) for data in Y])
        else:
            output_models = par_pool.starmap(balanced_logistic_model_training,
                                        [(data, labels[valid_trials], k_folds, subsampling_rounds, secondary_labels, model_params, balancing_mode, cover_all_samples
                                          ) for data in Y]) #In this condition secondary labels is just none
    
        stop_parallel = time()
        print('-------------------------------------------')
        print(f'Done fitting the models in {round(stop_parallel - start_parallel)} seconds')
        
        #Repackage the outputs
        decoding_models = []
        for k in output_models:
                    decoding_models.append(k.to_dict('list')) 
        #Use the exact models to predict other timepoints. Also use similar training and testing splits          
        #prediction_accuracy, shuffled_prediction_accuracy, confusion_matrix = cross_decoding_analysis(decoding_models, Y, labels, valid_trials)
        
        out_dict = {'decoding_models': decoding_models,
                    #'prediction_accuracy': prediction_accuracy,
                    #'shuffled_prediction_accuracy': shuffled_prediction_accuracy,
                    #'confusion_matrix': confusion_matrix
                    }
        
        np.save(os.path.join(session_dir, 'analysis', use_name[n]), out_dict)
    
#%%-----load and plot the results


#Get list of indices
insertion_index = np.zeros(len(time_frame)+1)
insertion_index[1] = time_frame[0][1] - time_frame[0][0]
idx_list = [np.arange(insertion_index[1],dtype=int)]
for k in range(1,len(time_frame)):
    insertion_index[k+1] = int(insertion_index[k] + time_frame[k][1] - time_frame[k][0])
    idx_list.append(np.arange(insertion_index[k],insertion_index[k+1], dtype=int))

#Get the subject ID for every session
subj = np.unique([os.path.split(os.path.split(session_dir)[0])[1] for session_dir in sessions])
subj_code = np.zeros([len(sessions)],dtype=int)
subj_string = []
session_string = []
for n in range(len(sessions)):
    for k in range(subj.shape[0]):
        if subj[k] in sessions[n]:
            subj_code[n] = k
            subj_string.append(str(subj[k]))
    session_string.append(os.path.split(sessions[n])[1])
    
#Load the data for choice, outcome and choice x outcome models
file_names = ['Supp_Fig2_decoding_choice_spr.npy' ,'Supp_Fig2_decoding_outcome_spr.npy','Fig2_decoding_previous_choice_outcome_combination_spr.npy']
all_decoding_accuracy = np.zeros([133, len(sessions), len(file_names)]) * np.nan
confusion_matrices = []
for ses in range(len(sessions)):
        for cond in range(len(file_names)):
            dec = np.load(os.path.join(sessions[ses],'analysis',file_names[cond]),allow_pickle=True).tolist()
            all_decoding_accuracy[:, ses, cond] = np.array([np.mean(x['model_accuracy'],axis=0) for x in dec['decoding_models']])
            if cond == 2: #The trial history condition
                confusion_matrices.append(np.squeeze(dec['confusion_matrix']))

dec_acc_subj = np.zeros([133, np.unique(subj_code).shape[0], len(file_names)]) * np.nan
for k in subj_code:
    dec_acc_subj[:, k, :] = np.mean(all_decoding_accuracy[:, subj_code==k, :], axis=1)
#The 4-class accuracy in the third column might not actually be comparable to the binary decoders
#But one could "reconstruct" a binary decoding from the multi class case by averaging parts of the confusion matrix 

reconstructed_accuracy = np.zeros([133, len(sessions), 2]) * np.nan
for k in range(len(sessions)):
    for n in range(confusion_matrices[0].shape[0]):
        conf = confusion_matrices[k][n,:,:,n]
        reconstructed_accuracy[n,k,0] = np.mean(np.hstack((np.sum(conf[:2,:2],axis=1), np.sum(conf[2:,2:],axis=1))))
        #Here access the confusion matrix for every time point, get all the true correct percentages for the collapsed
        #classification problem (even if a correct left is called incorrect left it still is left) and average 
        reconstructed_accuracy[n,k,1] = np.mean(np.hstack((np.sum(conf[[0,0,3,3],[0,3,0,3]].reshape(2,2),axis=1), np.sum(conf[[1,1,2,2],[1,2,1,2]].reshape(2,2),axis=1))))
        #this is the outcome
        
reconstructed_subj = np.zeros([reconstructed_accuracy.shape[0], len(subj), 2]) * np.nan
for k in range(subj.shape[0]):
    reconstructed_subj[:,k,:] = np.mean(reconstructed_accuracy[:,subj_code==k,:],axis=1)



# fi = plt.figure()
# ax = fi.add_subplot(111)
# cols = ['#9f6023', '#345f69','#4a3c77']
# labels = ['Choice', 'Outcome', 'Choice x Outcome']
# plot_timecourse(ax, [np.squeeze(dec_acc_subj[:,:,0]), np.squeeze(dec_acc_subj[:,:,1]), np.squeeze(dec_acc_subj[:,:,2])], frame_rate ,idx_list, colors = cols, line_labels=labels)
# #ax.set_ylim([0.4,1])
# separate_axes(ax)


# sem = np.std(np.mean(dec_acc_subj,axis=0),axis=0) / np.sqrt(dec_acc_subj.shape[1])
# fi = plt.figure(figsize=[2,4.8])
# ax = fi.add_subplot(111)
# bars = ax.bar(labels,np.mean(np.mean(dec_acc_subj,axis=0),axis=0), edgecolor='k', yerr=sem, capsize=4)
# for k in range(len(bars)):
#     bars[k].set_color(cols[k])
#     bars[k].set_linewidth(1)
#     bars[k].set_edgecolor('k')
# ax.set_ylim([0,1])
# separate_axes(ax)
# ax.set_ylabel('Decoding accuracy')


# #Construct the data frame to export for the lme -> do this for the full range of samples
# d_dict = dict()
# d_dict['accuracy'] = np.reshape(all_decoding_accuracy.transpose(1,2,0), (np.prod(all_decoding_accuracy.shape))) # will be time from condition from session
# d_dict['which_model'] = np.repeat(['choice', 'outcome', 'history'], all_decoding_accuracy.shape[0]).tolist() * all_decoding_accuracy.shape[1]
# d_dict['subject'] = np.repeat(subj_string, all_decoding_accuracy.shape[0] * all_decoding_accuracy.shape[2]).tolist()
# d_dict['session'] = np.repeat(session_string, all_decoding_accuracy.shape[0] * all_decoding_accuracy.shape[2]).tolist()
# d_dict['time'] = np.arange(all_decoding_accuracy.shape[0]).tolist() * np.prod(all_decoding_accuracy.shape[1:])
# df = pd.DataFrame(d_dict)
# output_loc = '/Users/loesch/Documents/Churchland_lab/Trial-history_manuscript/Figures/Suppl_fig2_decoding'
# df.to_csv(os.path.join(output_loc, 'decoding_accuracy_different_decoders.csv'))


fi = plt.figure()
ax = fi.add_subplot(111)
cols = ['#9f6023','#345f69']
labels = ['Previous choice', 'Previous outcome']
plot_timecourse(ax, [np.squeeze(dec_acc_subj[:,:,0]), np.squeeze(dec_acc_subj[:,:,1])], frame_rate ,idx_list, colors = cols, line_labels=labels)
#ax.set_ylim([0.4,1])
separate_axes(ax)


cols = ['#9f6023','#4a3c77','k', '#345f69', '#4a3c77']
sem = np.hstack((np.std(np.mean(dec_acc_subj[:,:,:2],axis=0),axis=0) / np.sqrt(dec_acc_subj.shape[1]), np.nan, np.std(np.mean(reconstructed_subj,axis=0),axis=0) / np.sqrt(reconstructed_subj.shape[1])))
av = np.hstack((np.mean(np.mean(dec_acc_subj[:,:,:2],axis=1),axis=0), np.nan, np.mean(np.mean(reconstructed_subj,axis=1),axis=0)))
fi = plt.figure(figsize=[3,4.8])
ax = fi.add_subplot(111)
bars = ax.bar(np.arange(av.shape[0]),av, edgecolor='k', yerr=sem, capsize=4)
for k in range(len(bars)):
    bars[k].set_color(cols[k])
    bars[k].set_linewidth(1)
    bars[k].set_edgecolor('k')
ax.set_ylim([0,1])
separate_axes(ax)
ax.set_xticks([0.5, 3.5])
ax.set_xticklabels(['Previous choice', 'Previous outcome'])
ax.set_ylabel('Decoding accuracy')



# fi = plt.figure()
# ax = fi.add_subplot(111)
# cols = ['#9f6023', '#daae67', 'k', '#345f69','#75bdbd']
# labels = ['Choice', 'Choice from trial history', '', 'Outcome', 'Outcome from trial history']
# plot_timecourse(ax, [np.squeeze(dec_acc_subj[:,:,0]), np.squeeze(reconstructed_subj[:,:,0]), np.squeeze(dec_acc_subj[:,:,1]), np.squeeze(reconstructed_subj[:,:,1])], frame_rate ,idx_list, colors = cols, line_labels=labels)
# #ax.set_ylim([0.4,1])
# separate_axes(ax)

# sem = np.hstack((np.std(np.mean(dec_acc_subj[:,:,:2],axis=0),axis=0) / np.sqrt(dec_acc_subj.shape[1]), np.nan, np.std(np.mean(reconstructed_subj,axis=0),axis=0) / np.sqrt(reconstructed_subj.shape[1])))
# av = np.hstack((np.mean(np.mean(dec_acc_subj[:,:,:2],axis=1),axis=0), np.nan, np.mean(np.mean(reconstructed_subj,axis=1),axis=0)))
# fi = plt.figure(figsize=[2.5,4.8])
# ax = fi.add_subplot(111)
# bars = ax.bar(labels,av, edgecolor='k', yerr=sem, capsize=4)
# for k in range(len(bars)):
#     bars[k].set_color(cols[k])
#     bars[k].set_linewidth(1)
#     bars[k].set_edgecolor('k')
# ax.set_ylim([0,1])
# separate_axes(ax)
# ax.set_xticklabels(labels, rotation=45, ha='right')
# ax.set_ylabel('Decoding accuracy')



# #Construct the data frame to export for the lme -> do this for the full range of samples
# d_dict = dict()
# d_dict['accuracy'] = np.hstack((all_decoding_accuracy[:,:,:2].transpose(1,2,0).flatten(), reconstructed_accuracy.transpose(1,2,0).flatten())) # will be time from condition from session
# d_dict['decoded_variable'] = np.repeat(['choice', 'outcome'], all_decoding_accuracy.shape[0]).tolist() * all_decoding_accuracy.shape[1] * 2
# d_dict['is_history'] = ['no'] * np.size(reconstructed_accuracy) + ['yes'] * np.size(reconstructed_accuracy) 
# d_dict['subject'] = np.repeat(subj_string, all_decoding_accuracy.shape[0] * 2).tolist() * 2
# d_dict['session'] = np.repeat(session_string, all_decoding_accuracy.shape[0] * 2).tolist() * 2
# d_dict['time'] = np.arange(all_decoding_accuracy.shape[0]).tolist() * np.prod(reconstructed_accuracy.shape[1:]) * 2
# df = pd.DataFrame(d_dict)
# output_loc = '/Users/loesch/Documents/Churchland_lab/Trial-history_manuscript/Figures/Suppl_fig2_decoding'
# df.to_csv(os.path.join(output_loc, 'decoding_accuracy_different_decoders.csv'))



#Construct the data frame to export for the lme -> do this for time averages
d_dict = dict()
d_dict['accuracy'] = np.hstack((np.mean(all_decoding_accuracy[:,:,:2],axis=0).T.flatten(), np.mean(reconstructed_accuracy[:,:,:2],axis=0).T.flatten())) # will be time from condition from session
d_dict['decoded_variable'] = np.repeat(['choice', 'outcome'], len(sessions)).tolist() * 2
d_dict['is_history'] = ['no'] * len(sessions) * 2 + ['yes'] *  len(sessions) * 2
d_dict['subject'] = np.repeat(subj_string, 2).tolist() * 2
d_dict['session'] = np.repeat(session_string, 2).tolist() * 2
df = pd.DataFrame(d_dict)
output_loc = '/Users/loesch/Documents/Churchland_lab/Trial-history_manuscript/Figures/Suppl_fig2_decoding'
df.to_csv(os.path.join(output_loc, 'decoding_accuracy_different_decoders.csv'))


#-----------------------------------------
