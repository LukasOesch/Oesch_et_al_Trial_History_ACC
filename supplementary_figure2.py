#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 27 23:48:53 2025

@author: loesch
"""

from chiCa import *
import chiCa
import numpy as np
import pandas as pd
from labdatatools import *
import os
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Arial"
from glob import glob
from scipy.ndimage import gaussian_filter1d
import multiprocessing as mp
from time import time
import sys
sys.path.append('/Users/loesch/Documents/Churchland_lab/chiCa') #Add this path for multiprocessing
from sklearn.metrics import confusion_matrix

#%%-----Include these sessions into the analysis

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

#%%---Fit decoding models to either classify animal's choices or the outcomes 
# thereof from the neural population activity. These decoders serve as best-case
# decoding scenarios against which we will compare the effective decodability of 
# choice and outcome that we could obtain when we fit the combined trial history
# decoders. We do this because for certain neural geometries, optimizing decoders
# to classify the four trial history contexts may require trade-offs in the 
# decodability of the choice or the outcome.

use_name = ['Supp_Fig2_decoding_choice_spr.npy' ,'Supp_Fig2_decoding_outcome_spr.npy'] #Fit decoders for animal's choices and outcomes
signal_type = 'S' # Use the inferred spike rate signal for decoding
balancing_mode = 'equal' # Within each main class equally balance secondary class labels
cover_all_samples = True # Efficiently sample all available trials
k_folds = 8 # One session has a very limited number of trials for one of the conditions and so 8 folds is the maximum number for cross-validation
subsampling_rounds = 20 # Determine how many times to re-draw samples from the majority class
penalty = 'l2' # L2-norm, ridge penalty
normalize = True # Standardize the inputs
reg = 10**(np.linspace(-10,1,12)) # Define the search grid for the inverse regularization strength
model_params = {'penalty': penalty, 'solver': 'liblinear', 'inverse_regularization_strength': reg, 'fit_intercept': True, 'normalize': normalize} #Po

for session_dir in sessions:
    trial_alignment_file = glob(session_dir + '/analysis/*miniscope_data.npy')[0]
    miniscope_data = np.load(trial_alignment_file, allow_pickle = True).tolist()  
    frame_rate = miniscope_data['frame_rate']
    
    #Define the task phase alignment
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
    
#%%----Load the decoding models------

## Re-define the task phase alignment if you have already obtained the model fits
# trial_alignment_file = glob(sessions[0] + '/analysis/*miniscope_data.npy')[0] # Use the first session to retrieve the frame rate
# miniscope_data = np.load(trial_alignment_file, allow_pickle = True).tolist()  
# frame_rate = miniscope_data['frame_rate']

# #Define the task phase alignment
# aligned_to = ['outcome_presentation', 'outcome_end', 'PlayStimulus', 'DemonWaitForResponse']
# time_frame = [ np.array([round(0*frame_rate), round(1*frame_rate)+1], dtype=int),
#               np.array([round(-1*frame_rate), round(0.3*frame_rate)+1], dtype=int),
#               np.array([round(-0.5*frame_rate), round(1*frame_rate)+1], dtype=int),
#               np.array([round(-0.2*frame_rate), round(0.3*frame_rate)+1], dtype=int)]

# Get indices of the timepoints within the different sessions
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
    
#Load the data for previous choice, outcome and  previous choice x outcome models
file_names = ['Supp_Fig2_decoding_choice_spr.npy' ,'Supp_Fig2_decoding_outcome_spr.npy','Fig2_decoding_previous_choice_outcome_combination_spr.npy']
all_decoding_accuracy = np.zeros([133, len(sessions), len(file_names)]) * np.nan
confusion_matrices = []
for ses in range(len(sessions)):
        for cond in range(len(file_names)): #Loop through the different decoding models specified by the different file names
            dec = np.load(os.path.join(sessions[ses],'analysis',file_names[cond]),allow_pickle=True).tolist()
            all_decoding_accuracy[:, ses, cond] = np.array([np.mean(x['model_accuracy'],axis=0) for x in dec['decoding_models']])
            if cond == 2: #The trial history condition
                confusion_matrices.append(np.squeeze(dec['confusion_matrix']))

dec_acc_subj = np.zeros([133, np.unique(subj_code).shape[0], len(file_names)]) * np.nan
for k in subj_code:
    dec_acc_subj[:, k, :] = np.mean(all_decoding_accuracy[:, subj_code==k, :], axis=1)

# The 4-class accuracy in the third column of dec_acc_subj is not directly comparable
# to the binary decoders because it does a different classification and because
# its chance level is lower than the one of the binary decoders (0.25 vs. 0.5).
# But one could "reconstruct" a binary decoding problem from the multi class case
# by averaging parts of the confusion matrix. Specifically, we could consider all
# instances where the trial history decoder assigned either a correct or incorrect
# previous choice label to a trial, even if the decoder incorrectly assigns the outcome.
# This way we can "collapse" the confusion matrix and obtain a binary-like classification
# accuracy estimate from our 4-class decoding.

reconstructed_accuracy = np.zeros([133, len(sessions), 2]) * np.nan
for k in range(len(sessions)):
    for n in range(confusion_matrices[0].shape[0]):
        conf = confusion_matrices[k][n,:,:,n]
        reconstructed_accuracy[n,k,0] = np.mean(np.hstack((np.sum(conf[:2,:2],axis=1), np.sum(conf[2:,2:],axis=1))))
        #Here access the confusion matrix for every time point, get all the true correct percentages for the collapsed
        #classification problem (even if a correct left is called incorrect left it still is left) and average 
        reconstructed_accuracy[n,k,1] = np.mean(np.hstack((np.sum(conf[[0,0,3,3],[0,3,0,3]].reshape(2,2),axis=1), np.sum(conf[[1,1,2,2],[1,2,1,2]].reshape(2,2),axis=1))))
        #this is the previous outcome
        
reconstructed_subj = np.zeros([reconstructed_accuracy.shape[0], len(subj), 2]) * np.nan
for k in range(subj.shape[0]):
    reconstructed_subj[:,k,:] = np.mean(reconstructed_accuracy[:,subj_code==k,:],axis=1)
    # reconstructed_subj contains the reconstructed previous choice decoding
    # acuracy in column 0 and the previous outcome in column 1.

#%%---Plot the binary decoders over time and compare to reconstructed decoding
# accuracy from trial history decoders.

#########----Supplementary Figure 2a------############

fi = plt.figure()
ax = fi.add_subplot(111)
cols = ['#9f6023','#345f69']
labels = ['Previous choice', 'Previous outcome']
plot_timecourse(ax, [np.squeeze(dec_acc_subj[:,:,0]), np.squeeze(dec_acc_subj[:,:,1])], frame_rate ,idx_list, colors = cols, line_labels=labels)
ax.set_ylim([0.4, 1])
separate_axes(ax)
ax.set_ylabel('Decoding accuracy')
ax.set_xlabel('Time (s)')

#######-----Supplemenatry Figure 2b--------############

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

#%%---Take the fitted trial history decoders and split trials by how much time
# elapsed between the outcome presentation (the first time point in our alignment)
# and the moment the mouse initated the next trial (within the Stimulus phase).
# Compare the decoding accuracy for different ITI durations.

use_name = 'Fig2_decoding_previous_choice_outcome_combination_spr.npy'
class_labels = []
trial_accuracy = []
shuffle_trial_accuracy = []
ITI_duration = []

for session_dir in sessions:
    decoding_data = np.load(os.path.join(session_dir, 'analysis', use_name), allow_pickle=True).tolist()
    trialdata = pd.read_hdf(glob(session_dir + '/chipmunk/*.h5')[0], '/Data')
    
    #---First, extract the trial history labels for the current session
    s_dict = dict()
    s_dict['choice'] = np.array(trialdata['response_side'])
    s_dict['prior_choice'] = determine_prior_variable(np.array(trialdata['response_side']), np.ones(len(trialdata)), 1, 'consecutive')
   
    outcome = np.array(trialdata['response_side'] == trialdata['correct_side'], dtype=float) #Define as float so that nans (that are float) can be retained
    outcome[np.array(np.isnan(trialdata['response_side']))] = np.nan
    s_dict['outcome'] = outcome
    s_dict['prior_outcome'] = determine_prior_variable(outcome, np.ones(len(trialdata)), 1, 'consecutive')

    s_dict['valid_trials'] = (np.isnan(s_dict['choice']) == 0) & (np.isnan(s_dict['prior_choice']) == 0)
    #==========One session from LO067 has two unidentified response port out events
    if (os.path.split(session_dir)[1]=='20240209_105931') & (os.path.split(os.path.split(session_dir)[0])[1] == 'LO067'):
        valid_trials = np.where(s_dict['valid_trials'])[0] 
        valid_trials = np.delete(valid_trials, [113,115])
        s_dict['valid_trials'] = np.zeros([outcome.shape[0]], dtype=bool)
        s_dict['valid_trials'][valid_trials] = True
    #====================================================
    
    #To actually compute the trial-wise accuracy one also needs to furnish the labels
    labels = np.zeros([trialdata.shape[0]])
    labels[(s_dict['prior_choice']==0) & (s_dict['prior_outcome']==1)] = 0
    labels[(s_dict['prior_choice']==0) & (s_dict['prior_outcome']==0)] = 1
    labels[(s_dict['prior_choice']==1) & (s_dict['prior_outcome']==0)] = 2
    labels[(s_dict['prior_choice']==1) & (s_dict['prior_outcome']==1)] = 3
    labels = labels[s_dict['valid_trials']]
    
    #---Get the fitted decoding model
    mod = decoding_data['decoding_models']
    num_folds = np.unique(mod[0]['fold_number']).shape[0]
    if len(mod[0]['model_coefficients'][0].shape) > 1:
        num_classes = mod[0]['model_coefficients'][0].shape[0]
    else:
        num_classes = 2    
    
    trial_acc = np.zeros([len(mod), np.sum(s_dict['valid_trials'])]) * np.nan
    shuffle_trial_acc = np.zeros([len(mod), np.sum(s_dict['valid_trials'])]) * np.nan
    for time_p in range(len(mod)):
      # Because we only stored the trial-by-trial logodds for each class label,
      # we have to re-compute the classification using the class logodds and the
      # true class labels. So, belwo here we first retrieve the trial-wise
      # loggods.
      if num_classes == 2:
          tmp_dec = np.hstack(mod[time_p]['model_prediction_logodds']).T #list containing class logodds for every test sample, number of list elements is k_folds * subsampling roudns
          tmp_shuf = np.hstack(mod[time_p]['shuffle_prediction_logodds']).T
      else:
          tmp_dec = np.vstack(mod[time_p]['model_prediction_logodds'])
          tmp_shu = np.vstack(mod[time_p]['shuffle_prediction_logodds'])
      
      # Here, we locate the trials IDs from the test sample to verify that indeed
      # every trial was used at least once in the test set.
      tmp_idx = []
      for k in range(len(mod[time_p]['model_prediction_logodds'])):
        tmp_idx.append(mod[time_p]['pick_to_balance'][k][mod[time_p]['test_index'][k]])
      tmp_idx = np.hstack(tmp_idx)
      trial_idx = np.unique(tmp_idx) #This is with respect to the valid_trials!
      
      if trial_idx.shape[0] != np.sum(s_dict['valid_trials']):
          warning_string = f'Not all the trials were sampled in session: {session_dir} at timepoint: {time_p}'
          break
      else: # Now use the loggods to classify
          for k in trial_idx:
             if num_classes == 2: # In the binary case the sign of the loggods defines the label
                 trial_acc[time_p,k] = np.mean((tmp_dec[tmp_idx == k] >= 0) == labels[k])
                 shuffle_trial_acc[time_p,k] = np.mean((tmp_shu[tmp_idx == k] >= 0) == labels[k])
             else: # In the multinomial case, the label associated with the highest logodds is chosen
                 trial_acc[time_p,k] = np.mean(np.argmax(tmp_dec[tmp_idx == k],axis=1) == labels[k])
                 shuffle_trial_acc[time_p,k] = np.mean(np.argmax(tmp_shu[tmp_idx == k],axis=1) == labels[k])
      
    #Retrieve ITI duration
    tmp = np.squeeze(trialdata['trial_start_time'].tolist()) + np.squeeze(trialdata['DemonWaitForCenterFixation'].tolist())[:,1]
    ITI_duration.append(np.hstack((np.nan, np.diff(tmp)))[s_dict['valid_trials']])
    
    trial_accuracy.append(trial_acc)
    shuffle_trial_accuracy.append(shuffle_trial_acc)
    class_labels.append(labels)

# Bin the ITI 
upper_edges = [7.5, 12.5, 17.5, 22.5, 27.5, 10**3]
ITI_duration_bin = [np.digitize(x, upper_edges, right=True) for x in ITI_duration]
labels = ['5', '10', '15','20', '25', '>27.5']

dec_hist_ITI = [] #Dimensions are: sessions x time points x ITI bin x trial history context (last element is overall)
shu_hist_ITI = []
for k in range(len(trial_accuracy)):
    tmp = np.zeros([trial_accuracy[0].shape[0], len(upper_edges), np.unique(class_labels[0]).shape[0]+1]) * np.nan #The data structure is timepoints x ITI duration bin x trial history context 
    tmp_s = np.zeros([trial_accuracy[0].shape[0], len(upper_edges), np.unique(class_labels[0]).shape[0]+1]) * np.nan #The data structure is timepoints x ITI duration bin x trial history context for bi in range(len(upper_edges)):
    for bi in range(len(upper_edges)):
        for hist in range(np.unique(class_labels[0]).shape[0] +1):
            if hist <  4:
                tmp[:,bi,hist] = np.mean(trial_accuracy[k][:, (class_labels[k]==hist) & (ITI_duration_bin[k]==bi)], axis=1)
                tmp_s[:,bi,hist] = np.mean(shuffle_trial_accuracy[k][:, (class_labels[k]==hist) & (ITI_duration_bin[k]==bi)], axis=1)
            else:
                tmp[:,bi,hist] = np.mean(trial_accuracy[k][:, (ITI_duration_bin[k]==bi)], axis=1)
                tmp_s[:,bi,hist] = np.mean(shuffle_trial_accuracy[k][:, (ITI_duration_bin[k]==bi)], axis=1)
    dec_hist_ITI.append(tmp)
    shu_hist_ITI.append(tmp_s)
dec_hist_ITI = np.squeeze(dec_hist_ITI)
shu_hist_ITI = np.squeeze(shu_hist_ITI)

#%%---Now start the plotting of decioding accuracy by ITI duration

###########------Supplementary Figure 2c and d------------##################

cols = ['#606cbe', '#48a7b1', '#deb35e', '#de8468'] # the trial history colors
grey = '#858585'   
phase_label = ['Stimulus']
#phase_label = ['Stimulus', 'Action'] # Alternatively also plot this for the action phase
# Make sure to keep them in the right order because the phase will be represented
# as an integer in the loop!

for phase in range(len(phase_label)): #Look at this during the stim- and optionally also the action period after the ITI
    decoding_accuracy_ITI_duration = []
    shuffle_accuracy_ITI_duration = []
    tmp_data = np.nanmean(dec_hist_ITI[:,idx_list[phase+2],:,:],axis=1)
    tmp_shu = np.nanmean(shu_hist_ITI[:,idx_list[phase+2],:,:],axis=1)
    for k in np.unique(subj_code):
        decoding_accuracy_ITI_duration.append(np.nanmean(tmp_data[subj_code == k,:,:],axis=0))
        shuffle_accuracy_ITI_duration.append(np.nanmean(tmp_shu[subj_code == k,:,:],axis=0))
    decoding_accuracy_ITI_duration = np.squeeze(decoding_accuracy_ITI_duration)
    shuffle_accuracy_ITI_duration = np.squeeze(shuffle_accuracy_ITI_duration)
    
    # Panel c
    ax = plt.figure(figsize=[4.8,4.8]).add_subplot(111)
    data = [shuffle_accuracy_ITI_duration, decoding_accuracy_ITI_duration]
    color_sub = [grey, 'k']
    line_l = ['Shuffle', 'Data']
    for n in range(len(data)):
        av = np.nanmean(data[n][:,:,4],axis=0)
        sem = np.nanstd(data[n][:,:,4],axis=0) / np.sqrt(data[n].shape[0])
        ax.plot(av, color = color_sub[n], label = line_l[n])
        ax.errorbar(np.arange(av.shape[0]), av, yerr=sem, fmt='o', capsize=4, color = color_sub[n], markerfacecolor = 'w')
    ax.set_ylim([0,1])
    ax.set_xticks(np.arange(av.shape[0]))
    ax.set_xticklabels(labels)
    ax.set_title(phase_label[phase])
    ax.set_xlabel('ITI duration (s)')
    ax.set_ylabel('Decoding accuracy')
    ax.legend()
    separate_axes(ax)
    
    # Panel d
    ax = plt.figure(figsize=[4.8,4.8]).add_subplot(111)
    for k in range(decoding_accuracy_ITI_duration.shape[2]-1):
        av = np.nanmean(decoding_accuracy_ITI_duration[:,:,k],axis=0)
        sem = np.nanstd(decoding_accuracy_ITI_duration[:,:,k],axis=0) / np.sqrt(decoding_accuracy_ITI_duration.shape[0])
        ax.plot(av, color = cols[k])
        ax.errorbar(np.arange(av.shape[0]), av, yerr=sem, fmt='o', capsize=4, color = cols[k], markerfacecolor = 'w')
    ax.set_ylim([0,1])
    ax.set_xticks(np.arange(av.shape[0]))
    ax.set_xticklabels(labels)
    ax.set_title(phase_label[phase])
    ax.set_xlabel('ITI duration (s)')
    ax.set_ylabel('Decoding accuracy')
    separate_axes(ax)
        
#-----------------Generate tables to test for for significant decoding as a function of ITI duration
output_loc = '/Users/loesch/Documents/Churchland_lab/Trial-history_manuscript/Figures/Suppl_fig2_decoding'

trial_phase = 2 # The stimulus phase here
tmp_dec = np.hstack([np.nanmean(dec_hist_ITI[x, :,:,:][idx_list[trial_phase],:,:][:,:,4],axis=0) for x in range(len(sessions))])
tmp_shu = np.hstack([np.nanmean(shu_hist_ITI[x, :,:,:][idx_list[trial_phase],:,:][:,:,4],axis=0) for x in range(len(sessions))])
d_dict = dict()
d_dict['decoding_accuracy'] = np.hstack((tmp_dec,tmp_shu))
d_dict['condition'] = ['Model'] * tmp_dec.shape[0] + ['Shuffle'] * tmp_shu.shape[0] 
d_dict['ITI_bin'] = labels * int((d_dict['decoding_accuracy'].shape[0] / len(labels)))
d_dict['subject'] = np.repeat(subj_string, len(labels)).tolist() * 2 
d_dict['session'] = np.repeat(session_string, len(labels)).tolist() * 2
df = pd.DataFrame(d_dict)
df.to_csv(os.path.join(output_loc, 'stim_phase_decoding_accuracy_by_ITI_duration.csv'))

#---Now the decoding accuracy by ITI split by trial history context
tmp_dec = np.squeeze([np.nanmean(dec_hist_ITI[x, :,:,:][idx_list[trial_phase],:,:][:,:,:4],axis=0) for x in range(len(sessions))])
d_dict = dict()
d_dict['decoding_accuracy'] = np.reshape(tmp_dec, tmp_dec.size)
d_dict['hist_context'] = ['p_correct_left', 'p_incorrect_left', 'p_incorrect_right', 'p_correct_right'] * int(tmp_dec.shape[0] * tmp_dec.shape[1])
d_dict['ITI_bin'] = np.repeat(labels, tmp_dec.shape[2]).tolist() * tmp_dec.shape[0]
d_dict['subject'] = np.repeat(subj_string, tmp_dec.shape[1] * tmp_dec.shape[2]) 
d_dict['session'] = np.repeat(session_string, tmp_dec.shape[1] * tmp_dec.shape[2])
df = pd.DataFrame(d_dict)
df.to_csv(os.path.join(output_loc, 'Stim_phase_decoding_accuracy_by_ITI_duration_hist_context.csv'))


#%%---Decoding trial history from two trials back

dec_sessions = sessions[:9] + sessions[10:20] + sessions[21:] #Exclude one session for LY008 that has no more than five trials of one of the classes

use_name = 'decoding_choice_outcome_combination_twoBack_spr.npy'
signal_type = 'S'
balancing_mode = 'proportional'
cover_all_samples = True
k_folds = 8 #Was 8 before, let's see Folds for cross-validation -> out of necessity!
subsampling_rounds = 20 #Re-drawing of samples from majority class
penalty = 'l2' 
normalize = True
reg = 10**(np.linspace(-10,1,12))
model_params = {'penalty': penalty, 'solver': 'liblinear', 'inverse_regularization_strength': reg, 'fit_intercept': True, 'normalize': normalize} #Po

for session_dir in dec_sessions:
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
    choice_two_back = np.array(c_data['choice_two_back'])
    outcome_two_back = np.array(c_data['outcome_two_back'])
    
    labels = np.zeros([trialdata.shape[0]])
    labels[(choice_two_back==0) & (outcome_two_back==1)] = 0
    labels[(choice_two_back==0) & (outcome_two_back==0)] = 1
    labels[(choice_two_back==1) & (outcome_two_back==0)] = 2
    labels[(choice_two_back==1) & (outcome_two_back==1)] = 3
    secondary_labels = None
    #secondary_labels = np.array(c_data['choice'])
    valid_trials = np.where(c_data['valid_two_back'])[0] #Retrieve valid trials with a response before and after 
    # #==========One session from LO067 has two unidentified response port out events
    # if (os.path.split(session_dir)[1]=='20240209_105931') & (os.path.split(os.path.split(session_dir)[0])[1] == 'LO067'):
    #     valid_trials = np.delete(valid_trials, [113,115])
    # #====================================================
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
    prediction_accuracy, shuffled_prediction_accuracy, confusion_matrix = cross_decoding_analysis(decoding_models, Y, labels, valid_trials)
    
    out_dict = {'decoding_models': decoding_models,
                'prediction_accuracy': prediction_accuracy,
                'shuffled_prediction_accuracy': shuffled_prediction_accuracy,
                'confusion_matrix': confusion_matrix}
    
    np.save(os.path.join(session_dir, 'analysis', use_name), out_dict)   

#%%---Load the models and plot the data

use_name = 'decoding_choice_outcome_combination_twoBack_spr.npy'

#Retrieve important data
classes = 4
model_accuracy = []
shuffle_accuracy = []
label_wise_accuracy_list = []
cross_prediction = []
shuffled_cross_prediction = []
class_wise_cross_prediction = []
#confusion_timepoints = [14,85]
confusion_matrices = []
coefficients = []
ses_included = []
for session_dir in sessions:
    if os.path.isfile(os.path.join(session_dir, 'analysis', use_name)):
        decoding_data = np.load(os.path.join(session_dir, 'analysis', use_name), allow_pickle=True).tolist()
        label_wise_accuracy = np.zeros([len(decoding_data['confusion_matrix']),classes]) *np.nan
        for k in range(len(decoding_data['confusion_matrix'])):
            for n in range(classes): #Get the main diagonal of the confusion matrix at the training index position
                label_wise_accuracy[k,n] = decoding_data['confusion_matrix'][k][n,n,k]
        label_wise_accuracy_list.append(label_wise_accuracy)
        
        #Overall accuray and shuffle plus model coefficients
        model_accuracy.append(np.squeeze([np.mean(x['model_accuracy'],axis=0) for x in decoding_data['decoding_models']]))
        shuffle_accuracy.append(np.squeeze([np.mean(x['shuffle_accuracy'],axis=0) for x in decoding_data['decoding_models']]))
        coefficients.append(np.squeeze([np.mean(x['model_coefficients'],axis=0) for x in decoding_data['decoding_models']]).transpose(0,2,1))
        
        #Get cross-prediction accuracy over all class labels
        cross_prediction.append(decoding_data['prediction_accuracy'])
        shuffled_cross_prediction.append(decoding_data['shuffled_prediction_accuracy'])
        confusion_matrices.append(np.squeeze(decoding_data['confusion_matrix']))
        
        #Now get the class wise decoding accuracy
        tmp = np.zeros([len(decoding_data['confusion_matrix']),len(decoding_data['confusion_matrix']),classes]) * np.nan
        for time_p in range(len(decoding_data['confusion_matrix'])):
            for idx in range(len(decoding_data['confusion_matrix'])):
                tmp[idx,time_p,:] = np.diagonal(decoding_data['confusion_matrix'][time_p][:,:,idx])
        class_wise_cross_prediction.append(tmp)
        
        ses_included.append(session_dir)
    else:
        pass
    
#%%------Now start all the plotting 

#Get subject identifiers
subj = np.unique([os.path.split(os.path.split(session_dir)[0])[1] for session_dir in ses_included])
subj_code = np.zeros([len(ses_included)],dtype=int)
subj_string = []
session_string = []
for n in range(len(ses_included)):
    for k in range(subj.shape[0]):
        if subj[k] in ses_included[n]:
            subj_code[n] = k
            subj_string.append(str(subj[k]))
    session_string.append(os.path.split(ses_included[n])[1])
    
dec_acc_subj = np.zeros([133, np.unique(subj_code).shape[0]]) * np.nan
shu_acc_subj = np.zeros([133, np.unique(subj_code).shape[0]]) * np.nan
for k in subj_code:
    dec_acc_subj[:, k] = np.mean(np.vstack(model_accuracy)[subj_code==k,:], axis=0)
    shu_acc_subj[:, k] = np.mean(np.vstack(shuffle_accuracy)[subj_code==k,:], axis=0)
    
######---------Supplementary Figure 2e----------##################
    
# #Optional: Plot the overall decoding accuracy over trial time  
# fi = plt.figure()
# ax = fi.add_subplot(111)
# cols = ['k', grey]
# labels = ['Model', 'Shuffle']
# plot_timecourse(ax, [dec_acc_subj, shu_acc_subj], frame_rate ,idx_list, colors = cols, line_labels=labels)
# ax.set_ylim([0,1])
# separate_axes(ax)

cols = ['w', grey]
labels = ['Model', 'Shuffle']
sem = [np.std(np.mean(dec_acc_subj,axis=0)) / np.sqrt(dec_acc_subj.shape[1]), np.std(np.mean(shu_acc_subj,axis=0)) /  np.sqrt(dec_acc_subj.shape[1])]
jitter = (0.5 - np.random.rand(dec_acc_subj.shape[1]))/(1/0.8)
fi = plt.figure(figsize=[1,4.8])
ax = fi.add_subplot(111)
bars = ax.bar(labels,[np.mean(np.mean(dec_acc_subj,axis=0)), np.mean(np.mean(shu_acc_subj,axis=0))], edgecolor='k', yerr=sem, capsize=4)
for k in range(len(bars)):
    bars[k].set_color(cols[k])
    bars[k].set_linewidth(1)
    bars[k].set_edgecolor('k')

for k in range(dec_acc_subj.shape[1]):
    ax.scatter([0,1] + jitter[k], [np.mean(dec_acc_subj,axis=0)[k], np.mean(shu_acc_subj,axis=0)[k]], c=grey ,s=14)
ax.set_ylim([0,1])
separate_axes(ax)
ax.set_ylabel('Decoding accuracy')
ax.set_title('Trial history decoding t-2')

#######----------Supplementary Figure 2f---------###################
tmp_acc = np.squeeze(label_wise_accuracy_list)
label_wise_acc_subj = np.zeros([tmp_acc.shape[1],np.unique(subj_code).shape[0], 4]) * np.nan
for k in subj_code:
    label_wise_acc_subj[:, k,:] = np.mean(tmp_acc[subj_code==k,:], axis=0)

fi = plt.figure(figsize= [8,4.8])
ax = fi.add_subplot(111)
cols = ['#606cbe', '#48a7b1', '#deb35e', '#de8468',grey]
spacer = 6
labels = ['Previous correct left', 'Previous incorrect left', 'Previous incorrect right', 'Previous correct right', 'Shuffle']
plot_timecourse(ax, [k.T for k in label_wise_acc_subj.transpose(2,1,0)] + [shu_acc_subj] , frame_rate ,idx_list, colors = cols, line_labels=labels, spacer=spacer )
ax.set_ylim([0.2,1]) #limit the range to not have too much white space
separate_axes(ax)
ax.set_ylabel('Time (s)')
ax.set_ylabel('Decoding accuracy')
ax.set_title('Trial history decoding for t-2')

#----------
output_loc = '/Users/loesch/Documents/Churchland_lab/Trial-history_manuscript/Figures/Suppl_fig2_decoding'
df.to_csv(os.path.join(output_loc, 'decoding_accuracy_different_decoders.csv'))
#Construct the data frame to export for the lme --> this is the overall accuracy vs shuffle
d_dict = dict()
d_dict['decoding_accuracy'] = np.hstack(model_accuracy).tolist() +  np.hstack(shuffle_accuracy).tolist()
d_dict['condition'] = ['Model'] * np.size(np.hstack(model_accuracy)) + ['Shuffle'] * np.size(np.hstack(model_accuracy))
d_dict['subject'] = np.repeat(subj_string, model_accuracy[0].shape[0]).tolist() * 2
d_dict['time'] = np.array(np.arange(model_accuracy[0].shape[0])).tolist() * (len(model_accuracy * 2))
d_dict['session'] = np.repeat(session_string, model_accuracy[0].shape[0]).tolist() * 2
df = pd.DataFrame(d_dict)
df.to_csv(os.path.join(output_loc, 'History_decoding_t-2_overall.csv'))

#-----
#Construct the data frame for the history context wise decoding
d_dict = dict()
d_dict['decoding_accuracy'] = np.reshape(np.transpose(np.squeeze(label_wise_accuracy_list),(0,2,1)), np.squeeze(label_wise_accuracy_list).size) #Now first the time for one condition, then the condition, then session
d_dict['hist_context'] = np.repeat(labels[:-1], label_wise_accuracy_list[0].shape[0]).tolist() * len(label_wise_accuracy_list)
d_dict['time'] = np.arange(label_wise_accuracy_list[0].shape[0]).tolist() * (label_wise_accuracy_list[0].shape[1] * len(subj_string))
d_dict['subject'] = np.repeat(subj_string, label_wise_accuracy_list[0].size)
d_dict['session'] = np.repeat(session_string, label_wise_accuracy_list[0].size)
df = pd.DataFrame(d_dict)
df.to_csv(os.path.join(output_loc, 'History_decoding_t-2_by_context.csv'))


