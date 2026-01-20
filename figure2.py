# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 13:29:13 2025

@author: Lukas Oesch
"""

from chiCa import *
import chiCa
import numpy as np
import pandas as pd
from labdatatools import *
import os
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Arial"
import matplotlib
from glob import glob
from scipy.ndimage import gaussian_filter1d
from scipy.stats import zscore
import multiprocessing as mp
from time import time
import sys
sys.path.append('/Users/loesch/Documents/Churchland_lab/chiCa') #Add the path to chiCa,
#so that the decoding functions will be visible to all the processes when running multiprocess
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

#%%------Figure 2B, bottom: Plot the neuron contours for LO028----------#####

session_dir = sessions[0]
trial_alignment_file = glob(session_dir + '/analysis/*miniscope_data.npy')[0]
miniscope_data = np.load(trial_alignment_file, allow_pickle = True).tolist()

#Retrieve the spatial footprints and threshold 
A = miniscope_data['A']
A[A < 0.1] = 0

#Make colormap jumping from white to dark blue and easing into pale green-yellow
hex_cols = np.flip(['#ffffd6', '#ecf8af', '#ddf1b1', '#c7e9b4', '#7fccba', '#44b6c5', '#1b91c0', '#1e77b3', '#205fa7', '#253393', '#070d55']).tolist()
col_map = matplotlib.colors.LinearSegmentedColormap.from_list("", hex_cols)

fi = plt.figure()
ax = fi.add_subplot(1,1,1, aspect='equal')
ax.set_xticks([])
ax.set_yticks([])

for k in range(A.shape[2]):
   ax.contour(np.flipud(A[:,:,k]))
   #Image has to be flipped vertically because np.contour and plt.imshow use different conventions

#%%-------Figure 2C and D: Plot example neurons from LO069 

#Define session and colors, load imaging data 
session_dir = sessions[15] #This is the respective session
classes = 4 #Number of different trial history contexts - quite redundant because it could be retrieved from the unique label values or the string labels
signal_type = 'S'
cols = ['#606cbe', '#48a7b1', '#deb35e', '#de8468']
trial_alignment_file = glob(session_dir + '/analysis/*miniscope_data.npy')[0]
miniscope_data = np.load(trial_alignment_file, allow_pickle = True).tolist()  
frame_rate = miniscope_data['frame_rate']

#Set up the different alignment phases by defining a task state and the amount of frames before and
#after the start of that state should be included
aligned_to = ['outcome_presentation', 'outcome_end', 'PlayStimulus', 'DemonWaitForResponse']
time_frame = [ np.array([round(0*frame_rate), round(1*frame_rate)+1], dtype=int),
              np.array([round(-1*frame_rate), round(0.3*frame_rate)+1], dtype=int),
              np.array([round(-0.5*frame_rate), round(1*frame_rate)+1], dtype=int),
              np.array([round(-0.2*frame_rate), round(0.3*frame_rate)+1], dtype=int)]

trialdata = pd.read_hdf(glob(session_dir + '/chipmunk/*.h5')[0], '/Data')
c_data = get_chipmunk_behavior(session_dir)
prior_choice = np.array(c_data['prior_choice'])
prior_outcome = np.array(c_data['prior_outcome'])

#Get the labels for the different trial history contexts
labels = np.zeros([trialdata.shape[0]])
labels[(prior_choice==0) & (prior_outcome==1)] = 0
labels[(prior_choice==0) & (prior_outcome==0)] = 1
labels[(prior_choice==1) & (prior_outcome==0)] = 2
labels[(prior_choice==1) & (prior_outcome==1)] = 3

choice = c_data['choice']
outcome = c_data['outcome']

valid_trials = np.where(c_data['valid_past'])[0] #Retrieve valid trials with a response before and after 
valid_trials_before = valid_trials-1
val_trials =  [valid_trials_before]*2 + [valid_trials]*2 #Combine trial identities for the ITI before the new trial and once it has started

if signal_type == 'S': #If using inferred spike rate, convolve witha gaussian kernel.
      signal = gaussian_filter1d(miniscope_data[signal_type].T, 1, axis=0, mode='constant', cval=0, truncate=4)
      #Pad the first and last samples with zeros in this condition
else:
      signal = miniscope_data[signal_type].T #Transpose the original signal so that rows are timepoints and columns cells
  
#Determine whether to use all or only some of the recroded neurons and do the alignment on these
keep_neuron = np.arange(signal.shape[1])
Y = []
for k in range(len(aligned_to)):
        state_start_frame, state_time_covered = find_state_start_frame_imaging(aligned_to[k], trialdata, miniscope_data['frame_interval'], miniscope_data['trial_starts'], miniscope_data['trial_start_time_covered'])                                                                     
        zero_frame = np.array(state_start_frame[val_trials[k]] + time_frame[k][0], dtype=int) #The firts frame to consider
        
        for add_to in np.arange(time_frame[k][1] - time_frame[k][0]):
            Y.append(signal[zero_frame + add_to,:][:, keep_neuron])

Y = np.squeeze(Y)
Y_2d = zscore(np.reshape(Y.transpose(2,1,0), (Y.shape[2], Y.shape[1]*Y.shape[0])).T, axis=0)
Y_norm = np.reshape(Y_2d.T, (Y.shape[2], Y.shape[1], Y.shape[0])).transpose(2,1,0)

#this is the full set of neurons
#neuron_id = [154,155,151,16,185,183,181]

#For visualization get a list of indices for every task phase individually
insertion_index = np.zeros(len(time_frame)+1)
insertion_index[1] = time_frame[0][1] - time_frame[0][0]
idx_list = [np.arange(insertion_index[1],dtype=int)]
for k in range(1,len(time_frame)):
    insertion_index[k+1] = int(insertion_index[k] + time_frame[k][1] - time_frame[k][0])
    idx_list.append(np.arange(insertion_index[k],insertion_index[k+1], dtype=int))

###----------Figure 2D----------------###

neuron_id = [154,16,181]
ax = []
for n in neuron_id:
    ax.append(plt.figure(figsize=[8,4.8]).add_subplot())
    #Plot the PETH for the specified neurons for every trial history context
    plot_timecourse(ax[-1], [Y_norm[:,labels[valid_trials]==k,n] for k in range(classes)], frame_rate ,idx_list, colors = cols)
    ax[-1].set_xlabel('Time (s)')
    ax[-1].set_ylabel('Activity (SD)')
    ax[-1].set_title(f'neuron {n}')

#Make the y axis the same for the plots of the different neurons
lims = [k.get_ylim() for k in ax]
common_lim = [np.min(np.squeeze(lims)[:,0]), np.max(np.squeeze(lims)[:,1])]
for k in range(len(ax)):
    ax[k].set_ylim(common_lim)
    separate_axes(ax[k])

###-----------Figure 2C-------------###
#Plot the continuous traces with the chocie-outcome events
import matplotlib.patches as patches

neuron_id = [154,155,151,16,185,183,181] #The set of example neurons
sig_spacing = [0,-6, -13,-23, -29,-35,-40] #Set the spacing between their traces
ax = plt.figure(figsize=[6,4.8]).add_subplot()
#Start plotting the traces of the example neurons
for k in range(len(neuron_id)):
    ax.plot(zscore(miniscope_data['F'][neuron_id[k],:].T) + sig_spacing[k],color='k', linewidth=1)
ylims = ax.get_ylim()

#Draw colored boxes representing the just made choice and received outcome.
#Boxes start at outcome presentation and go to 1s after outcome presentation
box_cols = cols[:4]
#Find the timing of outcome presentation
state_start_frame, state_time_covered = find_state_start_frame_imaging('outcome_presentation', trialdata, miniscope_data['frame_interval'], miniscope_data['trial_starts'], miniscope_data['trial_start_time_covered'])         
#Draw the boxes
for k in range(state_start_frame.shape[0]):
    if np.isnan(state_start_frame[k])==0:
        if (choice[k]==0) & (outcome[k]==1):
            ax.add_patch(patches.Rectangle(
                            xy=(state_start_frame[k], ylims[0]),  # point of origin.
                                width=frame_rate, height=ylims[1] - ylims[0], linewidth=None,
                                color=box_cols[0], alpha=0.6))
            
        if (choice[k]==0) & (outcome[k]==0):
            ax.add_patch(patches.Rectangle(
                            xy=(state_start_frame[k], ylims[0]),  # point of origin.
                                width=frame_rate, height=ylims[1] - ylims[0], linewidth=None,
                                color=box_cols[1], alpha=0.6))
            
        if (choice[k]==1) & (outcome[k]==0):
            ax.add_patch(patches.Rectangle(
                            xy=(state_start_frame[k], ylims[0]),  # point of origin.
                                width=frame_rate, height=ylims[1] - ylims[0], linewidth=None,
                                color=box_cols[2], alpha=0.6))
            
        if (choice[k]==1) & (outcome[k]==1):
            ax.add_patch(patches.Rectangle(
                            xy=(state_start_frame[k], ylims[0]),  # point of origin.
                                width=frame_rate, height=ylims[1] - ylims[0], linewidth=None,
                                color=box_cols[3], alpha=0.6))
ax.set_xlim([84220,86650])
ax.set_xlabel('Frames')
ax.set_ylabel('Activity (SD)')

#%%----Figure 2E: Plot activity delta for left vs right choice and correct vs incorrect
# trials for one of the recorded sessions from each mouse

#------Figure 2e-------
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

signal_type = 'S'

trial_history_acti = [[], [], [],[]]
centered_trial_history_acti = [[], [], [],[]]
prev_delta_acti = [[],[]]
neuron_count = []
combos = np.array([[0,1],[0,0],[1,0],[1,1]])
for s_idx in ses_idx:
    session_dir = sessions[s_idx]    
    
    trial_alignment_file = glob(session_dir + '/analysis/*miniscope_data.npy')[0]
    miniscope_data = np.load(trial_alignment_file, allow_pickle = True).tolist()  
    frame_rate = miniscope_data['frame_rate']
    
    #Set up the alignment for the different task phases
    aligned_to = ['outcome_presentation', 'outcome_end', 'PlayStimulus', 'DemonWaitForResponse']
    time_frame = [ np.array([round(0*frame_rate), round(1*frame_rate)+1], dtype=int),
                  np.array([round(-1*frame_rate), round(0.3*frame_rate)+1], dtype=int),
                  np.array([round(-0.5*frame_rate), round(1*frame_rate)+1], dtype=int),
                  np.array([round(-0.2*frame_rate), round(0.3*frame_rate)+1], dtype=int)]
    
    trialdata = pd.read_hdf(glob(session_dir + '/chipmunk/*.h5')[0], '/Data')
    c_data = get_chipmunk_behavior(session_dir)

    prior_choice = np.array(c_data['prior_choice'])
    prior_outcome = np.array(c_data['prior_outcome'])
     
    valid_trials = np.where(c_data['valid_past'])[0] #Retrieve valid trials with a response before and after 
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
    #Reshape to z-score the aligned signal
    Y_2d = zscore(np.reshape(Y.transpose(2,1,0), (Y.shape[2], Y.shape[1]*Y.shape[0])).T, axis=0)
    col_notnan = np.isnan(np.sum(Y_2d,axis=0))==0
    if np.sum(col_notnan) < Y_2d.shape[1]:
        Y_2d = Y_2d[:,col_notnan]
    Y_norm = np.reshape(Y_2d.T, (np.sum(col_notnan), Y.shape[1], Y.shape[0])).transpose(2,1,0)
    
    trial_average = np.mean(Y_norm, axis=1)
    centered_y = np.transpose(Y_norm.transpose(1,0,2) - trial_average, [1,0,2])
    for k in range(combos.shape[0]):
        trial_history_acti[k].append(np.mean(Y_norm[:,(prior_choice[valid_trials]==combos[k][0]) & (prior_outcome[valid_trials]==combos[k][1]),:],axis=1))
        centered_trial_history_acti[k].append(np.mean(centered_y[:,(prior_choice[valid_trials]==combos[k][0]) & (prior_outcome[valid_trials]==combos[k][1]),:],axis=1))
    prev_delta_acti[0].append(np.mean(Y_norm[:,prior_choice[valid_trials]==1,:],axis=1) - np.mean(Y_norm[:,prior_choice[valid_trials]==0,:],axis=1))
    prev_delta_acti[1].append(np.mean(Y_norm[:,prior_outcome[valid_trials]==1,:],axis=1) - np.mean(Y_norm[:,prior_outcome[valid_trials]==0,:],axis=1))                                                                                               
    
    neuron_count.append(Y_norm.shape[2])

trial_history_activity = np.stack([np.hstack(k) for k in trial_history_acti],axis=2)
centered_trial_history_activity = np.stack([np.hstack(k) for k in centered_trial_history_acti],axis=2)
prev_cho = np.hstack([np.mean(x[15:31,:],axis=0) for x in prev_delta_acti[0]])
prev_out = np.hstack([np.mean(x[15:31,:],axis=0) for x in prev_delta_acti[1]])

################################################################################
###------Optional: Make tiled plots where neurons are sorted based on their
# # peak activity after correct left choices (plotted in the first preprint version)
# p_max = np.percentile(trial_history_activity, 99)
# p_min = 0
# titles = ['Previous correct left', 'Previous incorrect left','Previous incorrect right', 'Previous correct right']
# line_idx = [30.5, 71.5, 117.5]
# idx = [np.argsort(np.argmax(trial_history_activity[:,:,k],axis=0)) for k in range(len(trial_history_acti))]
# idx_cond = [0,3]
# fi = plt.figure(figsize=[6.8,3.8])
# axes = [fi.add_subplot(1,combos.shape[0],k+1) for k in range(combos.shape[0])]
# axes[0].set_ylabel('Neuron number')
# for k in range(trial_history_activity.shape[2]):
#     im = axes[k].imshow(trial_history_activity[:,idx[idx_cond[0]],k].T,aspect='auto',cmap='gray_r', vmin=p_min, vmax=p_max)
#     [axes[k].axvline(idx, color = '#7a7a7a', linewidth=0.5, linestyle = '--') for idx in line_idx]
#     axes[k].set_title(titles[k])
#     axes[k].get_xaxis().set_visible(False)
    
#     if k > 0 :
#             axes[k].get_yaxis().set_visible(False)
  
# cbar = fi.colorbar(im, ax=np.array(axes).ravel().tolist())
# cbar.set_label('Activity (z-score)')
# cbar.set_ticks([0,0.5,1,1.5])
##############################################################################

#------- Retrieve the frame indices for the different trial phases
insertion_index = np.zeros(len(time_frame)+1)
insertion_index[1] = time_frame[0][1] - time_frame[0][0]
idx_list = [np.arange(insertion_index[1],dtype=int)]
for k in range(1,len(time_frame)):
    insertion_index[k+1] = int(insertion_index[k] + time_frame[k][1] - time_frame[k][0])
    idx_list.append(np.arange(insertion_index[k],insertion_index[k+1], dtype=int))

ax_list = []
gray = '#858585'
titles = ['Early ITI', 'Late ITI', 'Stimulus', 'Action']
for k in range(len(idx_list)):
    ax_list.append(plt.figure().add_subplot(1,1,1))
    prev_cho = np.hstack([np.mean(x[idx_list[k],:],axis=0) for x in prev_delta_acti[0]])
    prev_out = np.hstack([np.mean(x[idx_list[k],:],axis=0) for x in prev_delta_acti[1]])
    ax_list[-1].scatter(prev_cho, prev_out, s=20, c=gray, edgecolor='w', linewidth=0.5)
    ax_list[-1].axvline(0, color='k', linestyle='--', linewidth=0.5)
    ax_list[-1].axhline(0, color='k', linestyle='--', linewidth=0.5)
    ax_list[-1].set_xlabel('Previous right - left activity (SD)')
    ax_list[-1].set_ylabel('Previous correct - incorrect activity (SD)')
    ax_list[-1].set_title(titles[k])
    
    #Symmetric axes
    lim = np.max(np.abs(ax_list[-1].get_xlim()))
    ax_list[-1].set_xlim([-lim, lim])
    lim = np.max(np.abs(ax_list[-1].get_ylim()))
    ax_list[-1].set_ylim([-lim, lim])
    
#%%----Do the previous choice x outcome decoding and cross decoding for the later analyses

use_name = 'Fig2_decoding_previous_choice_outcome_combination_spr.npy'
signal_type = 'S' #Use the deconvolved calcium traces as input for the decoding
balancing_mode = 'proportional' #Mathch proportions of secondary labels within the
    # the primary label classes -> here there are no secondary labels, so this
    # is equivalent to the 'equal' mode in chiCa's balance_dataset function
subsampling_rounds = 20 #Re-draw samples 20 times from the majority class
cover_all_samples = True # Remove samples drawn from the majority class from the 
    # pool for the next round of draws until all samples have been picked, after which
    # the pool is filled again. This guarantees that all trials were used at least once
    # to train and test the decoders.
k_folds = 8 # Use 8 fold cross-validation because the for at least one session one 
    # one of the trial history contexts only occurs less than 10 times.
penalty = 'l2' #Use ridge penality for the decoding analysis
normalize = True #Z-score the activity of the neurons before fitting the decoder.
    # Here, the z-scoring is performed on both training and testing data based on the 
    # mean and std found from the training data only.
reg = 10**(np.linspace(-10,1,12)) # Search for optimal penalty over an exponentially 
    # growing sequence of values and pick best one.
model_params = {'penalty': penalty, 'solver': 'liblinear', 'inverse_regularization_strength': reg, 'fit_intercept': True, 'normalize': normalize} # Pass params as dict

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
    prior_choice = np.array(c_data['prior_choice'])
    prior_outcome = np.array(c_data['prior_outcome'])
    
    labels = np.zeros([trialdata.shape[0]])
    labels[(prior_choice==0) & (prior_outcome==1)] = 0
    labels[(prior_choice==0) & (prior_outcome==0)] = 1
    labels[(prior_choice==1) & (prior_outcome==0)] = 2
    labels[(prior_choice==1) & (prior_outcome==1)] = 3
    secondary_labels = None
   
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
    
    #Make sure the analysis directory exists before saving the model results
    if not os.path.isdir(os.path.join(session_dir, 'analysis')):
        os.mkdir(os.path.join(session_dir, 'analysis'))
    
    np.save(os.path.join(session_dir, 'analysis', use_name), out_dict)
    
#%%---Load the models

#Get list of frame indices for the different trial phases
insertion_index = np.zeros(len(time_frame)+1)
insertion_index[1] = time_frame[0][1] - time_frame[0][0]
idx_list = [np.arange(insertion_index[1],dtype=int)]
for k in range(1,len(time_frame)):
    insertion_index[k+1] = int(insertion_index[k] + time_frame[k][1] - time_frame[k][0])
    idx_list.append(np.arange(insertion_index[k],insertion_index[k+1], dtype=int))

#Retrieve model data
use_name = 'Fig2_decoding_previous_choice_outcome_combination_spr.npy' # Re-define here from section above in case the user doesn't need to fit the models
classes = 4
model_accuracy = []
shuffle_accuracy = []
label_wise_accuracy_list = []
cross_prediction = []
shuffled_cross_prediction = []
class_wise_cross_prediction = []
confusion_matrices = []
coefficients = []
for session_dir in sessions:
    decoding_data = np.load(os.path.join(session_dir, 'analysis', use_name), allow_pickle=True).tolist()
    label_wise_accuracy = np.zeros([len(decoding_data['confusion_matrix']),classes]) *np.nan
    for k in range(len(decoding_data['confusion_matrix'])): # k tracks the individual timepoints
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
    
    #Retain the entire confusion matrix
    confusion_matrices.append(np.squeeze(decoding_data['confusion_matrix']))
    
    #Now get the class-wise cross-decoding accuracy
    tmp = np.zeros([len(decoding_data['confusion_matrix']),len(decoding_data['confusion_matrix']),classes]) * np.nan
    for time_p in range(len(decoding_data['confusion_matrix'])):
        for idx in range(len(decoding_data['confusion_matrix'])):
            tmp[idx,time_p,:] = np.diagonal(decoding_data['confusion_matrix'][time_p][:,:,idx])
    class_wise_cross_prediction.append(tmp)
    
#%%------Figure 2F: Overall decoding accuracy for data and shuffle

#Get subject identifiers
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
    
dec_acc_subj = np.zeros([133, np.unique(subj_code).shape[0]]) * np.nan
shu_acc_subj = np.zeros([133, np.unique(subj_code).shape[0]]) * np.nan
for k in subj_code:
    dec_acc_subj[:, k] = np.mean(np.vstack(model_accuracy)[subj_code==k,:], axis=0)
    shu_acc_subj[:, k] = np.mean(np.vstack(shuffle_accuracy)[subj_code==k,:], axis=0)
    
# #####----Optional: Overall average over time
# gray = '#858585'   
# fi = plt.figure()
# ax = fi.add_subplot(111)
# cols = ['k', gray]
# labels = ['Model', 'Shuffle']
# plot_timecourse(ax, [dec_acc_subj, shu_acc_subj], frame_rate ,idx_list, colors = cols, line_labels=labels)
# ax.set_ylim([0,1])
# separate_axes(ax)

cols = ['w', gray]
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
    ax.scatter([0,1] + jitter[k], [np.mean(dec_acc_subj,axis=0)[k], np.mean(shu_acc_subj,axis=0)[k]], c=gray,s=14)
ax.set_ylim([0,1])
separate_axes(ax)
ax.set_ylabel('Decoding accuracy')

#Construct the data frame to export for the lme --> this is the overall accuracy vs shuffle
d_dict = dict()
d_dict['accuracy'] = np.hstack(model_accuracy).tolist() +  np.hstack(shuffle_accuracy).tolist()
d_dict['condition'] = ['model'] * np.size(np.hstack(model_accuracy)) + ['shuffle'] * np.size(np.hstack(model_accuracy))
d_dict['subject'] = np.repeat(subj_string, model_accuracy[0].shape[0]).tolist() * 2
d_dict['time'] = np.array(np.arange(model_accuracy[0].shape[0])).tolist() * (len(model_accuracy * 2))
d_dict['session'] = np.repeat(session_string, model_accuracy[0].shape[0]).tolist() * 2
df = pd.DataFrame(d_dict)

#Define your output directory analogous to the line below
#output_loc = '/Users/loesch/Documents/Churchland_lab/Trial-history_manuscript/Figures/Fig2_panels'
df.to_csv(os.path.join(output_loc, 'decoding_accuracy_decoder_vs_shuffle.csv'))


#%%-------Figure 2G: Class-wise decoding accuracy over time

tmp_acc = np.squeeze(label_wise_accuracy_list)
label_wise_acc_subj = np.zeros([tmp_acc.shape[1],np.unique(subj_code).shape[0], 4]) * np.nan
for k in subj_code:
    label_wise_acc_subj[:, k,:] = np.mean(tmp_acc[subj_code==k,:], axis=0)

fi = plt.figure(figsize= [8,4.8])
ax = fi.add_subplot(111)
cols = ['#606cbe', '#48a7b1', '#deb35e', '#de8468',gray]
spacer = 6
labels = ['Previous correct left', 'Previous incorrect left', 'Previous incorrect right', 'Previous correct right', 'Shuffle']
plot_timecourse(ax, [k.T for k in label_wise_acc_subj.transpose(2,1,0)] + [shu_acc_subj] , frame_rate ,idx_list,
                colors = cols, line_labels=labels, spacer=spacer, x_label = 'Time (s)', y_label = 'Decoding accuracy' )
ax.set_ylim([0.2,1]) #limit the range to not have too much white space
separate_axes(ax)

#Construct the data frame to export the data. Here, test for differences between the different contexts
d_dict = dict()
d_dict['accuracy'] = np.reshape(tmp_acc.transpose(0,2,1), np.size(tmp_acc)).tolist()
d_dict['time'] = np.arange(tmp_acc.shape[1]).tolist() * (tmp_acc.shape[0] * tmp_acc.shape[2])
tmp = []
for k in ['Previous correct left', 'Previous incorrect left', 'Previous incorrect right', 'Previous correct right']:
    tmp = tmp + [k] * tmp_acc.shape[1]
d_dict['history_context'] = tmp * tmp_acc.shape[0]
tmp = []
for ses in sessions:
    tmp = tmp +  [os.path.split(os.path.split(ses)[0])[1]] * (tmp_acc.shape[1] * tmp_acc.shape[2])
d_dict['subject'] = tmp
d_dict['session'] = np.repeat(session_string, np.size(label_wise_acc_subj[:,0,:]))
df = pd.DataFrame(d_dict)

#output_loc = '/Users/loesch/Documents/Churchland_lab/Trial-history_manuscript/Figures/Fig2_panels'
df.to_csv(os.path.join(output_loc, 'class_wise_decoding_accuracy.csv'))

#%%-----Figure 2H: Plot confusion matrices from different timepoints on the same figure

# -> Pushed to chiCa.visualization_utils
# def white_to_color(rgb_color):
#     '''Generate a colormap that start with white and ends with
#     a desired color.'''
#     from matplotlib.colors import ListedColormap
#     N = 256
#     vals = np.ones((N, 4))
#     vals[:, 0] = np.linspace(1,rgb_color[0]/256, N)
#     vals[:, 1] = np.linspace(1,rgb_color[1]/256, N)
#     vals[:, 2] = np.linspace(1,rgb_color[2]/256, N)
#     newcmp = ListedColormap(vals)
#     return newcmp

newcmp = white_to_color((10, 10, 77))

confusion_timepoints = [0,15,75]
con_mats = np.zeros([classes,classes,len(confusion_timepoints)]) * np.nan
for k in range(len(confusion_timepoints)):
    all_con_mats = np.squeeze([x[confusion_timepoints[k],:,:,confusion_timepoints[k]] for x in confusion_matrices])
    tmp = [np.mean(all_con_mats[subj_code==x,:,:],axis=0) for x in np.unique(subj_code)] # Average over subjects first
    con_mats[:,:,k] = np.mean(np.squeeze(tmp),axis=0)
    
titles = ['t = 0 s','t = 0.5 s', 't > 2 s']
fi = plt.figure()
ax = [fi.add_subplot(len(confusion_timepoints),1,k+1, aspect='equal') for k in range(con_mats.shape[2])]
#ax = [fi.add_subplot(int(len(confusion_timepoints)/2),int(len(confusion_timepoints)/2),k+1, aspect='equal') for k in range(con_mats.shape[2])]
for k in range(con_mats.shape[2]):
    #axes[k] = fi.add_subplot(1,2,k+1, aspect='equal')
    im = ax[k].imshow(con_mats[:,:,k], cmap = newcmp, vmin=0, vmax=1)
    ax[k].set_title(titles[k])
    ax[k].tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
    
ax[-1].set_xlabel('Predicted labels')
ax[-1].set_ylabel('True labels')
cbar = fi.colorbar(im, ax=ax)
cbar.set_label('Fraction of true labels')
cbar.set_ticks([0,0.25,0.5,0.75,1])

#%%---Figure 2I: Average cross-decoding accuracy matrix over all mice

#Average cross-decoding accuracy across sessions within subject first
tmp = [np.mean(np.squeeze(cross_prediction)[subj_code==x,:,:],axis=0) for x in np.unique(subj_code)] # Average over subjects first
cross_pred_acc = np.mean(np.squeeze(tmp),axis=0)

#Specify the position of the dashed lines indicating the separation between different task phases
line_idx = [idx_list[k][-1] + 0.5 for k in range(len(idx_list)-1)]

#Generate a pale yellow-green to dark blue colormap
hex_cols = ['#ffffd6',
 '#ecf8af',
 '#ddf1b1',
 '#c7e9b4',
 '#7fccba',
 '#44b6c5',
 '#1b91c0',
 '#1e77b3',
 '#205fa7',
 '#253393',
 '#070d55']
from matplotlib.colors import ListedColormap
col_map = matplotlib.colors.LinearSegmentedColormap.from_list("", hex_cols)

fi = plt.figure()
ax = fi.add_subplot(111)
im = ax.imshow(cross_pred_acc.T, vmin=0, vmax=1, cmap = col_map)
for k in line_idx:
    ax.axvline(k, color = 'k', linewidth=0.5, linestyle='--')
    ax.axhline(k, color = 'k', linewidth=0.5, linestyle='--')
# ax.set_xticks([30.5, 60.5, 90.5, 120.5]) # Match the ticks with trial phase transitions
# ax.set_yticks([30.5, 60.5, 90.5, 120.5])
ax.set_ylabel('Training time points')
ax.set_xlabel('Testing time points')
cbar = fi.colorbar(im)
cbar.set_label('Decoding accuracy')
cbar.set_ticks([0,0.5,1])

#############################################################
# Optional: Just for validation show cross-prediction for different label types
# class_wise =  []
# for k in range(classes):
#     tmp = []
#     for n in range(len(class_wise_cross_prediction)):
#         tmp.append(class_wise_cross_prediction[n][:,:,k])
#     tmp_s = [np.mean(np.squeeze(tmp)[subj_code==x,:,:],axis=0) for x in np.unique(subj_code)] # Average over subjects first
#     class_wise.append(np.mean(np.squeeze(tmp_s),axis=0))

# titles =  ['Previous correct left', 'Previous incorrect left', 'Previous incorrect right', 'Previous correct right']
# for n in range(classes):
#     fi = plt.figure()
#     ax = fi.add_subplot(111)
#     im = ax.imshow(class_wise[n].T, vmin=0, vmax=1)
#     for k in line_idx:
#         ax.axvline(k, color = 'w', linewidth=0.5, linestyle='--')
#         ax.axhline(k, color = 'w', linewidth=0.5, linestyle='--')
#     ax.set_xlabel('Training time points')
#     ax.set_ylabel('Testing time points')
#     cbar = fi.colorbar(im)
#     cbar.set_label('Decoding accuracy')
#     cbar.set_ticks([0,0.5,1])
#     ax.set_title(titles[n])
##############################################################################  
  
#%%-----Figure 2J: Compare the average cross-decoding accuracy for each phase
# with all the other phases. Here, "seed" refers to the phase for which the 
# decoders were trained while "phase" stands for the task phase of testing.

cross_prediction_subj = np.squeeze([np.mean(np.squeeze(cross_prediction)[subj_code==k,:,:],axis=0) for k in np.unique(subj_code)])
shuffled_prediction_subj = np.squeeze([np.mean(np.squeeze(shuffled_cross_prediction)[subj_code==k,:,:],axis=0) for k in np.unique(subj_code)])

cross_acc_full = []
shu_acc_full = []
cross_accuracy_by_phase = np.zeros([np.unique(subj_code).shape[0], len(idx_list), len(idx_list)])
shuffled_accuracy_by_phase = np.zeros([np.unique(subj_code).shape[0], len(idx_list), len(idx_list)])
for k in range(np.unique(subj_code).shape[0]):
    for seed in range(len(idx_list)):
        for phase in  range(len(idx_list)):
            cross_accuracy_by_phase[k, seed, phase] = np.mean(cross_prediction_subj[k,idx_list[seed][0]:idx_list[seed][-1]+1,idx_list[phase][0]:idx_list[phase][-1]+1])
            shuffled_accuracy_by_phase[k, seed, phase] = np.mean(shuffled_prediction_subj[k,idx_list[seed][0]:idx_list[seed][-1]+1,idx_list[phase][0]:idx_list[phase][-1]+1])
            
import matplotlib
cm = matplotlib.cm.get_cmap('Greys')
cols = [cm(k) for k in np.flip(np.arange(0,1,1/len(idx_list)))]

ax = plt.figure().add_subplot()
for k in range(cross_accuracy_by_phase.shape[2]):
    sem = np.std(cross_accuracy_by_phase[:,k,:],axis=0) / np.sqrt(cross_accuracy_by_phase.shape[0])
    ax.bar(np.arange(len(idx_list)) + 0.6 + k*0.2, np.mean(cross_accuracy_by_phase[:,k,:],axis=0), width = 0.2, color = cols[k], yerr=sem, capsize=4, edgecolor = 'k', align='edge')
ax.axhline(0.25, color='#940a0a', linestyle='--', linewidth=0.5)
ax.set_ylim([0,1])
separate_axes(ax)
ax.set_xticks(np.arange(1,len(idx_list)+1))
ax.set_xticklabels(['Early ITI', 'Late ITI', 'Stimulus', 'Action'], rotation=45, ha="right", rotation_mode="anchor")
ax.set_ylabel('Cross-decoding acuracy')

#Construct the data frame for the stats
cross_acc_full = np.zeros([len(sessions), len(idx_list), len(idx_list)])
shu_acc_full = np.zeros([len(sessions), len(idx_list), len(idx_list)])
for k in range(len(sessions)):
    for seed in range(len(idx_list)):
        for phase in  range(len(idx_list)):
            cross_acc_full[k, seed, phase] = np.mean(cross_prediction[k][idx_list[seed][0]:idx_list[seed][-1]+1,idx_list[phase][0]:idx_list[phase][-1]+1])
            shu_acc_full[k, seed, phase] = np.mean(shuffled_cross_prediction[k][idx_list[seed][0]:idx_list[seed][-1]+1,idx_list[phase][0]:idx_list[phase][-1]+1])

d_dict = dict()
d_dict['accuracy'] = np.hstack((cross_acc_full.flatten(), shu_acc_full.flatten()))
d_dict['phase'] = ['Early ITI', 'Late ITI', 'Stimulus', 'Action'] * (np.size(cross_acc_full[:,:,0]) * 2)
d_dict['seed'] = np.repeat(['Early ITI', 'Late ITI', 'Stimulus', 'Action'], cross_acc_full.shape[2]).tolist() * (cross_acc_full.shape[0] * 2)
d_dict['subject'] = np.repeat(subj_string, np.size(cross_acc_full[0,:,:])).tolist() * 2
d_dict['session'] = np.repeat(session_string, np.size(cross_acc_full[0,:,:])).tolist() * 2
d_dict['condition'] = ['model'] * np.size(cross_acc_full) + ['shuffle'] * np.size(cross_acc_full)
df = pd.DataFrame(d_dict)
#output_loc = '/Users/loesch/Documents/Churchland_lab/Trial-history_manuscript/Figures/Fig2_panels'
df.to_csv(os.path.join(output_loc, 'phase_wise_cross_decoding.csv'))          
  
########################################################################      
# Optional: Simply run correlations of the deocder coefficients
# corr_mats = np.zeros([coefficients[0].shape[0], coefficients[0].shape[0], classes, len(coefficients)])
# for n in range(len(coefficients)):
#     for k in range(classes):
#         corr_mats[:,:,k,n] = np.corrcoef(coefficients[n][:,:,k])


# titles =  ['Previous correct left', 'Previous incorrect left', 'Previous incorrect right', 'Previous correct right']
# for n in range(classes):
#     fi = plt.figure()
#     ax = fi.add_subplot(111)
#     im = ax.imshow(np.mean(corr_mats[:,:,n,:],axis=2).T, vmin=-1, vmax=1, cmap = 'RdBu_r')
#     for k in line_idx:
#         ax.axvline(k, color = 'k', linewidth=0.5, linestyle='--')
#         ax.axhline(k, color = 'k', linewidth=0.5, linestyle='--')
#     ax.set_xlabel('Training time points')
#     ax.set_ylabel('Testing time points')
#     cbar = fi.colorbar(im)
#     cbar.set_label('Correlation coefficient')
#     cbar.set_ticks([-1,0,1])
#     ax.set_title(titles[n])
    
