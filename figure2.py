# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 13:29:13 2025

@author: Lukas Oesch
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

#%%-----Include these sessions into the analysis

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



#%%----Load cell data for different sessions

#------Figure 2e-------
ses_idx = [0,1,7,9,14,17,18,20,22] #Include the session with most recorded neurons for each mouse

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


#--------------------------------
#%% ----- Make scatter plot for 

# corr_acti = np.zeros([trial_history_activity.shape[1],2]) * np.nan
# incorr_acti = np.zeros([trial_history_activity.shape[1],2]) * np.nan

# c_idx = [0,3]
# i_idx = [1,2]
# for k in range(2):
#     corr_acti[:,k] = np.mean(trial_history_activity[15:31,:,c_idx[k]],axis=0)
#     incorr_acti[:,k] = np.mean(trial_history_activity[15:31,:,i_idx[k]],axis=0)
    
# fi = plt.figure()
# ax = fi.add_subplot(111)
# ax.scatter(corr_acti[:,0], corr_acti[:,1], s=6)

# fi = plt.figure()
# ax = fi.add_subplot(111)
# ax.scatter(incorr_acti[:,0], incorr_acti[:,1], s=6)



# p_min = np.min(trial_history_activity)
# p_max = np.percentile(trial_history_activity, 99)

# p_min = 0
# # p_max=5
# titles = ['Previous correct left', 'Previous incorrect left','Previous incorrect right', 'Previous correct right']
# line_idx = [30.5, 71.5, 117.5]
# idx = [np.argsort(np.argmax(trial_history_activity[:,:,k],axis=0)) for k in range(len(trial_history_acti))]
# idx_cond = [0,3]
# fi = plt.figure(figsize=[6.8,4.8])
# axes = [fi.add_subplot(2,combos.shape[0],k+1) for k in range(2*combos.shape[0])]
# axes[0].set_ylabel('Neuron number')
# axes[4].set_ylabel('Neuron number')
# for k in range(trial_history_activity.shape[2]):
#     im = axes[k].imshow(trial_history_activity[:,idx[idx_cond[0]],k].T,aspect='auto',cmap='gray_r', vmin=p_min, vmax=p_max)
#     [axes[k].axvline(idx, color = '#7a7a7a', linewidth=0.5, linestyle = '--') for idx in line_idx]
#     axes[k].set_title(titles[k])
#     axes[k].get_xaxis().set_visible(False)
    
#     im = axes[k+4].imshow(trial_history_activity[:,idx[idx_cond[1]],k].T,aspect='auto',cmap='gray_r', vmin=p_min, vmax=p_max)
#     [axes[k+4].axvline(idx, color = '#7a7a7a', linewidth=0.5, linestyle = '--') for idx in line_idx]
#     axes[k+4].set_xlabel('Frames')
#     if k > 0 :
#             axes[k].get_yaxis().set_visible(False)
#             axes[k+4].get_yaxis().set_visible(False)
  
# cbar = fi.colorbar(im, ax=np.array(axes).ravel().tolist())
# cbar.set_label('Activity (z-score)')

p_min = np.percentile(trial_history_activity)
p_max = np.percentile(trial_history_activity, 99)
p_min = 0
# p_max=5
titles = ['Previous correct left', 'Previous incorrect left','Previous incorrect right', 'Previous correct right']
line_idx = [30.5, 71.5, 117.5]
idx = [np.argsort(np.argmax(trial_history_activity[:,:,k],axis=0)) for k in range(len(trial_history_acti))]
idx_cond = [0,3]
fi = plt.figure(figsize=[6.8,3.8])
axes = [fi.add_subplot(1,combos.shape[0],k+1) for k in range(combos.shape[0])]
axes[0].set_ylabel('Neuron number')
for k in range(trial_history_activity.shape[2]):
    im = axes[k].imshow(trial_history_activity[:,idx[idx_cond[0]],k].T,aspect='auto',cmap='gray_r', vmin=p_min, vmax=p_max)
    [axes[k].axvline(idx, color = '#7a7a7a', linewidth=0.5, linestyle = '--') for idx in line_idx]
    axes[k].set_title(titles[k])
    axes[k].get_xaxis().set_visible(False)
    
    if k > 0 :
            axes[k].get_yaxis().set_visible(False)
  
cbar = fi.colorbar(im, ax=np.array(axes).ravel().tolist())
cbar.set_label('Activity (z-score)')
cbar.set_ticks([0,0.5,1,1.5])



#Do the same with the centered version too
p_min = np.percentile(centered_trial_history_activity,1)
p_max = np.percentile(centered_trial_history_activity, 99)

titles = ['Previous correct left', 'Previous incorrect left','Previous incorrect right', 'Previous correct right']
line_idx = [30.5, 71.5, 117.5]
idx = [np.argsort(np.argmax(centered_trial_history_activity[:,:,k],axis=0)) for k in range(len(trial_history_acti))]
idx_cond = [0,3]
fi = plt.figure(figsize=[6.8,3.8])
axes = [fi.add_subplot(1,combos.shape[0],k+1) for k in range(combos.shape[0])]
axes[0].set_ylabel('Neuron number')
for k in range(centered_trial_history_activity.shape[2]):
    im = axes[k].imshow(centered_trial_history_activity[:,idx[idx_cond[0]],k].T,aspect='auto',cmap='gray_r', vmin=p_min, vmax=p_max)
    [axes[k].axvline(idx, color = '#7a7a7a', linewidth=0.5, linestyle = '--') for idx in line_idx]
    axes[k].set_title(titles[k])
    axes[k].get_xaxis().set_visible(False)
    
    if k > 0 :
            axes[k].get_yaxis().set_visible(False)
  
cbar = fi.colorbar(im, ax=np.array(axes).ravel().tolist())
cbar.set_label('Activity (z-score)')

#---------
insertion_index = np.zeros(len(time_frame)+1)
insertion_index[1] = time_frame[0][1] - time_frame[0][0]
idx_list = [np.arange(insertion_index[1],dtype=int)]
for k in range(1,len(time_frame)):
    insertion_index[k+1] = int(insertion_index[k] + time_frame[k][1] - time_frame[k][0])
    idx_list.append(np.arange(insertion_index[k],insertion_index[k+1], dtype=int))


ax_list = []
#ax = plt.figure().add_subplot(111)
titles = ['Early ITI', 'Late ITI', 'Stimulus', 'Action']
for k in range(len(idx_list)):
    ax_list.append(plt.figure().add_subplot(1,1,1))
    prev_cho = np.hstack([np.mean(x[idx_list[k],:],axis=0) for x in prev_delta_acti[0]])
    prev_out = np.hstack([np.mean(x[idx_list[k],:],axis=0) for x in prev_delta_acti[1]])
    ax_list[-1].scatter(prev_cho, prev_out, s=20, c=gray, edgecolor='w', linewidth=0.5)
    ax_list[-1].axvline(0, color='k', linestyle='--', linewidth=0.5)
    ax_list[-1].axhline(0, color='k', linestyle='--', linewidth=0.5)
    ax_list[-1].set_title(titles[k])
    
    #Symmetric axes
    lim = np.max(np.abs(ax_list[-1].get_xlim()))
    ax_list[-1].set_xlim([-lim, lim])
    lim = np.max(np.abs(ax_list[-1].get_ylim()))
    ax_list[-1].set_ylim([-lim, lim])
    # d = {'prev_cho': prev_cho, 'prev_out': prev_out}
    # df = pd.DataFrame(d)
    # plt.sca(ax)
    # sns.kdeplot(
    #     data=df, x="prev_cho", y="prev_out", fill=False)
    
#%%----Do the previous choice x outcome decoding and cross decoding


use_name = 'Fig2_decoding_previous_choice_outcome_combination_spr.npy'
signal_type = 'S'
balancing_mode = 'proportional'
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
    prior_choice = np.array(c_data['prior_choice'])
    prior_outcome = np.array(c_data['prior_outcome'])
    
    labels = np.zeros([trialdata.shape[0]])
    labels[(prior_choice==0) & (prior_outcome==1)] = 0
    labels[(prior_choice==0) & (prior_outcome==0)] = 1
    labels[(prior_choice==1) & (prior_outcome==0)] = 2
    labels[(prior_choice==1) & (prior_outcome==1)] = 3
    secondary_labels = None
    #secondary_labels = np.array(c_data['choice'])
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
    
    np.save(os.path.join(session_dir, 'analysis', use_name), out_dict)
    
#%%----Start the plotting of the example neurons from LO069 
session_dir = sessions[15]
classes = 4
#cols = ['#2f65d0', '#2fd0aa', '#d09a2f', '#d04a2f']
cols = ['#606cbe', '#48a7b1', '#deb35e', '#de8468']
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

choice = c_data['choice']
outcome = c_data['outcome']
#secondary_labels = np.array(c_data['choice'])
valid_trials = np.where(c_data['valid_past'])[0] #Retrieve valid trials with a response before and after 
valid_trials_before = valid_trials-1
val_trials =  [valid_trials_before]*2 + [valid_trials]*2 #Combine trial identities for the ITI before the new trial and once it has started

if signal_type == 'S':
      signal = gaussian_filter1d(miniscope_data[signal_type].T, 1, axis=0, mode='constant', cval=0, truncate=4)
      #Pad the first and last samples with zeros in this condition
else:
      signal = miniscope_data[signal_type].T #Transpose the original signal so that rows are timepoints and columns cells
  
#Decide whether to z-swcore on the entire trace or only on the trial aligned portions
signal = zscore(signal,axis=0)
#Determine which neurons to include in the analysis
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


#Select the neurons of interest and plot their PSTHs
#neuron_id = [17,186,187,133]
#neuron_id = [16,133,129,154]
#neuron_id = [16,133,181,154]
#neuron_id = [16,280,254,154]
#neuron_id = [16,280,291,154]

#this is the full set
#neuron_id = [154,155,151,16,185,183,181]
#but we'll remove 183 because it's activity is very high
neuron_id = [154,16,181]
ax = []
for n in neuron_id:
    ax.append(plt.figure(figsize=[8,4.8]).add_subplot())
    #plot_timecourse(ax[-1], [Y_norm[:,labels[valid_trials]==k,n] for k in range(classes)], frame_rate ,idx_list, colors = cols)
    plot_timecourse(ax[-1], [Y_norm[:,labels[valid_trials]==k,n] for k in range(classes)], frame_rate ,idx_list, colors = cols)
    #plot_timecourse(ax[-1], [Y[:,labels[valid_trials]==k,n] for k in range(classes)], frame_rate ,idx_list, colors = cols)
    ax[-1].set_title(f'neuron {n}')

lims = [k.get_ylim() for k in ax]
common_lim = [np.min(np.squeeze(lims)[:,0]), np.max(np.squeeze(lims)[:,1])]
for k in range(len(ax)):
    ax[k].set_ylim(common_lim)
    separate_axes(ax[k])


#Plot the continuous traces with the chocie-outcome events
import matplotlib.patches as patches
# ax = plt.figure().add_subplot(111)
# ax.plot(zscore(miniscope_data['F'][neuron_id,:].T), color = 'k', linewidth=1)
#neuron_id = [16,181,291,154,151,146]
#neuron_id = [16,185,181,183,154,155,151]

neuron_id = [154,155,151,16,185,183,181]
sig_spacing = [0,-6, -13,-23, -29,-35,-40]
ax = plt.figure(figsize=[6,4.8]).add_subplot()
for k in range(len(neuron_id)):
    ax.plot(zscore(miniscope_data['F'][neuron_id[k],:].T) + sig_spacing[k],color='k')
    #ax.plot(signal[:,neuron_id[k]] + 6*k, color = 'k')
ylims = ax.get_ylim()
box_cols = cols[:4]
state_start_frame, state_time_covered = find_state_start_frame_imaging('outcome_presentation', trialdata, miniscope_data['frame_interval'], miniscope_data['trial_starts'], miniscope_data['trial_start_time_covered'])         
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

#%%------Plot the neuron contours for LO028

session_dir = sessions[0]
trial_alignment_file = glob(session_dir + '/analysis/*miniscope_data.npy')[0]
miniscope_data = np.load(trial_alignment_file, allow_pickle = True).tolist()

#Retrieve the spatial footprints
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
   # masked_data = np.ma.masked_array(A[:,:,k], mask=(A[:,:,k] < 0.1))
   # ax.imshow(np.flipud(masked_data))
   #Flip the image
   ax.contour(np.flipud(A[:,:,k]))


#%%---Load the models and plot


#Get list of indices
insertion_index = np.zeros(len(time_frame)+1)
insertion_index[1] = time_frame[0][1] - time_frame[0][0]
idx_list = [np.arange(insertion_index[1],dtype=int)]
for k in range(1,len(time_frame)):
    insertion_index[k+1] = int(insertion_index[k] + time_frame[k][1] - time_frame[k][0])
    idx_list.append(np.arange(insertion_index[k],insertion_index[k+1], dtype=int))

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
for session_dir in sessions:
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
    
    #Confusion matrices
    # tmp = []
    # for t in confusion_timepoints:
    #     tmp.append(decoding_data['confusion_matrix'][t][:,:,t])
    confusion_matrices.append(np.squeeze(decoding_data['confusion_matrix']))
    
    #Now get the class wise decoding accuracy
    tmp = np.zeros([len(decoding_data['confusion_matrix']),len(decoding_data['confusion_matrix']),classes]) * np.nan
    for time_p in range(len(decoding_data['confusion_matrix'])):
        for idx in range(len(decoding_data['confusion_matrix'])):
            tmp[idx,time_p,:] = np.diagonal(decoding_data['confusion_matrix'][time_p][:,:,idx])
    class_wise_cross_prediction.append(tmp)
    
#%%------Now start all the plotting 
#

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
    
#Overall average over time
gray = '#858585'   
fi = plt.figure()
ax = fi.add_subplot(111)
cols = ['k', gray]
labels = ['Model', 'Shuffle']
plot_timecourse(ax, [dec_acc_subj, shu_acc_subj], frame_rate ,idx_list, colors = cols, line_labels=labels)
ax.set_ylim([0,1])
separate_axes(ax)

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
# av = [np.mean(np.squeeze(model_accuracy).T), np.mean(np.squeeze(shuffle_accuracy).T)]
# sem = [np.std(np.mean(np.squeeze(model_accuracy).T,axis=0)) / np.sqrt(len(model_accuracy)), np.std(np.mean(np.squeeze(shuffle_accuracy).T,axis=0)) /  np.sqrt(len(model_accuracy))]
# err = ax.errorbar(labels, av, yerr=sem, color = 'k', fmt='o', capsize=4)
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
output_loc = '/Users/loesch/Documents/Churchland_lab/Trial-history_manuscript/Figures/Fig2_panels'
df.to_csv(os.path.join(output_loc, 'decoding_accuracy_decoder_vs_shuffle.csv'))
#-----------------------------------------

#calss-wise accuracy over time
tmp_acc = np.squeeze(label_wise_accuracy_list)
label_wise_acc_subj = np.zeros([tmp_acc.shape[1],np.unique(subj_code).shape[0], 4]) * np.nan
for k in subj_code:
    label_wise_acc_subj[:, k,:] = np.mean(tmp_acc[subj_code==k,:], axis=0)

fi = plt.figure(figsize= [8,4.8])
ax = fi.add_subplot(111)
#cols = ['#2f65d0', '#2fd0aa', '#d09a2f', '#d04a2f', gray]
cols = ['#606cbe', '#48a7b1', '#deb35e', '#de8468',gray]
spacer = 6
labels = ['Previous correct left', 'Previous incorrect left', 'Previous incorrect right', 'Previous correct right', 'Shuffle']
plot_timecourse(ax, [k.T for k in label_wise_acc_subj.transpose(2,1,0)] + [shu_acc_subj] , frame_rate ,idx_list, colors = cols, line_labels=labels, spacer=spacer )
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
output_loc = '/Users/loesch/Documents/Churchland_lab/Trial-history_manuscript/Figures/Fig2_panels'
df.to_csv(os.path.join(output_loc, 'class_wise_decoding_accuracy.csv'))
#------------------------------------------------------------------------------

#Plot the two confusion matrices on the same plot

def white_to_color(rgb_color):
    '''Generate a colormap that start with white and ends with
    a desired color.'''
    from matplotlib.colors import ListedColormap
    N = 256
    vals = np.ones((N, 4))
    vals[:, 0] = np.linspace(1,rgb_color[0]/256, N)
    vals[:, 1] = np.linspace(1,rgb_color[1]/256, N)
    vals[:, 2] = np.linspace(1,rgb_color[2]/256, N)
    newcmp = ListedColormap(vals)
    return newcmp

# newcmp = white_to_color(rgb(40, 5, 113))

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
ax[0].tick_params(labelbottom=False, bottom=False)
cbar = fi.colorbar(im, ax=ax)
cbar.set_label('Fraction of true labels')
cbar.set_ticks([0,0.25,0.5,0.75,1])

#---Plot cross prediction now
tmp = [np.mean(np.squeeze(cross_prediction)[subj_code==x,:,:],axis=0) for x in np.unique(subj_code)] # Average over subjects first
cross_pred_acc = np.mean(np.squeeze(tmp),axis=0)
line_idx = [idx_list[k][-1] + 0.5 for k in range(len(idx_list)-1)]

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
col_map = matplotlib.colors.LinearSegmentedColormap.from_list("", hex_cols)

fi = plt.figure()
ax = fi.add_subplot(111)
im = ax.imshow(cross_pred_acc.T, vmin=0, vmax=1, cmap = col_map)
for k in line_idx:
    ax.axvline(k, color = 'k', linewidth=0.5, linestyle='--')
    ax.axhline(k, color = 'k', linewidth=0.5, linestyle='--')
ax.set_xticks([30.5, 60.5, 90.5, 120.5])
ax.set_yticks([30.5, 60.5, 90.5, 120.5])
ax.set_ylabel('Training time points')
ax.set_xlabel('Testing time points')
cbar = fi.colorbar(im)
cbar.set_label('Decoding accuracy')
#cbar.set_ticks([0,0.5,1])


#---Just for validation show cross-prediction for different label types
class_wise =  []
for k in range(classes):
    tmp = []
    for n in range(len(class_wise_cross_prediction)):
        tmp.append(class_wise_cross_prediction[n][:,:,k])
    tmp_s = [np.mean(np.squeeze(tmp)[subj_code==x,:,:],axis=0) for x in np.unique(subj_code)] # Average over subjects first
    class_wise.append(np.mean(np.squeeze(tmp_s),axis=0))

titles =  ['Previous correct left', 'Previous incorrect left', 'Previous incorrect right', 'Previous correct right']
for n in range(classes):
    fi = plt.figure()
    ax = fi.add_subplot(111)
    im = ax.imshow(class_wise[n].T, vmin=0, vmax=1)
    for k in line_idx:
        ax.axvline(k, color = 'w', linewidth=0.5, linestyle='--')
        ax.axhline(k, color = 'w', linewidth=0.5, linestyle='--')
    ax.set_xlabel('Training time points')
    ax.set_ylabel('Testing time points')
    cbar = fi.colorbar(im)
    cbar.set_label('Decoding accuracy')
    cbar.set_ticks([0,0.5,1])
    ax.set_title(titles[n])
    
   
    
    
#------this is some quite tricky an analysis    
    
    
# supra_thresh = np.zeros([prediction_accuracy.shape[0], prediction_accuracy.shape[1]])
# for k in range(len(decoding_models)):
#     threshold = 2*np.std(decoding_models[k]['model_accuracy'])
#     supra_thresh[:,k] = prediction_accuracy[:,k] >= threshold


# dist_shuffled_acc = shuffled_prediction_accuracy.flatten()
# alpha_level = 0.05
# significant_prctile = 1 - 0.5 *(alpha_level / len(decoding_models)**2)
# threshold = np.percentile(dist_shuffled_acc,significant_prctile*100)


# significant_decoding = np.array(prediction_accuracy > threshold)
    
# t_l = []

# for k in significant_decoding.shape[0]:
#     idx = np.where(significant_decoding[k,:] > 0)[0]
#     seq = []
    
#     last_break = idx[0]
#     for n in range(idx.shape[0]-1):
#         if idx[n] != idx[n+1] + 1:
#             seq.append(idx[n] - last_break)
#             last_break = idx[n]
       
        
#     if len(seq) = 0:
    
    
    
#-----Okay, just compare the overall decoding accuracy between the different phases of the trial

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
            
            
            
######-------Can't take only upper triangular because the matrix is not 
#necessarily symmetrical
# cross_prediction_subj = np.squeeze([np.mean(np.squeeze(cross_prediction)[subj_code==k,:,:],axis=0) for k in np.unique(subj_code)])
# row_indices, column_indices = np.triu_indices_from(cross_prediction_subj[0], k=1) #Get indices of upper triangular excluding diagonal

# seed_indices = []
# for seed in idx_list:
#     phase_indices = []
#     for phase in idx_list:
#         phase_indices.append([x for x in range(column_indices.shape[0]) if (column_indices[x] in seed) and (row_indices[x] in phase)])
#     seed_indices.append(phase_indices)

# cross_accuracy_by_phase = np.zeros([cross_prediction_subj.shape[0], len(idx_list), len(idx_list)])
# for subj in range(cross_prediction_subj.shape[0]):
#     tmp_complete = []
#     for k in range(row_indices.shape[0]):
#         for n in range(len(idx_list)):
#             if column_indices[k] in idx_list[n]:
                
#                 for p in range(len(idx_list)):
#                     if row_indices[k] in idx_list[p]:
#                         tmp[n][p].extend([cross_prediction_subj[subj,row_indices[k],column_indices[k]]])
#############
import matplotlib
cm = matplotlib.cm.get_cmap('Greys')
cols = [cm(k) for k in np.flip(np.arange(0,1,1/len(idx_list)))]
# ax = plt.figure().add_subplot()
# for k in range(cross_accuracy_by_phase.shape[2]):
#     fancy_boxplot(ax,np.squeeze(cross_accuracy_by_phase[:,k,:]), [cols[k]]*len(idx_list), widths = 0.2, positions = np.arange(len(idx_list)) + 0.6 + k*0.2)

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


# #Construct the data frame for the stats
# d_dict = dict()
# d_dict['accuracy'] = np.hstack((cross_accuracy_by_phase.flatten(), shuffled_accuracy_by_phase.flatten()))
# d_dict['phase'] = ['Early ITI', 'Late ITI', 'Stimulus', 'Action'] * (np.size(cross_accuracy_by_phase[:,:,0]) * 2)
# d_dict['seed'] = np.repeat(['Early ITI', 'Late ITI', 'Stimulus', 'Action'], cross_accuracy_by_phase.shape[2]).tolist() * (cross_accuracy_by_phase.shape[0] * 2)
# d_dict['subject'] = np.repeat(np.unique(subj_string).tolist(), np.size(cross_accuracy_by_phase[0,:,:])).tolist() * 2
# d_dict['session'] = np.repeat(session_string).tolist(), np.size(cross_accuracy_by_phase[0,:,:])).tolist() * 2
# d_dict['condition'] = ['model'] * np.size(cross_accuracy_by_phase) + ['shuffle'] * np.size(cross_accuracy_by_phase)

# # tmp = []
# # for seed in  ['Early ITI', 'Late ITI', 'Stimulus', 'Action']:
# #     tmp = tmp +  [seed] * cross_accuracy_by_phase.shape[2]
# # d_dict['seed'] = tmp * cross_accuracy_by_phase.shape[0] 
# # d_dict['phase'] = ['Early ITI', 'Late ITI', 'Stimulus', 'Action'] * (cross_accuracy_by_phase.shape[0] * cross_accuracy_by_phase.shape[1])
# # tmp = []
# # for ses in sessions:
# #     tmp = tmp +  [os.path.split(os.path.split(ses)[0])[1]] * (cross_accuracy_by_phase.shape[1] *cross_accuracy_by_phase.shape[2])
# # d_dict['subject'] = tmp 
# df = pd.DataFrame(d_dict)
# output_loc = '/Users/loesch/Documents/Churchland_lab/Trial-history_manuscript/Figures/Fig2_panels'
# df.to_csv(os.path.join(output_loc, 'phase_wise_cross_decoding.csv'))


#Construct the data frame
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
output_loc = '/Users/loesch/Documents/Churchland_lab/Trial-history_manuscript/Figures/Fig2_panels'
df.to_csv(os.path.join(output_loc, 'phase_wise_cross_decoding.csv'))          
            



        
#------Simply run correlations of the deocder coefficients
corr_mats = np.zeros([coefficients[0].shape[0], coefficients[0].shape[0], classes, len(coefficients)])
for n in range(len(coefficients)):
    for k in range(classes):
        corr_mats[:,:,k,n] = np.corrcoef(coefficients[n][:,:,k])


titles =  ['Previous correct left', 'Previous incorrect left', 'Previous incorrect right', 'Previous correct right']
for n in range(classes):
    fi = plt.figure()
    ax = fi.add_subplot(111)
    im = ax.imshow(np.mean(corr_mats[:,:,n,:],axis=2).T, vmin=-1, vmax=1, cmap = 'RdBu_r')
    for k in line_idx:
        ax.axvline(k, color = 'k', linewidth=0.5, linestyle='--')
        ax.axhline(k, color = 'k', linewidth=0.5, linestyle='--')
    ax.set_xlabel('Training time points')
    ax.set_ylabel('Testing time points')
    cbar = fi.colorbar(im)
    cbar.set_label('Correlation coefficient')
    cbar.set_ticks([-1,0,1])
    ax.set_title(titles[n])
    
