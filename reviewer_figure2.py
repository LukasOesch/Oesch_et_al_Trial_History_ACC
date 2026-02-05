#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  2 16:59:08 2025

@author: loesch
"""


import numpy as np
import pandas as pd
import glob
from os.path import splitext, join
import os
import itertools
from sklearn.model_selection import KFold, StratifiedKFold
from scipy.ndimage import gaussian_filter1d
import chiCa
from chiCa import *
from scipy.stats import zscore
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Arial"
import multiprocessing as mp
import sys
sys.path.append('/Users/loesch/Documents/Churchland_lab/chiCa') #Make sure to add this so that multiprocessing finds chiCa
from time import time
from labdatatools import *

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

#------------------------------------------------------------------------------
#%%----- Re-run the decoding analysis excluding trials where data could get
# repeated due to a very short ITI duration. These are sessions where the
# punishment only lasted for 1s or where the mice collected the reward very fast. 


#----IMPORTANT NOTE: Do not run this as full section but rather as a selection--
# Running it as a section or entire file will cause multiprocessing to crash
# beacuse it is not protected by a if name == "__main__" statement!


use_name = 'Reviewer_Fig2_decoding_no_overlap_during_outcome_spr.npy'
signal_type = 'S'
balancing_mode = 'proportional'
k_folds = 8 # Match the number of folds from the original analysis
subsampling_rounds = 20 #Re-drawing of samples from majority class 20 times
cover_all_samples = True #Ensure that all trials are used for model fitting when subsampling from majority class
penalty = 'l2'
normalize = True
reg = 10**(np.linspace(-10,1,12))
model_params = {'penalty': penalty, 'solver': 'liblinear', 'inverse_regularization_strength': reg, 'fit_intercept': True, 'normalize': normalize} #Po

successfully_fitted = []
for session_dir in sessions:
    try:
        trial_alignment_file = glob.glob(session_dir + '/analysis/*miniscope_data.npy')[0]
        miniscope_data = np.load(trial_alignment_file, allow_pickle = True).tolist()  
        frame_rate = miniscope_data['frame_rate']
        
        # Define the alignment for the different task phases
        aligned_to = ['outcome_presentation', 'outcome_end', 'PlayStimulus', 'DemonWaitForResponse']
        time_frame = [ np.array([round(0*frame_rate), round(1*frame_rate)+1], dtype=int),
                      np.array([round(-1*frame_rate), round(0.3*frame_rate)+1], dtype=int),
                      np.array([round(-0.5*frame_rate), round(1*frame_rate)+1], dtype=int),
                      np.array([round(-0.2*frame_rate), round(0.3*frame_rate)+1], dtype=int)]
        
        trialdata = pd.read_hdf(glob.glob(session_dir + '/chipmunk/*.h5')[0], '/Data')
        c_data = get_chipmunk_behavior(session_dir)
        prior_choice = np.array(c_data['prior_choice'])
        prior_outcome = np.array(c_data['prior_outcome'])
        
        labels = np.zeros([trialdata.shape[0]])
        labels[(prior_choice==0) & (prior_outcome==1)] = 0
        labels[(prior_choice==0) & (prior_outcome==0)] = 1
        labels[(prior_choice==1) & (prior_outcome==0)] = 2
        labels[(prior_choice==1) & (prior_outcome==1)] = 3
        secondary_labels = None
        
        #Get the outcome duration
        outcome_dur = np.squeeze((trialdata['outcome_end'] - trialdata['outcome_presentation']).tolist())[:,0]
        
        #secondary_labels = np.array(c_data['choice'])
        valid_trials = np.where(c_data['valid_past'])[0] #Retrieve valid trials with a response before and after 
        #==========One session from LO067 has two unidentified response port out events
        if (os.path.split(session_dir)[1]=='20240209_105931') & (os.path.split(os.path.split(session_dir)[0])[1] == 'LO067'):
            valid_trials = np.delete(valid_trials, [113,115])
        #====================================================
        
        #Now exclude all short outcome time trials
        valid_trials = np.array([x for x in valid_trials if outcome_dur[x-1] > 2])
        
        valid_trials_before = valid_trials-1
        val_trials =  [valid_trials_before]*2 + [valid_trials]*2 #Combine trial identities for the ITI before the new trial and once it has started
        
        assert np.unique(labels[valid_trials]).shape[0] == 4, "Could not retain all 4 classes"
        
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
        successfully_fitted.append()
    except Exception as err:
        print(err)

#%%----Check which models were fitted successfully and load them

successfully_fitted = [x for x in sessions if os.path.isfile(os.path.join(x, 'analysis', use_name))] #Two sessions fail to fit with these new criteria

# Load models and retrieve data
classes = 4
model_accuracy = []
shuffle_accuracy = []
label_wise_accuracy_list = []
cross_prediction = []
shuffled_cross_prediction = []
class_wise_cross_prediction = []
confusion_matrices = []
coefficients = []
for session_dir in successfully_fitted:
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

#Get the subject and session ID for the fitted sessions
subj = np.unique([os.path.split(os.path.split(session_dir)[0])[1] for session_dir in successfully_fitted])
subj_code = np.zeros([len(successfully_fitted)],dtype=int)
subj_string = []
session_string = []
for n in range(len(successfully_fitted)):
    for k in range(subj.shape[0]):
        if subj[k] in successfully_fitted[n]:
            subj_code[n] = k
            subj_string.append(str(subj[k]))
    session_string.append(os.path.split(successfully_fitted[n])[1])
    
#%%----Start the plotting for the decoding analyses here

# Split the 
insertion_index = np.zeros(len(time_frame)+1)
insertion_index[1] = time_frame[0][1] - time_frame[0][0]
idx_list = [np.arange(insertion_index[1],dtype=int)]
for k in range(1,len(time_frame)):
    insertion_index[k+1] = int(insertion_index[k] + time_frame[k][1] - time_frame[k][0])
    idx_list.append(np.arange(insertion_index[k],insertion_index[k+1], dtype=int))

dec_acc_subj = np.zeros([133, np.unique(subj_code).shape[0]]) * np.nan
shu_acc_subj = np.zeros([133, np.unique(subj_code).shape[0]]) * np.nan
for k in subj_code:
    dec_acc_subj[:, k] = np.mean(np.vstack(model_accuracy)[subj_code==k,:], axis=0)
    shu_acc_subj[:, k] = np.mean(np.vstack(shuffle_accuracy)[subj_code==k,:], axis=0)

grey = '#858585' 

# Optional: Plot overall average over time
# fi = plt.figure()
# ax = fi.add_subplot(111)
# cols = ['k', grey]
# labels = ['Model', 'Shuffle']
# plot_timecourse(ax, [dec_acc_subj, shu_acc_subj], frame_rate ,idx_list, colors = cols, line_labels=labels)
# ax.set_ylim([0,1])
# separate_axes(ax)

#-----Reviewer Figure 2a - overall decoding accuracy ---------------------------
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
    ax.scatter([0,1] + jitter[k], [np.mean(dec_acc_subj,axis=0)[k], np.mean(shu_acc_subj,axis=0)[k]], c=grey,s=14)
ax.set_ylim([0,1])
separate_axes(ax)
ax.set_ylabel('Decoding accuracy')
ax.set_xticklabels(labels, rotation=45, ha ='right')

#Construct the data frame to export for the lme --> this is the overall accuracy vs shuffle
d_dict = dict()
d_dict['accuracy'] = np.hstack(model_accuracy).tolist() +  np.hstack(shuffle_accuracy).tolist()
d_dict['condition'] = ['model'] * np.size(np.hstack(model_accuracy)) + ['shuffle'] * np.size(np.hstack(model_accuracy))
d_dict['subject'] = np.repeat(subj_string, model_accuracy[0].shape[0]).tolist() * 2
d_dict['time'] = np.array(np.arange(model_accuracy[0].shape[0])).tolist() * (len(model_accuracy * 2))
d_dict['session'] = np.repeat(session_string, model_accuracy[0].shape[0]).tolist() * 2
df = pd.DataFrame(d_dict)
output_loc = '/Users/loesch/Documents/Churchland_lab/Trial-history_manuscript/Figures/reviewer_figure2'
df.to_csv(os.path.join(output_loc, 'Reviewer_Figure2a_decoding_accuracy_decoder_vs_shuffle_no_overlap.csv'))


#----------Reviewer Figure 2b - calss-wise decoding accuracy over time --------------
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
ax.set_ylabel('Decoing accuracy')
ax.set_xlabel('Time (s)')

#Construct the data frame to export the data. Here, test for differences between the different contexts
d_dict = dict()
d_dict['accuracy'] = np.reshape(tmp_acc.transpose(0,2,1), np.size(tmp_acc)).tolist()
d_dict['time'] = np.arange(tmp_acc.shape[1]).tolist() * (tmp_acc.shape[0] * tmp_acc.shape[2])
tmp = []
for k in ['Previous correct left', 'Previous incorrect left', 'Previous incorrect right', 'Previous correct right']:
    tmp = tmp + [k] * tmp_acc.shape[1]
d_dict['history_context'] = tmp * tmp_acc.shape[0]
tmp = []
for ses in successfully_fitted:
    tmp = tmp +  [os.path.split(os.path.split(ses)[0])[1]] * (tmp_acc.shape[1] * tmp_acc.shape[2])
d_dict['subject'] = tmp
d_dict['session'] = np.repeat(session_string, np.size(label_wise_acc_subj[:,0,:]))
df = pd.DataFrame(d_dict)
df.to_csv(os.path.join(output_loc, 'Reviewer_figure2b_class_wise_decoding_accuracy_no_overlap.csv'))


#---------Reviewer Figure 2c - confusion matrices for different time points ----
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

newcmp = white_to_color((10, 10, 77)) # White to purple

confusion_timepoints = [0,15,75]
con_mats = np.zeros([classes,classes,len(confusion_timepoints)]) * np.nan
for k in range(len(confusion_timepoints)):
    all_con_mats = np.squeeze([x[confusion_timepoints[k],:,:,confusion_timepoints[k]] for x in confusion_matrices])
    tmp = [np.mean(all_con_mats[subj_code==x,:,:],axis=0) for x in np.unique(subj_code)] # Average over subjects first
    con_mats[:,:,k] = np.mean(np.squeeze(tmp),axis=0)
    
titles = ['t = 0 s','t = 0.5 s', 't > 2 s']
labels = ['Correct left', 'Incorrect left', 'Incorrect right', 'Correct right'] # Previous trial
fi = plt.figure()
ax = [fi.add_subplot(len(confusion_timepoints),1,k+1, aspect='equal') for k in range(con_mats.shape[2])]
for k in range(con_mats.shape[2]):
    im = ax[k].imshow(con_mats[:,:,k], cmap = newcmp, vmin=0, vmax=1)
    ax[k].set_title(titles[k])
ax[0].tick_params(labelbottom=False, bottom=False)
ax[1].tick_params(labelbottom=False, bottom=False)
ax[2].set_xticks(np.arange(4))
ax[2].set_xticklabels(labels, rotation=45, ha ='right')
cbar = fi.colorbar(im, ax=ax)
cbar.set_label('Fraction of true labels')
cbar.set_ticks([0,0.25,0.5,0.75,1])


#------------Reviewer Figure 2d - plot cross-predictino accuracy --------------
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
im = ax.imshow(cross_pred_acc.T, vmin=0, vmax=1, cmap = col_map, interpolation='none')
for k in line_idx:
    ax.axvline(k, color = 'k', linewidth=0.5, linestyle='--')
    ax.axhline(k, color = 'k', linewidth=0.5, linestyle='--')
ax.set_xticks([])
ax.set_yticks([])
ax.set_ylabel('Training time points')
ax.set_xlabel('Testing time points')
cbar = fi.colorbar(im)
cbar.set_label('Decoding accuracy')

    
#----------Reviewer Figure 2e - Compare cross-decoding accuracy between different phases -------
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

#Construct the data frame
cross_acc_full = np.zeros([len(successfully_fitted), len(idx_list), len(idx_list)])
shu_acc_full = np.zeros([len(successfully_fitted), len(idx_list), len(idx_list)])
for k in range(len(successfully_fitted)):
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
df.to_csv(os.path.join(output_loc, 'Reviewer_Figure2e_phase_wise_cross_decoding_no_overlap.csv'))          
            
#%%------Fit encoding models to the sessions and data where short ITI duration
# trials are excluded.
#
# IMPORTANT NOTE:
# This version of the encoding model fitting script is a modification of an earlier
# version of "encoding_model_fitting.py" with minimal comment editing and
# documentation. Please refer to "encoding_model_fitting.py" for a more thorough
# documentation on the individual steps in the encoding model fitting process.

for session_dir in successfully_fitted:
    #------Some parameter definitions--------------------
    print(f'Starting: {session_dir}')
    file_name = 'RidgeCV_encoding_models_ITI_partitioned_no_overlap_reviwer_figure2'
    
    signal_type = 'F'
    
    k_folds = 10 #For regular random cross-validation
    which_models = 'single_var' #'cognitive' #'group' # 'individual', 'timepoint'
    #Determines what type of models should be fitted. 'group' will lump specified regressors
    #into a group and fit models for the cvR2 and dR2 for each of the groups, 'individual'
    #will assess the explained variance for each regressor alone and 'timepoint' will
    #look at the task regressors collectively but evaluate the model performance at each
    #trial time separately
    
    add_complete_shuffles = 0 #Allows one to add models where all the regressors
    #are shuffled idependently. This can be used to generate a null distribution for
    #the beta weights of certain regressors
    
    use_parallel = False #Whether to do parallel processing on the different shuffles
    
    #------Loading the data--------------------
    trial_alignment_file = glob.glob(session_dir + '/analysis/*miniscope_data.npy')[0]
    miniscope_data = np.load(trial_alignment_file, allow_pickle = True).tolist()  
    frame_rate = miniscope_data['frame_rate']
    trialdata = pd.read_hdf(glob.glob(session_dir + '/chipmunk/*.h5')[0], '/Data')
    
    #Get video alignment
    video_alignment_files = glob.glob(session_dir + '/analysis/*video_alignment.npy')
    if len(video_alignment_files) > 1:
            print('More than one video is currently not supported')
    video_alignment = np.load(video_alignment_files[0], allow_pickle = True).tolist()
    
    #Retrieve dlc tracking
    dlc_file = glob.glob(session_dir + '/dlc_analysis/*.h5')[-1] #Load the latest extraction!
    dlc_data = pd.read_hdf(dlc_file)
    dlc_metadata = pd.read_pickle(splitext(dlc_file)[0] + '_meta.pickle')
    
    #Load the video components and the video motion energy components
    video_svd = np.load(glob.glob(session_dir + '/analysis/*video_SVD.npy')[0], allow_pickle = True).tolist()[1].T #Only load temporal components
    me_svd = np.load(glob.glob(session_dir + '/analysis/*motion_energy_SVD.npy')[0], allow_pickle = True).tolist()[1].T #Only load temporal components 
    
    #--------Alignments and construction of the design matrix-----------------
    #Define the shift for the event-based regressors
    poke_min_shift = -0.5 * frame_rate
    poke_max_shift = 1 * frame_rate
    
    #Also do this for the stimulus events
    stim_min_shift = 0
    stim_max_shift = 0.5 * frame_rate
    
    #Set the times up
    aligned_to = ['outcome_presentation', 'outcome_end', 'PlayStimulus', 'DemonWaitForResponse']
    time_frame = [np.array([round(0*frame_rate), round(1*frame_rate)+1], dtype=int),
                  np.array([round(-1*frame_rate), round(0.3*frame_rate)+1], dtype=int),
                  np.array([round(-0.5*frame_rate), round(1*frame_rate)+1], dtype=int),
                  np.array([round(-0.2*frame_rate), round(0.3*frame_rate)+1], dtype=int)]
    
    #Assemble the task variable design matrix
    choice = np.array(trialdata['response_side'])
    category = np.array(trialdata['correct_side'])
    prior_choice =  chiCa.determine_prior_variable(np.array(trialdata['response_side']), np.ones(len(trialdata)), 1, 'consecutive')
    prior_category =  chiCa.determine_prior_variable(np.array(trialdata['correct_side']), np.ones(len(trialdata)), 1, 'consecutive')
    
    outcome = np.array(trialdata['response_side'] == trialdata['correct_side'], dtype=float) #Define as float so that nans (that are float) can be retained
    outcome[np.array(np.isnan(trialdata['response_side']))] = np.nan
    prior_outcome =  chiCa.determine_prior_variable(outcome, np.ones(len(trialdata)), 1, 'consecutive')
       
    modality = np.zeros([trialdata.shape[0]])
    modality[trialdata['stimulus_modality'] == 'auditory'] = 1
    prior_modality = chiCa.determine_prior_variable(modality, np.ones(len(trialdata)), 1, 'consecutive')
    
    #Find stimulus strengths
    tmp_stim_strengths = np.zeros([trialdata.shape[0]], dtype=int) #Filter to find easiest stim strengths
    for k in range(trialdata.shape[0]):
        tmp_stim_strengths[k] = trialdata['stimulus_event_timestamps'][k].shape[0]
    
    #Get category boundary to normalize the stim strengths
    unique_freq = np.unique(tmp_stim_strengths)
    category_boundary = (np.min(unique_freq) + np.max(unique_freq))/2
    stim_strengths = (tmp_stim_strengths- category_boundary) / (np.max(unique_freq) - category_boundary)
    if trialdata['stimulus_modality'][0] == 'auditory':
        stim_strengths = stim_strengths * -1
        
    difficulty = 1 - np.abs(stim_strengths)
    prior_difficulty =  chiCa.determine_prior_variable(difficulty, np.ones(len(trialdata)), 1, 'consecutive')
        
    #Compute the bins
    bin_number = 60 #6°
    bins = np.arange(-np.pi, np.pi+np.pi/(bin_number/2), np.pi/(bin_number/2))[:bin_number+1] #Divide bin number by 2 because 360 deg is 2 pi
    angles = ['pitch', 'roll', 'yaw']
    
    for a in angles:
        bin_id = np.digitize(miniscope_data[a], bins, right = False)
        tmp = np.zeros([miniscope_data[a].shape[0],bins.shape[0]-1]) #The last bin is the upper boundary, pi
        for k in range(bin_id.shape[0]):
            tmp[k,bin_id[k]-1] = 1 #Weird behavior so that the first bin is at 1!
        exec(f'{a} = tmp')
    
    #################################################################
    #
    
    #Extract a set of dlc labels and standardize these.
    dlc_keys = dlc_data.keys().tolist()
    specifier = dlc_keys[0][0] #Retrieve the session specifier that is needed as a keyword
    body_part_name = dlc_metadata['data']['DLC-model-config file']['all_joints_names']
    
    temp_body_parts = []
    part_likelihood_estimate = []
    for bp in body_part_name:
        for axis in ['x', 'y']:
            temp_body_parts.append(np.array(dlc_data[(specifier, bp, axis)]))
        part_likelihood_estimate.append(np.array(dlc_data[(specifier, bp, 'likelihood')]))
    
    body_parts = np.array(temp_body_parts).T #To array and transpose
    part_likelihood_estimate = np.array(part_likelihood_estimate).T
    
    #Use the chest (body part 8 and 9)
    part_id = 8
    video_dimensions = np.array( dlc_metadata['data']['frame_dimensions']) #first value is y second is x
    pixels_per_bin = 32 #At 25 px/cm, so that each place field is 1.28 cm**2
    
    x_bins = np.arange(0, video_dimensions[1] + pixels_per_bin, pixels_per_bin) #Divide video width in bins
    y_bins = np.arange(0, video_dimensions[0] + pixels_per_bin, pixels_per_bin)
    combinations = np.array(list(itertools.product(np.arange(x_bins.shape[0]-1), np.arange(y_bins.shape[0]-1)))) #Get all possible combinations that may occur
    
    what_bin = np.vstack((np.digitize(body_parts[:,part_id], x_bins, right = False), np.digitize(body_parts[:,part_id+1], y_bins, right = False))).T
    
    position_vect = np.zeros([body_parts.shape[0],combinations.shape[0]]) #A linearized version of bin occupancy at each time point
    for k in range(combinations.shape[0]):
        idx = np.where((what_bin == combinations[k,:]).all(1))[0]
        if idx.shape[0] > 0:
            position_vect[idx, k] = 1
            
    #---------Extract instructed movement regressors and stimulus
    
    #Align the different poke events    
    actions = ['Port2In','Port1In', 'Port3In']
    response_matrix = []
    for k in range(len(actions)):
        event_start_frame, trial_id, state_time_covered = chiCa.align_miniscope_to_event(trialdata[actions[k]].tolist(),
                                                                                         np.array(trialdata['FinishTrial'].tolist())[:,0],
                                                                                         miniscope_data['frame_interval'],
                                                                                         miniscope_data['trial_starts'])
        event_trace = chiCa.assemble_event_trace(event_start_frame, miniscope_data['F'].shape[1])
        response_matrix.append(chiCa.shift_regressor(event_trace, int(poke_min_shift), int(poke_max_shift)))
    response_matrix = np.hstack(response_matrix)
    
    #Now align the stimulus regressor
    t_stamps = chiCa.get_experienced_stimulus_events(trialdata, stim_modalities = ['visual', 'auditory', 'audio-visual'])
    event_start_frame, trial_id, state_time_covered = chiCa.align_miniscope_to_event(t_stamps,
                                                                                         np.array(trialdata['FinishTrial'].tolist())[:,0],
                                                                                         miniscope_data['frame_interval'],
                                                                                         miniscope_data['trial_starts'])
    event_trace = chiCa.assemble_event_trace(event_start_frame, miniscope_data['F'].shape[1])
    stim_matrix = chiCa.shift_regressor(event_trace, int(stim_min_shift), int(stim_max_shift))
   
    #Find the valid trials to be included the criteria are the following:
    valid_trials = np.where((np.isnan(choice) == 0) & (np.isnan(prior_choice) == 0))[0]
    
    #-----The poke out was not detected in one session from LO067 on two trials, these have to be removed--
    #============================================================
    if (os.path.split(session_dir)[1] == '20240209_105931') & (os.path.split(os.path.split(session_dir)[0])[1] == 'LO067'):
        valid_trials = np.delete(valid_trials,[113,115])
    #===========================================================
    #Get the outcome duration
    outcome_dur = np.squeeze((trialdata['outcome_end'] - trialdata['outcome_presentation']).tolist())[:,0]
    
    #Now exclude all short outcome time trials
    valid_trials = np.array([x for x in valid_trials if outcome_dur[x-1] > 2])
    
    valid_trials_before = valid_trials-1
    val_trials =  [valid_trials_before]*2 + [valid_trials]*2
      
    #Now start settting up the regressors
    individual_reg_idx = [] #Keep indices of individual regressors
    reg_group_idx = [] #Keep indices of regressor groups
    
    total_frames = []
    for k in time_frame:
        total_frames.append(k[1] - k[0])
    total_frames = np.sum(total_frames)
    
    block = np.zeros([total_frames, total_frames])
    for k in range(block.shape[0]):
        block[k,k] = 1
      
    
    #Stack the blocks of cognitive regressors and multiply them by the respective value
    time_reg = np.array(block)
    choice_x = block * choice[valid_trials[0]]
    outcome_x = block * outcome[valid_trials[0]]
    #prior_difficulty_x = block  * prior_difficulty[valid_trials[0]]
    
    #Include all required interactions too, the one captured in the default is left choice undrewarded
    prior_incorrect_left_x = block * ((prior_choice[valid_trials[0]] == 0) & (prior_outcome[valid_trials[0]]==0))
    prior_correct_left_x = block * ((prior_choice[valid_trials[0]] == 0) & (prior_outcome[valid_trials[0]]==1))
    prior_incorrect_right_x = block * ((prior_choice[valid_trials[0]] == 1) & (prior_outcome[valid_trials[0]]==0))
    prior_correct_right_x = block * ((prior_choice[valid_trials[0]] == 1) & (prior_outcome[valid_trials[0]]==1))
    
    
    for k in range(1, valid_trials.shape[0]):
          time_reg = np.vstack((time_reg, block))
          choice_x = np.vstack((choice_x, block * choice[valid_trials[k]]))
          #prior_difficulty_x = np.vstack((prior_difficulty_x, block * prior_difficulty[valid_trials[k]]))
          outcome_x = np.vstack((outcome_x, block * outcome[valid_trials[k]]))
          prior_incorrect_left_x = np.vstack((prior_incorrect_left_x, block * ((prior_choice[valid_trials[k]] == 0) & (prior_outcome[valid_trials[k]]==0))))
          prior_correct_left_x = np.vstack((prior_correct_left_x, block * ((prior_choice[valid_trials[k]] == 0) & (prior_outcome[valid_trials[k]]==1))))
          prior_incorrect_right_x = np.vstack((prior_incorrect_right_x, block * ((prior_choice[valid_trials[k]] == 1) & (prior_outcome[valid_trials[k]]==0))))
          prior_correct_right_x = np.vstack((prior_correct_right_x, block * ((prior_choice[valid_trials[k]] == 1) & (prior_outcome[valid_trials[k]]==1))))

    #x_include = np.hstack((time_reg, choice_x, outcome_x, prior_difficulty_x, prior_incorrect_left_x, prior_correct_left_x, prior_incorrect_right_x, prior_correct_right_x)) 
    #Dropped previous difficulty for the paper
    x_include = np.hstack((time_reg, choice_x, outcome_x, prior_incorrect_left_x, prior_correct_left_x, prior_incorrect_right_x, prior_correct_right_x)) 
    
    #
    #Get the neural signals
    
    if signal_type == 'S':
        signal = gaussian_filter1d(miniscope_data[signal_type].T,1, axis=0, mode='constant', cval=0, truncate=4)
        #Pad the first and last samples with zeros in this condition
    else:
        signal = miniscope_data[signal_type].T #Transpose the original signal so that rows are timepoints and columns cells
    
    #Determine which neurons to include in the analysis
    keep_neuron = np.arange(signal.shape[1])
    
    #Align signal to the respsective state and retrieve the data
    Y = []
    x_analog = []
    part_likelihood = []
    #Also store the timestamps that are included into the trialized design matrix
    trial_timestamps_imaging = []
    trial_timestamps_video = []
    for k in range(len(aligned_to)):
        state_start_frame, state_time_covered = chiCa.find_state_start_frame_imaging(aligned_to[k], trialdata, miniscope_data['frame_interval'], miniscope_data['trial_starts'], miniscope_data['trial_start_time_covered'])                                                                     
        zero_frame = np.array(state_start_frame[val_trials[k]] + time_frame[k][0], dtype=int) #The firts frame to consider
        
        for add_to in np.arange(time_frame[k][1] - time_frame[k][0]):
            matching_frames = []
            for q in range(zero_frame.shape[0]): #unfortunately need to loop through the trials, should be improved in the future...
                    tmp = chiCa.match_video_to_imaging(np.array([zero_frame[q] + add_to]), miniscope_data['trial_starts'][val_trials[k][q]],
                           miniscope_data['frame_interval'], video_alignment['trial_starts'][val_trials[k][q]], video_alignment['frame_interval'])[0].astype(int)
                    matching_frames.append(tmp)
            
            Y.append(signal[zero_frame + add_to,:][:, keep_neuron])
           
            x_analog.append(np.concatenate((stim_matrix[zero_frame + add_to,:], response_matrix[zero_frame + add_to,:], pitch[zero_frame + add_to,:], roll[zero_frame + add_to,:], yaw[zero_frame + add_to,:], position_vect[matching_frames,:], video_svd[matching_frames,:], me_svd[matching_frames,:]), axis=1))
            
            part_likelihood.append(part_likelihood_estimate[matching_frames,:])
            trial_timestamps_imaging.append(zero_frame + add_to)
            trial_timestamps_video.append(matching_frames)
            
    #Back to array like, where columns are trials, rows time points and sheets cells   
    Y = np.squeeze(Y)
    x_analog = np.squeeze(x_analog)
    
    #Reshape the arrays to match the design matrix
    Y = np.reshape(Y, (x_include.shape[0], Y.shape[2]), order = 'F')
    x_analog = np.reshape(x_analog, (x_include.shape[0], x_analog.shape[2]), order = 'F')
    
    part_likelihood = np.squeeze(part_likelihood)
    
    #Transform to timepoint x (valid) trial matrix
    trial_timestamps_imaging = np.squeeze(trial_timestamps_imaging)
    trial_timestamps_video = np.squeeze(trial_timestamps_video)
    
    #Add the analog regressors to the cognitive regressor design matrix
    X = np.hstack((x_include, x_analog))  
    
    #
    #Track where the regressors and regressor groups live inside the design matrix
    regressor_labels = ['trial_time', 'choice',  'outcome', 'previous_choice_outcome_combination', 'stim_events', 'center_poke', 'left_poke', 'right_poke', 'pitch', 'roll', 'yaw', 'chest_point', 'video_svd', 'video_me_svd']
    regressor_idx = []
    loop_range = [x for x in range(len(regressor_labels)) if regressor_labels[x] == 'previous_choice_outcome_combination'][0]
    for k in range(loop_range): #time, choice, stim strength, outcome
       regressor_idx.append(np.arange(k*block.shape[0], (k+1)*block.shape[0]))
    regressor_idx.append(np.arange(regressor_idx[-1][-1]+1, regressor_idx[-1][-1]+1 + 4*block.shape[0]))
    regressor_idx.append(np.arange(regressor_idx[-1][-1]+1, regressor_idx[-1][-1]+1 + stim_matrix.shape[1]))
    for k in range(3): #All different poke events are shifted with the same lags
        regressor_idx.append(np.arange(regressor_idx[-1][-1]+1, regressor_idx[-1][-1]+1 + int(response_matrix.shape[1]/3)))
    for k in range(3):
        regressor_idx.append(np.arange(regressor_idx[-1][-1]+1, regressor_idx[-1][-1]+1 + pitch.shape[1]))
    regressor_idx.append(np.arange(regressor_idx[-1][-1]+1, regressor_idx[-1][-1]+1 + position_vect.shape[1]))
    regressor_idx.append(np.arange(regressor_idx[-1][-1]+1, regressor_idx[-1][-1]+1 + video_svd.shape[1]))      
    regressor_idx.append(np.arange(regressor_idx[-1][-1]+1, regressor_idx[-1][-1]+1 + me_svd.shape[1]))
                         
    individual_regressors = dict()
    for k in range(len(regressor_labels)):
        individual_regressors[regressor_labels[k]] = regressor_idx[k]
        #Unfortunately also ordered alphabetically
    
    #Update info about the regressor indices ,always remember uppper exclusive
    reg_group_idx = [np.arange(regressor_idx[1][0], regressor_idx[7][0])] #The task regressors (chocie, prior, etc. + stimulus events) exclusive the intercept
    reg_group_idx.append(np.arange(regressor_idx[7][0],regressor_idx[10][0])) #Instructed actions (center and side poking)
    reg_group_idx.append(np.arange(regressor_idx[10][0],regressor_idx[-1][-1])) #Uninstructed movements (head orientation tuning, chest position, video)
    
    #Dictionary of regressor idx for different variales
    variable_idx = dict()
    for k in range(len(regressor_labels)):
        variable_idx[regressor_labels[k]] = regressor_idx[k] 
    
    standardize_reg = np.arange(individual_regressors['video_svd'][0], individual_regressors['video_me_svd'][-1]+1) #Only z-score the truly anlog video variables
    #Video_svd is the first truly analog regressor, and video_me_svd is the last one of that kind.
    
    Q_vid = np.linalg.qr(np.hstack((X[:,individual_regressors['stim_events']], X[:,individual_regressors['video_svd']])))[0][:,individual_regressors['stim_events'].shape[0]:]
    Q_me =np.linalg.qr(np.hstack((X[:,individual_regressors['stim_events']], X[:,individual_regressors['video_me_svd']])))[0][:,individual_regressors['stim_events'].shape[0]:]
        
    X[:, individual_regressors['video_svd']] = Q_vid #Reassemble the design matrix
    X[:, individual_regressors['video_me_svd']] = Q_me
    
   #-------Draw the training and testing splits-------------------
    
    #First get the splits for training and testing sets. These will be constant throughout
    kf = StratifiedKFold(n_splits = k_folds, shuffle = True)
    timepoint_labels = np.tile(np.arange(block.shape[0]),valid_trials.shape[0])
    k_fold_generator = kf.split(X, timepoint_labels) #This returns a generator object that spits out a different split at each call
    training = []
    testing = []
    for draw_num in range(k_folds):
        tr, te = k_fold_generator.__next__()
        training.append(tr)
        testing.append(te)
     #------------Determine which regressor to shuffle
    shuffle_regressor = [None] #The full model with no shuffling
    
    #Compute the null model which here corresponds to the time regressor
    shuffle_regressor.append(np.arange(block.shape[0], X.shape[1]))
    
    if which_models == 'group': #Shuffle all the regressors belonging to one regressor group
        for k in reg_group_idx:
            tmp = np.arange(block.shape[0], X.shape[1]).tolist()
            tmp = list(set(tmp) - set(k))
            shuffle_regressor.append(tmp) #The single variable group models
        
        for k in reg_group_idx:
            shuffle_regressor.append(k.tolist()) #The eliminate-one group models
    
    elif which_models == 'single_var': #Get all the tracked variables
        for k in range(1, len(regressor_idx)):
            tmp = np.arange(block.shape[0], X.shape[1]).tolist()
            tmp = list(set(tmp) - set(regressor_idx[k]))
            shuffle_regressor.append(tmp) #The single variable group models
        
        for k in range(1, len(regressor_idx)):
            shuffle_regressor.append(regressor_idx[k].tolist()) #The eliminate-one group models
    
    elif which_models == 'individual': #One regressor or a pair of coordinates are shuffled -> unique explained variance
        # shuffle_individual = np.arange(block.shape[1], x_include.shape[1] + head_orientation.shape[1]).tolist() #Exclude the time regressors from shuffling for now
        # shuffle_pairs = np.arange(shuffle_individual[-1] + 1, shuffle_individual[-1] + 1 + body_parts.shape[1], 2).tolist()
        for k in single_regressors:
            tmp = np.arange(block.shape[0], X.shape[1]).tolist()
            tmp.remove(k)
            shuffle_regressor.append(tmp) #The single variable models
        for k in paired_regressors:
            tmp = np.arange(block.shape[0], X.shape[1]).tolist()
            tmp.remove(k)
            tmp.remove(k+1)
            shuffle_regressor.append(tmp)
        
        for k in single_regressors:
            shuffle_regressor.append([k]) #The eliminate-one models
        for k in paired_regressors:
            shuffle_regressor.append([k, k+1])
            
    elif which_models == 'timepoint': #This mode looks at how much the explained
    #variance for the different task regressors fluctuates over the course of the trial.
    #It takes advantage of fitting all the continuous regressors for the entire trial but
    #evaluates only a single timepoint in the trial, thus the baseline model to compare to
    #is the model with all task regressors shuffled.
    # NOTE: This is under construction and only cvR2 is implemented now...
        for k in range(block.shape[0]):
             tmp = np.arange(block.shape[0], x_include.shape[1]).tolist()
             rem_set = k + block.shape[0] * np.arange(1, x_include.shape[1]/block.shape[0]).astype(int)
             tmp = [i for i in tmp if i not in rem_set] #Remove by list content
             shuffle_regressor.append(tmp)
    
    elif which_models == 'cognitive':
        #Shuffles the individual cognitive regressors, don't use parallel 
        #to be able to reconstruct the timecourse
        cog_reg = int(x_include.shape[1] / block.shape[0]) - 1
        for k in range(cog_reg):
            tmp = np.arange(block.shape[0], X.shape[1]).tolist()
            tmp = list(set(tmp) - set(np.arange((k+1)*block.shape[0], (k+2)*block.shape[0])))
            shuffle_regressor.append(tmp) #The single variable group models
        
        for k in range(cog_reg):
            shuffle_regressor.append(np.arange((k+1)*block.shape[0], (k+2)*block.shape[0])) #The eliminate-one group models
    
    
    #Add a set number of complete shuffles, except for the intercept term
    for k in range(add_complete_shuffles):
        shuffle_regressor.append(np.arange(block.shape[0], X.shape[1]).tolist())
    
    #---------Start the model fitting------------------------------
    
    all_betas = []
    all_alphas = []
    all_rsquared = []
    all_corr = []
    all_r_timecourse = []
    all_corr_timecourse = []
    
    #Specific parameters for ridgeCV
    alpha_range = 10**np.linspace(-3,5,9)
    fit_intercept = False
    alpha_per_target = True
   
    for s_round in range(len(shuffle_regressor)):
        success = 0
        while success == 0: #Some shuffles lead to non-converging svd for the OLS solutions that will throw errors occasionally
            try:
                alphas, betas, r_squared, corr, y_test, y_hat = chiCa.fit_ridge_cv_shuffles(X, Y, alpha_range, alpha_per_target, fit_intercept, shuffle_regressor[s_round], standardize_reg, training, testing)    
                success = 1
            except:
                pass
        time_r, time_corr = chiCa.r_squared_timecourse(y_test, y_hat, testing, block.shape[0])
        all_alphas.append(alphas)
        all_betas.append(betas)
        all_rsquared.append(r_squared)
        all_corr.append(corr)
        all_r_timecourse.append(time_r)
        all_corr_timecourse.append(time_corr)
      
    
    #Store results
    out_dict = dict()
    out_dict['betas'] = all_betas
    out_dict['alphas'] = all_alphas
    out_dict['r_squared'] = all_rsquared
    out_dict['squared_correlation'] = all_corr
    out_dict['variable_index'] = variable_idx
    out_dict['regressor_groups'] = reg_group_idx
    out_dict['shuffle_regressor'] = shuffle_regressor
    out_dict['k_fold'] = k_folds
    out_dict['frames_per_trial'] = block.shape[0]
    out_dict['r_squared_timecourse'] = all_r_timecourse
    out_dict['corr_timecourse'] = all_corr_timecourse
    np.save(join(session_dir,'analysis',file_name), out_dict)


#%%--Start the encoding model loading and plotting

# Grab frame rate from the first session
trial_alignment_file = glob(session_dir + '/analysis/*miniscope_data.npy')[0] 
miniscope_data = np.load(trial_alignment_file, allow_pickle = True).tolist()  
frame_rate = miniscope_data['frame_rate']

# Make sure the alignment exists in case the model fitting is not run...
aligned_to = ['outcome_presentation', 'outcome_end', 'PlayStimulus', 'DemonWaitForResponse']
time_frame = [np.array([round(0*frame_rate), round(1*frame_rate)+1], dtype=int),
                  np.array([round(-1*frame_rate), round(0.3*frame_rate)+1], dtype=int),
                  np.array([round(-0.5*frame_rate), round(1*frame_rate)+1], dtype=int),
                  np.array([round(-0.2*frame_rate), round(0.3*frame_rate)+1], dtype=int)]#Get cv rsq for single variable models
insertion_index = np.zeros(len(time_frame)+1)
insertion_index[1] = time_frame[0][1] - time_frame[0][0]
idx_list = [np.arange(insertion_index[1],dtype=int)]
for k in range(1,len(time_frame)):
        insertion_index[k+1] = int(insertion_index[k] + time_frame[k][1] - time_frame[k][0])
        idx_list.append(np.arange(insertion_index[k],insertion_index[k+1], dtype=int))
     
# Load the encoding model results, this is analogous to figure3.py
single_var = []
from_session = []
original_neuron_id = []
encoding_model_name = 'RidgeCV_encoding_models_ITI_partitioned_Fig3.npy'
counter = 0
for session_dir in successfully_fitted:
    enc = np.load(os.path.join(session_dir, 'analysis', encoding_model_name),allow_pickle = True).tolist()
    single_var.append(enc)
    from_session.append(np.zeros([enc['r_squared'][0].shape[0]])  + counter)
    counter = counter+1
    original_neuron_id.append(np.arange(enc['r_squared'][0].shape[0]))
    
original_neuron_id = np.hstack(original_neuron_id)
from_session = np.hstack(from_session)

cvR_single = []
dR_single = []
cvR_timecourse_single = []
dR_timecourse_single = []
cvCorr_timecourse_single = []
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
        c.append(single_var[n]['r_squared'][k] - single_var[n]['r_squared'][1])
        ct.append(single_var[n]['r_squared_timecourse'][k] - single_var[n]['r_squared_timecourse'][1])
        ccort.append(single_var[n]['corr_timecourse'][k] - single_var[n]['corr_timecourse'][1])
        
        d.append(single_var[n]['r_squared'][0] - single_var[n]['r_squared'][k + model_num])
        dt.append(single_var[n]['r_squared_timecourse'][0] - single_var[n]['r_squared_timecourse'][k + model_num])
        dcort.append(single_var[n]['corr_timecourse'][0] - single_var[n]['corr_timecourse'][k + model_num])

    cvR_single.append(np.hstack(c).T)
    dR_single.append(np.hstack(d).T)
    cvR_timecourse_single.append(np.hstack(ct))
    dR_timecourse_single.append(np.hstack(dt))
    cvCorr_timecourse_single.append(np.hstack(ccort))
    dCorr_timecourse_single.append(np.hstack(dcort))

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

# Confusing code to identify both what sessions were obtained from what subject
# (subj_ses) and what the identity of the subject was for a given session
# (subj_code, subj_string). Apologies for this mess!
subj = np.unique([os.path.split(os.path.split(session_dir)[0])[1] for session_dir in successfully_fitted])
subj_code = np.zeros([len(successfully_fitted)],dtype=int)
subj_string = []
session_string = []
subj_ses = [[] for x in range(subj.shape[0])]
for n in range(len(successfully_fitted)):
    for k in range(subj.shape[0]):
        if subj[k] in successfully_fitted[n]:
            subj_code[n] = k
            subj_string.append(str(subj[k]))
            subj_ses[k].append(n)
    session_string.append(os.path.split(successfully_fitted[n])[1])

#Calculate mean and sem mdodel accuracy
ses_av = np.array([np.mean(x['r_squared'][0]) for x in single_var])
subj_av = np.array([np.mean(ses_av[subj_code==x]) for x in np.unique(subj_code)])
subj_sem = np.std(subj_av) / np.sqrt(np.unique(subj_code).shape[0])
print(f'Mean +/- sem full model variance is: {np.mean(subj_av)} +/- {subj_sem}')

mean_cvR = []
mean_dR = []
for k in range(len(subj.tolist())):
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
        
        
#------------Reviewer Figure 2f-------------------------------------------

grey = '#858585' 
ax = plt.figure(figsize=[10, 4.8]).add_subplot()
widths = 0.7
fancy_boxplot(ax, np.squeeze(mean_cvR), None, widths = widths, labels = labels, y_label = 'Maximal explained variance')
ax.axhline(0, linestyle='--', color='k', linewidth=0.5)
jitter = (0.5 - np.random.rand(len(mean_cvR))) * (widths*0.8)
for k in range(len(mean_cvR)):
    ax.scatter(np.arange(1, model_num + 1) + jitter[k], np.squeeze(mean_cvR)[k,:],c=grey,s=10)
separate_axes(ax)
ax.spines['bottom'].set_bounds([0.5, len(cvR_single)+0.5])
ax.set_xticklabels(labels, rotation=45, ha ='right')

#-------------Reviewer Figure 2g-----------------------------------------------

ax = plt.figure(figsize=[10, 4.8]).add_subplot()
widths = 0.7
fancy_boxplot(ax, np.squeeze(mean_dR), None, widths = widths, labels = labels, y_label = 'Unique explained variance')
ax.axhline(0, linestyle='--', color='k', linewidth=0.5)
jitter = (0.5 - np.random.rand(len(mean_cvR))) * (widths*0.8)
for k in range(len(mean_dR)):
    ax.scatter(np.arange(1, model_num + 1) + jitter[k], np.squeeze(mean_dR)[k,:],c=grey,s=10)
separate_axes(ax)
ax.spines['bottom'].set_bounds([0.5, len(cvR_single)+0.5])
ax.set_xticklabels(labels, rotation=45, ha ='right')

#------------Reviewer Figures 2h, i, and j------------------------------------

#Find number of cells per successfully fitted session
cell_num = []
for session_dir in successfully_fitted:
    trial_alignment_file = glob(session_dir + '/analysis/*miniscope_data.npy')[0]
    miniscope_data = np.load(trial_alignment_file, allow_pickle = True).tolist()
    cell_num.append(miniscope_data['F'].shape[0])
unique_subj = np.unique(subj_string)

#Finally for each subject find the session with most recorded neurons
ses_idx = []
for subj in unique_subj:
    tmp = np.where(subj == np.array(subj_string))[0]
    ses_idx.append(tmp[np.argmax(np.array(cell_num)[tmp])])
use_neuron = np.array([x for x in range(from_session.shape[0]) if from_session[x] in ses_idx])

reg_idx = [11, 10, 0, 9] #Indices to the unique variance for the Video SVD, Chest point, Choice, and Yaw tuning
cols = ['#629bc6',  '#98446f', '#bc6238', '#c8a91e']
ylims = [-0.01, np.max(np.squeeze(dR_single)[reg_idx,:][:,use_neuron]+ 0.01)] #The regresssor of interest
xlims = [-0.01, np.max(np.squeeze(dR_single)[2,use_neuron]+ 0.01)] #The trial history
xlims = [-0.01, 0.35]
ax_list = []
for k in range(len(reg_idx)):
    ax_list.append(plt.figure(figsize=[5.5, 4.8]).add_subplot(111))
    ax_list[-1].scatter(dR_single[2][use_neuron], dR_single[reg_idx[k]][use_neuron], c=cols[k], s = 20, edgecolor='w', linewidth=0.5)
    ax_list[-1].set_xlim(xlims)
    ax_list[-1].set_ylim(ylims)
    ax_list[-1].set_xlabel('Unique explained variance\nfor trial history')
    ax_list[-1].set_ylabel(f'Unique explained variance\nfor {labels[reg_idx[k]]}')
    ax_list[-1].set_title(f'{labels[reg_idx[k]]}')
    separate_axes(ax_list[-1])
   