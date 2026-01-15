#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 30 09:58:35 2025

@author: loesch
"""

from chiCa import *
import chiCa
import matplotlib.pyplot as plt
import pandas as pd
from labdatatools import *
import os
from glob import glob
# from scipy.ndimage import gaussian_filter1d
# from scipy.stats import pearsonr, zscore

from scipy.ndimage import zoom


#%%---Load the data from the example session and compute cvR and dR

subject_data = dict()
base_dir = get_labdata_preferences()['paths'][0] #Retrieve the base directory
## Alternatively set the base directory manualy:
#base_dir = 'my base directory'

session_dir = os.path.join(base_dir, 'LO069', '20240102_125134') #Construct the session_dir
encoding_model_name = 'RidgeCV_encoding_models_ITI_partitioned_Fig3.npy'
enc = np.load(os.path.join(session_dir, 'analysis', encoding_model_name),allow_pickle = True).tolist()

dR = []
cvR = []
betas_single = []
model_num = int((len(enc['r_squared']) - 2) / 2) #The first two mdodels are the full and the time regressor only models
for k in range(2, model_num + 2):  
    cvR.append(enc['r_squared'][k] - enc['r_squared'][1])
    dR.append(enc['r_squared'][0] - enc['r_squared'][k + model_num])
    betas_single.append(enc['betas'][k])
full_R = enc['r_squared']
full_betas = enc['betas'][0]

# Generate the alignment timestamps and the list of frame indices for each trial phase
frame_rate = int(np.load(glob(os.path.join(session_dir, 'analysis', '*miniscope_data.npy'))[0],allow_pickle = True).tolist()['frame_rate'])

aligned_to = ['outcome_presentation', 'outcome_end', 'PlayStimulus', 'DemonWaitForResponse']
time_frame = [ np.array([round(0*frame_rate), round(1*frame_rate)+1], dtype=int),
              np.array([round(-1*frame_rate), round(0.3*frame_rate)+1], dtype=int),
              np.array([round(-0.5*frame_rate), round(1*frame_rate)+1], dtype=int),
              np.array([round(-0.2*frame_rate), round(0.3*frame_rate)+1], dtype=int)]
insertion_index = np.zeros(len(time_frame)+1)
insertion_index[1] = time_frame[0][1] - time_frame[0][0]
idx_list = [np.arange(insertion_index[1],dtype=int)]
for k in range(1,len(time_frame)):
    insertion_index[k+1] = int(insertion_index[k] + time_frame[k][1] - time_frame[k][0])
    idx_list.append(np.arange(insertion_index[k],insertion_index[k+1], dtype=int))

#%%---- generate a bettery of plots with the encdoing model weights comapring the full model to the single var models

#neuron_id = 296
#neuron_id = 78
#neuron_id = 107 # For chest
#neuron_id = 275
#neuron_id = 107
save_location = '/Users/loesch/Documents/Churchland_lab/Trial-history_manuscript/Supp_fig3_encdoing/'

n_id = [296, 275]

cols = ['#8a2525', '#393838'] #Dark red for individual models, dark grey for full

for neuron_id in n_id:
    #-----Start plotting the trial history regressors
    
    # hist_cols = np.array(['#606cbe', '#48a7b1', '#deb35e', '#de8468'])
    # hist_cols = hist_cols[[1,0,2,3]]
    
    single_b = np.reshape(betas_single[2][neuron_id,enc['variable_index']['previous_choice_outcome_combination']], (4,133)).T
    full_b = np.reshape(full_betas[neuron_id,enc['variable_index']['previous_choice_outcome_combination']], (4,133)).T
    
    order = [[0,1],[0,0],[1,0],[1,1]] #History order is incor left, corr left, incorr right, corr right, which will be on top right, top left, bottom left and bottom right
    fi, ax = plt.subplots(nrows = 2, ncols=2, sharex=True, sharey=True) 
    for k in range(len(order)):
        plot_timecourse(ax[order[k][0], order[k][1]], [single_b[:,k], full_b[:,k]], 30, idx_list, colors = cols)
        # ax[order[k][0], order[k][1]].plot(single_b[:,k],color='k', linestyle='--')
        # ax[order[k][0], order[k][1]].plot(full_b[:,k],color='k')
    fi.savefig(os.path.join(save_location, f'N{neuron_id}_trial_history.svg'))
    
    #----Plot head orientation tuning
    orientation_angle = 'yaw'
    line_color = '#99733e'
    #Get the bin definitions as used in the encoding model fitting
    bin_number = 60 #6°
    bins = np.arange(-np.pi, np.pi+np.pi/(bin_number/2), np.pi/(bin_number/2))[:bin_number+1]
    full_tuning = full_betas[neuron_id, enc['variable_index'][orientation_angle]]
    single_tuning = betas_single[9][neuron_id, enc['variable_index'][orientation_angle]]
    
    fi = plt.figure()
    ax = fi.add_subplot(111, projection='polar')
    ax.plot(bins, np.zeros(bins.shape[0]), color='k', linewidth= ax.spines['inner'].get_linewidth(), linestyle='--')
    ax.plot(bins[:-1], single_tuning, color = cols[0])
    ax.plot([bins[-2], bins[-1]], [single_tuning[-1], single_tuning[0]], color = cols[0])
    
    ax.plot(bins[:-1], full_tuning, color = cols[1])
    ax.plot([bins[-2], bins[-1]], [full_tuning[-1], full_tuning[0]], color = cols[1])
    fi.savefig(os.path.join(save_location, f'N{neuron_id}_yaw.svg'))
    
    #------Now chest point tuning
    colormap_str = 'BrBG_r'
    
    video_dimensions = [512,640]
    pixels_per_bin = 32 #At 25 px/cm, so that each place field is 1.28 cm**2
    x_bins = np.arange(0, video_dimensions[1] + pixels_per_bin, pixels_per_bin) #Divide video width in bins
    y_bins = np.arange(0, video_dimensions[0] + pixels_per_bin, pixels_per_bin)
    
    
    #Get the weights, reshape to image dimensions and interpolate to actual pixel values
    cb = full_betas[neuron_id,  enc['variable_index']['chest_point']]
    beta_image = np.reshape(cb, (x_bins.shape[0]-1, y_bins.shape[0]-1)).T
    full_interpolated = zoom(beta_image, 32, order = 3)
    
    cb = betas_single[10][neuron_id,  enc['variable_index']['chest_point']]
    beta_image = np.reshape(cb, (x_bins.shape[0]-1, y_bins.shape[0]-1)).T
    single_interpolated = zoom(beta_image, 32, order = 3)
    
    a_min = np.min(np.vstack((full_interpolated, single_interpolated)))
    b_max = np.max(np.vstack((full_interpolated, single_interpolated)))
    bound = np.max([b_max, np.abs(a_min)])
    
    fig = plt.figure()
    ax = fig.add_subplot()
    im = ax.imshow(np.flipud(single_interpolated), cmap = colormap_str, vmin= -bound, vmax=bound)
    cax = fig.add_axes([ax.get_position().x1+0.01,ax.get_position().y0,0.02,ax.get_position().height])
    cbar = plt.colorbar(im, cax=cax)
    ax.set_xticks([])
    ax.set_yticks([])
    cbar.set_label('\u03b2 weights')
    ax.set_title('Chest point only')
    fig.savefig(os.path.join(save_location, f'N{neuron_id}_chest_single_var.svg'))
    
    fig = plt.figure()
    ax = fig.add_subplot()
    im = ax.imshow(np.flipud(full_interpolated), cmap = colormap_str, vmin= -bound, vmax=bound)
    cax = fig.add_axes([ax.get_position().x1+0.01,ax.get_position().y0,0.02,ax.get_position().height])
    cbar = plt.colorbar(im, cax=cax)
    ax.set_xticks([])
    ax.set_yticks([])
    cbar.set_label('\u03b2 weights')
    ax.set_title('Full model')
    fig.savefig(os.path.join(save_location, f'N{neuron_id}_chest_single_var.svg'))
    
    
#%%-----Also make correlation matrices for dR and cvR

tmp  = [np.vstack(cvR), np.vstack(dR)]

for k in tmp:
    R = np.corrcoef(k)
    np.fill_diagonal(R,0) #Works within array without reassigning new variable

    fi = plt.figure()
    ax =fi.add_subplot(111, aspect='equal')
    im = ax.matshow(R, vmin=-1, vmax=1, cmap='RdBu_r')
    fi.colorbar(im)
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xticks(np.arange(R.shape[0]))
    ax.set_yticks(np.arange(R.shape[0]))
    
    
#%%
    
# #%%
# #The harder task: make a correlation matrix for design matrix
# from os.path import splitext
# import itertools
# #Copied from ridge_cv_encoding_models_partitioned 
# signal_type = 'F'

# trial_alignment_file = glob(session_dir + '/analysis/*miniscope_data.npy')[0]
# miniscope_data = np.load(trial_alignment_file, allow_pickle = True).tolist()  
# frame_rate = miniscope_data['frame_rate']
# trialdata = pd.read_hdf(glob(session_dir + '/chipmunk/*.h5')[0], '/Data')

# #Get video alignment
# video_alignment_files = glob(session_dir + '/analysis/*video_alignment.npy')
# if len(video_alignment_files) > 1:
#         print('More than one video is currently not supported')
# video_alignment = np.load(video_alignment_files[0], allow_pickle = True).tolist()

# #Retrieve dlc tracking
# dlc_file = glob(session_dir + '/dlc_analysis/*.h5')[-1] #Load the latest extraction!
# dlc_data = pd.read_hdf(dlc_file)
# dlc_metadata = pd.read_pickle(splitext(dlc_file)[0] + '_meta.pickle')

# #Load the video components and the video motion energy components
# video_svd = np.load(glob(session_dir + '/analysis/*video_SVD.npy')[0], allow_pickle = True).tolist()[1].T #Only load temporal components
# me_svd = np.load(glob(session_dir + '/analysis/*motion_energy_SVD.npy')[0], allow_pickle = True).tolist()[1].T #Only load temporal components 

# #Define the shift for the event-based regressors
# poke_min_shift = -0.5 * frame_rate
# poke_max_shift = 1 * frame_rate

# #Also do this for the stimulus events
# stim_min_shift = 0
# stim_max_shift = 0.5 * frame_rate

# #Set the times up
# aligned_to = ['outcome_presentation', 'outcome_end', 'PlayStimulus', 'DemonWaitForResponse']
# time_frame = [np.array([round(0*frame_rate), round(1*frame_rate)+1], dtype=int),
#               np.array([round(-1*frame_rate), round(0.3*frame_rate)+1], dtype=int),
#               np.array([round(-0.5*frame_rate), round(1*frame_rate)+1], dtype=int),
#               np.array([round(-0.2*frame_rate), round(0.3*frame_rate)+1], dtype=int)]

# #Assemble the task variable design matrix
# choice = np.array(trialdata['response_side'])
# category = np.array(trialdata['correct_side'])
# prior_choice =  chiCa.determine_prior_variable(np.array(trialdata['response_side']), np.ones(len(trialdata)), 1, 'consecutive')
# prior_category =  chiCa.determine_prior_variable(np.array(trialdata['correct_side']), np.ones(len(trialdata)), 1, 'consecutive')

# outcome = np.array(trialdata['response_side'] == trialdata['correct_side'], dtype=float) #Define as float so that nans (that are float) can be retained
# outcome[np.array(np.isnan(trialdata['response_side']))] = np.nan
# prior_outcome =  chiCa.determine_prior_variable(outcome, np.ones(len(trialdata)), 1, 'consecutive')
   
# modality = np.zeros([trialdata.shape[0]])
# modality[trialdata['stimulus_modality'] == 'auditory'] = 1
# prior_modality = chiCa.determine_prior_variable(modality, np.ones(len(trialdata)), 1, 'consecutive')

# #Find stimulus strengths
# tmp_stim_strengths = np.zeros([trialdata.shape[0]], dtype=int) #Filter to find easiest stim strengths
# for k in range(trialdata.shape[0]):
#     tmp_stim_strengths[k] = trialdata['stimulus_event_timestamps'][k].shape[0]

# #Get category boundary to normalize the stim strengths
# unique_freq = np.unique(tmp_stim_strengths)
# category_boundary = (np.min(unique_freq) + np.max(unique_freq))/2
# stim_strengths = (tmp_stim_strengths- category_boundary) / (np.max(unique_freq) - category_boundary)
# if trialdata['stimulus_modality'][0] == 'auditory':
#     stim_strengths = stim_strengths * -1
    
# difficulty = 1 - np.abs(stim_strengths)
# prior_difficulty =  chiCa.determine_prior_variable(difficulty, np.ones(len(trialdata)), 1, 'consecutive')
    
# bin_number = 60 #6°
# bins = np.arange(-np.pi, np.pi+np.pi/(bin_number/2), np.pi/(bin_number/2))[:bin_number+1] #Divide bin number by 2 because 360 deg is 2 pi
# angles = ['pitch', 'roll', 'yaw']

# for a in angles:
#     bin_id = np.digitize(miniscope_data[a], bins, right = False)
#     tmp = np.zeros([miniscope_data[a].shape[0],bins.shape[0]-1]) #The last bin is the upper boundary, pi
#     for k in range(bin_id.shape[0]):
#         tmp[k,bin_id[k]-1] = 1 #Weird behavior so that the first bin is at 1!
#     exec(f'{a} = tmp')

# #Extract a set of dlc labels and standardize these.
# dlc_keys = dlc_data.keys().tolist()
# specifier = dlc_keys[0][0] #Retrieve the session specifier that is needed as a keyword
# body_part_name = dlc_metadata['data']['DLC-model-config file']['all_joints_names']

# temp_body_parts = []
# part_likelihood_estimate = []
# for bp in body_part_name:
#     for axis in ['x', 'y']:
#         temp_body_parts.append(np.array(dlc_data[(specifier, bp, axis)]))
#     part_likelihood_estimate.append(np.array(dlc_data[(specifier, bp, 'likelihood')]))

# body_parts = np.array(temp_body_parts).T #To array and transpose
# part_likelihood_estimate = np.array(part_likelihood_estimate).T

# #Use the chest (body part 8 and 9)
# part_id = 8
# video_dimensions = np.array( dlc_metadata['data']['frame_dimensions']) #first value is y second is x
# pixels_per_bin = 32 #At 25 px/cm, so that each place field is 1.28 cm**2

# x_bins = np.arange(0, video_dimensions[1] + pixels_per_bin, pixels_per_bin) #Divide video width in bins
# y_bins = np.arange(0, video_dimensions[0] + pixels_per_bin, pixels_per_bin)
# combinations = np.array(list(itertools.product(np.arange(x_bins.shape[0]-1), np.arange(y_bins.shape[0]-1)))) #Get all possible combinations that may occur

# what_bin = np.vstack((np.digitize(body_parts[:,part_id], x_bins, right = False), np.digitize(body_parts[:,part_id+1], y_bins, right = False))).T

# position_vect = np.zeros([body_parts.shape[0],combinations.shape[0]]) #A linearized version of bin occupancy at each time point
# for k in range(combinations.shape[0]):
#     idx = np.where((what_bin == combinations[k,:]).all(1))[0]
#     if idx.shape[0] > 0:
#         position_vect[idx, k] = 1
        
# #Align the different poke events    
# actions = ['Port2In','Port1In', 'Port3In']
# response_matrix = []
# for k in range(len(actions)):
#     event_start_frame, trial_id, state_time_covered = chiCa.align_miniscope_to_event(trialdata[actions[k]].tolist(),
#                                                                                      np.array(trialdata['FinishTrial'].tolist())[:,0],
#                                                                                      miniscope_data['frame_interval'],
#                                                                                      miniscope_data['trial_starts'])
#     event_trace = chiCa.assemble_event_trace(event_start_frame, miniscope_data['F'].shape[1])
#     response_matrix.append(chiCa.shift_regressor(event_trace, int(poke_min_shift), int(poke_max_shift)))
# response_matrix = np.hstack(response_matrix)

# #Now align the stimulus regressor
# t_stamps = chiCa.get_experienced_stimulus_events(trialdata, stim_modalities = ['visual', 'auditory', 'audio-visual'])
# event_start_frame, trial_id, state_time_covered = chiCa.align_miniscope_to_event(t_stamps,
#                                                                                      np.array(trialdata['FinishTrial'].tolist())[:,0],
#                                                                                      miniscope_data['frame_interval'],
#                                                                                      miniscope_data['trial_starts'])
# event_trace = chiCa.assemble_event_trace(event_start_frame, miniscope_data['F'].shape[1])
# stim_matrix = chiCa.shift_regressor(event_trace, int(stim_min_shift), int(stim_max_shift))

# # val_trials = [valid_trials_before] + [valid_trials]*3
# valid_trials = np.where((np.isnan(choice) == 0) & (np.isnan(prior_choice) == 0))[0]
# valid_trials_before = valid_trials-1
# val_trials =  [valid_trials_before]*2 + [valid_trials]*2

# #Now start settting up the regressors
# individual_reg_idx = [] #Keep indices of individual regressors
# reg_group_idx = [] #Keep indices of regressor groups

# total_frames = []
# for k in time_frame:
#     total_frames.append(k[1] - k[0])
# total_frames = np.sum(total_frames)

# block = np.zeros([total_frames, total_frames])
# for k in range(block.shape[0]):
#     block[k,k] = 1
  

# #Stack the blocks of cognitive regressors and multiply them by the respective value
# time_reg = np.array(block)
# choice_x = block * choice[valid_trials[0]]
# outcome_x = block * outcome[valid_trials[0]]
# prior_difficulty_x = block  * prior_difficulty[valid_trials[0]]

# #Include all required interactions too, the one captured in the default is left choice undrewarded
# prior_incorrect_left_x = block * ((prior_choice[valid_trials[0]] == 0) & (prior_outcome[valid_trials[0]]==0))
# prior_correct_left_x = block * ((prior_choice[valid_trials[0]] == 0) & (prior_outcome[valid_trials[0]]==1))
# prior_incorrect_right_x = block * ((prior_choice[valid_trials[0]] == 1) & (prior_outcome[valid_trials[0]]==0))
# prior_correct_right_x = block * ((prior_choice[valid_trials[0]] == 1) & (prior_outcome[valid_trials[0]]==1))


# for k in range(1, valid_trials.shape[0]):
#       time_reg = np.vstack((time_reg, block))
#       choice_x = np.vstack((choice_x, block * choice[valid_trials[k]]))
#       prior_difficulty_x = np.vstack((prior_difficulty_x, block * prior_difficulty[valid_trials[k]]))
#       outcome_x = np.vstack((outcome_x, block * outcome[valid_trials[k]]))
#       prior_incorrect_left_x = np.vstack((prior_incorrect_left_x, block * ((prior_choice[valid_trials[k]] == 0) & (prior_outcome[valid_trials[k]]==0))))
#       prior_correct_left_x = np.vstack((prior_correct_left_x, block * ((prior_choice[valid_trials[k]] == 0) & (prior_outcome[valid_trials[k]]==1))))
#       prior_incorrect_right_x = np.vstack((prior_incorrect_right_x, block * ((prior_choice[valid_trials[k]] == 1) & (prior_outcome[valid_trials[k]]==0))))
#       prior_correct_right_x = np.vstack((prior_correct_right_x, block * ((prior_choice[valid_trials[k]] == 1) & (prior_outcome[valid_trials[k]]==1))))

# #Dropped previous difficulty here for the sake of clarity
# x_include = np.hstack((time_reg, choice_x, outcome_x, prior_incorrect_left_x, prior_correct_left_x, prior_incorrect_right_x, prior_correct_right_x)) 

# x_analog = []
# part_likelihood = []
# #Also store the timestamps that are included into the trialized design matrix
# trial_timestamps_imaging = []
# trial_timestamps_video = []
# for k in range(len(aligned_to)):
#     state_start_frame, state_time_covered = chiCa.find_state_start_frame_imaging(aligned_to[k], trialdata, miniscope_data['frame_interval'], miniscope_data['trial_starts'], miniscope_data['trial_start_time_covered'])                                                                     
#     zero_frame = np.array(state_start_frame[val_trials[k]] + time_frame[k][0], dtype=int) #The firts frame to consider
    
#     for add_to in np.arange(time_frame[k][1] - time_frame[k][0]):
#         matching_frames = []
#         for q in range(zero_frame.shape[0]): #unfortunately need to loop through the trials, should be improved in the future...
#                 tmp = chiCa.match_video_to_imaging(np.array([zero_frame[q] + add_to]), miniscope_data['trial_starts'][val_trials[k][q]],
#                        miniscope_data['frame_interval'], video_alignment['trial_starts'][val_trials[k][q]], video_alignment['frame_interval'])[0].astype(int)
#                 matching_frames.append(tmp)
       
#         x_analog.append(np.concatenate((stim_matrix[zero_frame + add_to,:], response_matrix[zero_frame + add_to,:], pitch[zero_frame + add_to,:], roll[zero_frame + add_to,:], yaw[zero_frame + add_to,:], position_vect[matching_frames,:], video_svd[matching_frames,:], me_svd[matching_frames,:]), axis=1))
        
#         part_likelihood.append(part_likelihood_estimate[matching_frames,:])
#         trial_timestamps_imaging.append(zero_frame + add_to)
#         trial_timestamps_video.append(matching_frames)
        
# #Back to array like, where columns are trials, rows time points and sheets cells   
# x_analog = np.squeeze(x_analog)
# x_analog = np.reshape(x_analog, (x_include.shape[0], x_analog.shape[2]), order = 'F')

# part_likelihood = np.squeeze(part_likelihood)

# #Transform to timepoint x (valid) trial matrix
# trial_timestamps_imaging = np.squeeze(trial_timestamps_imaging)
# trial_timestamps_video = np.squeeze(trial_timestamps_video)

# #Add the analog regressors to the cognitive regressor design matrix
# X = np.hstack((x_include, x_analog))  
# #Track where the regressors and regressor groups live inside the design matrix
# #regressor_labels = ['trial_time', 'choice', 'stim_strength', 'outcome', 'previous_incorrect_left', 'previous_correct_left', 'previous_incorrect_right', 'previous_correct_right', 'stim_events', 'center_poke', 'left_poke', 'right_poke', 'pitch', 'roll', 'yaw', 'chest_point', 'video_svd', 'video_me_svd']
# regressor_labels = ['trial_time', 'choice',  'outcome', 'previous_choice_outcome_combination', 'stim_events', 'center_poke', 'left_poke', 'right_poke', 'pitch', 'roll', 'yaw', 'chest_point', 'video_svd', 'video_me_svd']
# regressor_idx = []
# loop_range = [x for x in range(len(regressor_labels)) if regressor_labels[x] == 'previous_choice_outcome_combination'][0]
# for k in range(loop_range): #time, choice, stim strength, outcome
#    regressor_idx.append(np.arange(k*block.shape[0], (k+1)*block.shape[0]))
# regressor_idx.append(np.arange(regressor_idx[-1][-1]+1, regressor_idx[-1][-1]+1 + 4*block.shape[0]))
# regressor_idx.append(np.arange(regressor_idx[-1][-1]+1, regressor_idx[-1][-1]+1 + stim_matrix.shape[1]))
# for k in range(3): #All different poke events are shifted with the same lags
#     regressor_idx.append(np.arange(regressor_idx[-1][-1]+1, regressor_idx[-1][-1]+1 + int(response_matrix.shape[1]/3)))
# for k in range(3):
#     regressor_idx.append(np.arange(regressor_idx[-1][-1]+1, regressor_idx[-1][-1]+1 + pitch.shape[1]))
# regressor_idx.append(np.arange(regressor_idx[-1][-1]+1, regressor_idx[-1][-1]+1 + position_vect.shape[1]))
# regressor_idx.append(np.arange(regressor_idx[-1][-1]+1, regressor_idx[-1][-1]+1 + video_svd.shape[1]))      
# regressor_idx.append(np.arange(regressor_idx[-1][-1]+1, regressor_idx[-1][-1]+1 + me_svd.shape[1]))
                     
# individual_regressors = dict()
# for k in range(len(regressor_labels)):
#     individual_regressors[regressor_labels[k]] = regressor_idx[k]
#     #Unfortunately also ordered alphabetically

# #Update info about the regressor indices ,always remember uppper exclusive
# reg_group_idx = [np.arange(regressor_idx[1][0], regressor_idx[7][0])] #The task regressors (chocie, prior, etc. + stimulus events) exclusive the intercept
# reg_group_idx.append(np.arange(regressor_idx[7][0],regressor_idx[10][0])) #Instructed actions (center and side poking)
# reg_group_idx.append(np.arange(regressor_idx[10][0],regressor_idx[-1][-1])) #Uninstructed movements (head orientation tuning, chest position, video)

# #Dictionary of regressor idx for different variales
# variable_idx = dict()
# for k in range(len(regressor_labels)):
#     variable_idx[regressor_labels[k]] = regressor_idx[k] 

# standardize_reg = np.arange(individual_regressors['video_svd'][0], individual_regressors['video_me_svd'][-1]+1) #Only z-score the truly anlog video variables
# #Video_svd is the first truly analog regressor, and video_me_svd is the last one of that kind.

# Q_vid = np.linalg.qr(np.hstack((X[:,individual_regressors['stim_events']], X[:,individual_regressors['video_svd']])))[0][:,individual_regressors['stim_events'].shape[0]:]
# Q_me =np.linalg.qr(np.hstack((X[:,individual_regressors['stim_events']], X[:,individual_regressors['video_me_svd']])))[0][:,individual_regressors['stim_events'].shape[0]:]
    
# X[:, individual_regressors['video_svd']] = Q_vid #Reassemble the design matrix
# X[:, individual_regressors['video_me_svd']] = Q_me


# #%%

# R = np.corrcoef(X.T)
