# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 00:11:53 2025

@author: Lukas Oesch
"""

import numpy as np
import pandas as pd
import glob
#from time import time
from os.path import splitext, join
import itertools

from sklearn.model_selection import KFold, StratifiedKFold
from scipy.ndimage import gaussian_filter1d

# import regbench as rb
import chiCa
from scipy.stats import zscore
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

#%%
for session_dir in sessions:
    #------Some parameter definitions--------------------
    #session_dir = 'C:/data/LO068/20230831_132717'
    print(f'Starting: {session_dir}')
    file_name = 'RidgeCV_encoding_models_ITI_partitioned_Fig3'
    
    signal_type = 'F'
    
    k_folds = 10 #For regular random cross-validation
    
    # fit_regressor_group = True #Groups refers to a set of regressors obtained from one modality, like cognitive regressors from
    # #task events, or head orientation from miniscope gyro.
    # #If set to false the model will be fit to individual regressors, excluding
    # #the intercept regressors
    
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
    
    #%%------Loading the data--------------------
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
    
    #%%--------Alignments and construction of the design matrix-----------------
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
        
    #%% 
    #New part for head orientation tuning
    #Retrieve head direction and map to carthesian coordinates
    # head_orientation = np.stack((np.cos(miniscope_data['pitch']), np.sin(miniscope_data['pitch']),
    #                              np.cos(miniscope_data['roll']), np.sin(miniscope_data['roll']),
    #                              np.cos(miniscope_data['yaw']), np.sin(miniscope_data['yaw'])),axis=1)
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
    #%%
    
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
            
    #%%---------Extract instructed movement regressors and stimulus
    
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
    
    #%%
    #Find the valid trials to be included the criteria are the following:
    #There has to be a prior choice a current choice and the stimulus is one of the easy two
    # valid_trials = np.where((np.isnan(choice) == 0) & (np.isnan(prior_choice) == 0))[0]
    # valid_trials_before = valid_trials-1
    
    # val_trials = [valid_trials_before] + [valid_trials]*3
    valid_trials = np.where((np.isnan(choice) == 0) & (np.isnan(prior_choice) == 0))[0]
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
    prior_difficulty_x = block  * prior_difficulty[valid_trials[0]]
    
    #Include all required interactions too, the one captured in the default is left choice undrewarded
    prior_incorrect_left_x = block * ((prior_choice[valid_trials[0]] == 0) & (prior_outcome[valid_trials[0]]==0))
    prior_correct_left_x = block * ((prior_choice[valid_trials[0]] == 0) & (prior_outcome[valid_trials[0]]==1))
    prior_incorrect_right_x = block * ((prior_choice[valid_trials[0]] == 1) & (prior_outcome[valid_trials[0]]==0))
    prior_correct_right_x = block * ((prior_choice[valid_trials[0]] == 1) & (prior_outcome[valid_trials[0]]==1))
    
    
    for k in range(1, valid_trials.shape[0]):
          time_reg = np.vstack((time_reg, block))
          # incorrect_left_x = np.vstack((incorrect_left_x, block * ((choice[valid_trials[k]] == 0) & (outcome[valid_trials[k]]==0))))
          # correct_left_x = np.vstack((correct_left_x, block * ((choice[valid_trials[k]] == 0) & (outcome[valid_trials[k]]==1))))
          # incorrect_right_x = np.vstack((incorrect_right_x, block * ((choice[valid_trials[k]] == 1) & (outcome[valid_trials[k]]==0))))
          # correct_right_x = np.vstack((correct_right_x, block * ((choice[valid_trials[k]] == 1) & (outcome[valid_trials[k]]==1))))
          
          choice_x = np.vstack((choice_x, block * choice[valid_trials[k]]))
          # stim_tmp = np.array(block)
          # stim_tmp[:total_frames, :] = stim_tmp[:total_frames, :] * stim_strengths[before_after[0][k]]
          # stim_tmp[total_frames:, :] = stim_tmp[total_frames:, :] * stim_strengths[before_after[1][k]]
          prior_difficulty_x = np.vstack((prior_difficulty_x, block * prior_difficulty[valid_trials[k]]))
          outcome_x = np.vstack((outcome_x, block * outcome[valid_trials[k]]))
          prior_incorrect_left_x = np.vstack((prior_incorrect_left_x, block * ((prior_choice[valid_trials[k]] == 0) & (prior_outcome[valid_trials[k]]==0))))
          prior_correct_left_x = np.vstack((prior_correct_left_x, block * ((prior_choice[valid_trials[k]] == 0) & (prior_outcome[valid_trials[k]]==1))))
          prior_incorrect_right_x = np.vstack((prior_incorrect_right_x, block * ((prior_choice[valid_trials[k]] == 1) & (prior_outcome[valid_trials[k]]==0))))
          prior_correct_right_x = np.vstack((prior_correct_right_x, block * ((prior_choice[valid_trials[k]] == 1) & (prior_outcome[valid_trials[k]]==1))))

    #x_include = np.hstack((time_reg, choice_x, outcome_x, prior_difficulty_x, prior_incorrect_left_x, prior_correct_left_x, prior_incorrect_right_x, prior_correct_right_x)) 
    #Dropped previous difficulty here for the sake of clarity
    x_include = np.hstack((time_reg, choice_x, outcome_x, prior_incorrect_left_x, prior_correct_left_x, prior_incorrect_right_x, prior_correct_right_x)) 
    
#%%
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
    
  #%%
    #Track where the regressors and regressor groups live inside the design matrix
    #regressor_labels = ['trial_time', 'choice', 'stim_strength', 'outcome', 'previous_incorrect_left', 'previous_correct_left', 'previous_incorrect_right', 'previous_correct_right', 'stim_events', 'center_poke', 'left_poke', 'right_poke', 'pitch', 'roll', 'yaw', 'chest_point', 'video_svd', 'video_me_svd']
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
    
   #%%-------Draw the training and testing splits-------------------
    
    #First get the splits for training and testing sets. These will be constant throughout
    #kf = KFold(n_splits = k_folds, shuffle = True) 
    kf = StratifiedKFold(n_splits = k_folds, shuffle = True)
    timepoint_labels = np.tile(np.arange(block.shape[0]),valid_trials.shape[0])
    k_fold_generator = kf.split(X, timepoint_labels) #This returns a generator object that spits out a different split at each call
    training = []
    testing = []
    for draw_num in range(k_folds):
        tr, te = k_fold_generator.__next__()
        training.append(tr)
        testing.append(te)
     #%%------------Determine which regressor to shuffle
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
    
    #%%---------Start the model fitting------------------------------
    
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
     
        alphas, betas, r_squared, corr, y_test, y_hat = chiCa.fit_ridge_cv_shuffles(X, Y, alpha_range, alpha_per_target, fit_intercept, shuffle_regressor[s_round], standardize_reg, training, testing)    
        time_r, time_corr = chiCa.r_squared_timecourse(y_test, y_hat, testing, block.shape[0])
        all_alphas.append(alphas)
        all_betas.append(betas)
        all_rsquared.append(r_squared)
        all_corr.append(corr)
        all_r_timecourse.append(time_r)
        all_corr_timecourse.append(time_corr)
      
    # del X
    # del x_analog
    # del x_include
    # del Y
    # del dlc_data
    
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
    
