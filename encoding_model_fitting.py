# -*- coding: utf-8 -*-
"""
This script performs the encoding model fits for all the sessions included in
the analyses.

Created on Mon Feb 10 00:11:53 2025

@author: Lukas Oesch
"""

import numpy as np
import pandas as pd
import glob
from os.path import splitext, join
import itertools
from labdatatools import *
from sklearn.model_selection import KFold, StratifiedKFold
from scipy.ndimage import gaussian_filter1d
import chiCa # Make sure to be inside your chiCa folder to run this import
from scipy.stats import zscore

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

############################################################################3
#%%------------Define some of the model parameters and start looping through sessions
for session_dir in sessions:
   
    print(f'Starting: {session_dir}')
    file_name = 'RidgeCV_encoding_models_ITI_partitioned_Fig3'
    signal_type = 'F' # Use the detrended raw calcium data to fit encoding models.
    # The raw fluorescence traces are more continuous than the deconvolved 
    # traces and tend to yield normally distributed residuals after the encoding
    # model fits.
    k_folds = 10 # Perfrom 10-fold cross-validation
    
    which_models = 'single_var' #'cognitive' #'group' # 'individual', 'timepoint'
    #Determines what type of models should be fitted. 'group' will lump specified regressors
    #into a group and fit models for the cvR2 and dR2 for each of the groups, 'individual'
    #will assess the explained variance for each regressor alone and 'timepoint' will
    #look at the task regressors collectively but evaluate the model performance at each
    #trial time separately, 'single_var' will fit all the regressors that 
    # belong to an individual variable together, for example all the 133 choice 
    # timepoint regressors.
    
    add_complete_shuffles = 0 #Allows one to add models where all the regressors
    #are shuffled independently. This can be used to generate a null distribution for
    #the beta weights of certain regressors
    
    #%%------Loading the data--------------------
    
    # Get neural and behavioral data
    trial_alignment_file = glob(session_dir + '/analysis/*miniscope_data.npy')[0]
    miniscope_data = np.load(trial_alignment_file, allow_pickle = True).tolist()  
    frame_rate = miniscope_data['frame_rate']
    trialdata = pd.read_hdf(glob(session_dir + '/chipmunk/*.h5')[0], '/Data')
    
    #Get video alignment
    video_alignment_files = glob(session_dir + '/analysis/*video_alignment.npy')
    if len(video_alignment_files) > 1:
            print('More than one video is currently not supported, picking the first one...')
    video_alignment = np.load(video_alignment_files[0], allow_pickle = True).tolist()
    
    #Retrieve dlc tracking
    dlc_file = glob(session_dir + '/dlc_analysis/*.h5')[-1] #Load the latest extraction!
    dlc_data = pd.read_hdf(dlc_file)
    dlc_metadata = pd.read_pickle(splitext(dlc_file)[0] + '_meta.pickle')
    
    #Load the video components and the video motion energy components
    video_svd = np.load(glob(session_dir + '/analysis/*video_SVD.npy')[0], allow_pickle = True).tolist()[1].T #Only load temporal components
    me_svd = np.load(glob(session_dir + '/analysis/*motion_energy_SVD.npy')[0], allow_pickle = True).tolist()[1].T #Only load temporal components 
    
#############-------Align regressors and neural signas and construct the design matrix---############################################################
    #%%-----Get neural frame alignment, the trial-wise task variables and find valid trials-----------------
   
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
    
    ########---Optional: if one wants to look at other task variables--##############3
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
    ####################################################################################################    
    
    #Find the valid trials. Here, we have to consider that for the neural activity
    #the timestamps of early and late ITI are still part of the previous trial.
    # Thus we get a list of valid_trials for each of the trial phases.
    valid_trials = np.where((np.isnan(choice) == 0) & (np.isnan(prior_choice) == 0))[0]
    valid_trials_before = valid_trials-1
    val_trials =  [valid_trials_before]*2 + [valid_trials]*2
    
   #%%-------------Construct the full trial time-shifted regressors for choice,
   # outcome and trial history plus the time regressor.
    #Now start settting up the regressors
    individual_reg_idx = [] #Keep indices of individual regressors
    reg_group_idx = [] #Keep indices of regressor groups
    
    # Determine how many frames in total will be included
    total_frames = []
    for k in time_frame:
        total_frames.append(k[1] - k[0])
    total_frames = np.sum(total_frames)
    
    #Construct a square matrix of zeros with ones on the diagonal. This will be
    #our time regressor.
    block = np.zeros([total_frames, total_frames])
    for k in range(block.shape[0]):
        block[k,k] = 1   
    
    #Use this initial square matrix mutiply it with the value of the task variable
    #for this trial and vertically stack the result onto the already existing 
    #matrix. 
    
    #Initialize the regressor blocks for the first trial
    time_reg = np.array(block) #This can be understood as a time-varying intercept for all the shifted task variable regressors
    choice_x = block * choice[valid_trials[0]] #0 encodes left choice and 1 encodes right choice
    outcome_x = block * outcome[valid_trials[0]] #0 encodes incorrect and 1 correct trials
    
    #Include all required trial history interactions too, the one captured in the default is left choice undrewarded
    prior_incorrect_left_x = block * ((prior_choice[valid_trials[0]] == 0) & (prior_outcome[valid_trials[0]]==0))
    prior_correct_left_x = block * ((prior_choice[valid_trials[0]] == 0) & (prior_outcome[valid_trials[0]]==1))
    prior_incorrect_right_x = block * ((prior_choice[valid_trials[0]] == 1) & (prior_outcome[valid_trials[0]]==0))
    prior_correct_right_x = block * ((prior_choice[valid_trials[0]] == 1) & (prior_outcome[valid_trials[0]]==1))
    
    
    for k in range(1, valid_trials.shape[0]):
          time_reg = np.vstack((time_reg, block))
          choice_x = np.vstack((choice_x, block * choice[valid_trials[k]]))
          outcome_x = np.vstack((outcome_x, block * outcome[valid_trials[k]]))
          prior_incorrect_left_x = np.vstack((prior_incorrect_left_x, block * ((prior_choice[valid_trials[k]] == 0) & (prior_outcome[valid_trials[k]]==0))))
          prior_correct_left_x = np.vstack((prior_correct_left_x, block * ((prior_choice[valid_trials[k]] == 0) & (prior_outcome[valid_trials[k]]==1))))
          prior_incorrect_right_x = np.vstack((prior_incorrect_right_x, block * ((prior_choice[valid_trials[k]] == 1) & (prior_outcome[valid_trials[k]]==0))))
          prior_correct_right_x = np.vstack((prior_correct_right_x, block * ((prior_choice[valid_trials[k]] == 1) & (prior_outcome[valid_trials[k]]==1))))
    
    #Horizontally stack the different variables 
    x_include = np.hstack((time_reg, choice_x, outcome_x, prior_incorrect_left_x, prior_correct_left_x, prior_incorrect_right_x, prior_correct_right_x)) 
    
   #%%-------Construct partially shifted regressors for individual stimulus-
   # and poke events
   
    #Define the shift for the event-based regressors
    poke_min_shift = -0.5 * frame_rate #Allow a movement preparation signal 0.5 s before the poke to influence neural activity
    poke_max_shift = 1 * frame_rate #Allow poke to influence neural activity for up to 1 s after the event
    
    #Also do this for the stimulus events
    stim_min_shift = 0 #Stimuli can't be anticipated, so start at 0
    stim_max_shift = 0.5 * frame_rate
    
    #First, align the poke events
    actions = ['Port2In','Port1In', 'Port3In']
    response_matrix = []
    for k in range(len(actions)):
        #Find the imaging frame during which the poke event happened
        event_start_frame, trial_id, state_time_covered = chiCa.align_miniscope_to_event(trialdata[actions[k]].tolist(),
                                                                                         np.array(trialdata['FinishTrial'].tolist())[:,0],
                                                                                         miniscope_data['frame_interval'],
                                                                                         miniscope_data['trial_starts'])
        #Generate a trace of 0s that is 1 whenever the event happens. This trace has the 
        #same length as the imaging data
        event_trace = chiCa.assemble_event_trace(event_start_frame, miniscope_data['F'].shape[1])
        #Generate time-shifted copies of this trace according to the shifts defined
        #above. 
        response_matrix.append(chiCa.shift_regressor(event_trace, int(poke_min_shift), int(poke_max_shift)))
    response_matrix = np.hstack(response_matrix)
    
    #Now align the stimulus regressor, use the same approach
    t_stamps = chiCa.get_experienced_stimulus_events(trialdata, stim_modalities = ['visual', 'auditory', 'audio-visual'])
    event_start_frame, trial_id, state_time_covered = chiCa.align_miniscope_to_event(t_stamps,
                                                                                         np.array(trialdata['FinishTrial'].tolist())[:,0],
                                                                                         miniscope_data['frame_interval'],
                                                                                         miniscope_data['trial_starts'])
    event_trace = chiCa.assemble_event_trace(event_start_frame, miniscope_data['F'].shape[1])
    stim_matrix = chiCa.shift_regressor(event_trace, int(stim_min_shift), int(stim_max_shift))
    
    
    #%%------------Construct the head-orientation tuning regressors-------
   
    bin_number = 60 #6° head-orientation angle bins for the tuning 
    bins = np.arange(-np.pi, np.pi+np.pi/(bin_number/2), np.pi/(bin_number/2))[:bin_number+1] #Divide bin number by 2 because 360 deg is 2 pi
    angles = ['pitch', 'roll', 'yaw']
    
    for a in angles:
        bin_id = np.digitize(miniscope_data[a], bins, right = False)
        tmp = np.zeros([miniscope_data[a].shape[0],bins.shape[0]-1]) #The last bin is the upper boundary, pi
        for k in range(bin_id.shape[0]):
            tmp[k,bin_id[k]-1] = 1 #Weird behavior so that the first bin is at 1!
        exec(f'{a} = tmp')
    
    #%%--------Construct the chest position tuning regressors
    
    #Extract dlc label positions and stack to matrix
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
    
    #Use the chest (body part index 8 and 9)
    part_id = 8
    video_dimensions = np.array(dlc_metadata['data']['frame_dimensions']) #first value is y second is x
    pixels_per_bin = 32 #At 25 px/cm, so that each place field is 1.28 cm**2
    #Map the bins to the video dimensions
    x_bins = np.arange(0, video_dimensions[1] + pixels_per_bin, pixels_per_bin) #Divide video width in bins
    y_bins = np.arange(0, video_dimensions[0] + pixels_per_bin, pixels_per_bin)
    #Get all the possible combinations of x and y bins
    combinations = np.array(list(itertools.product(np.arange(x_bins.shape[0]-1), np.arange(y_bins.shape[0]-1)))) #Get all possible combinations that may occur
    
    #Allocate all the x and y positions to their corresponding bins
    what_bin = np.vstack((np.digitize(body_parts[:,part_id], x_bins, right = False), np.digitize(body_parts[:,part_id+1], y_bins, right = False))).T
    
    #Construct a video frames x total_number_of_bins matrix where for every frame
    #(row) all values are 0 except for the x and y bin combination where the
    #chest point is currently located.
    position_vect = np.zeros([body_parts.shape[0],combinations.shape[0]]) #A linearized version of bin occupancy at each time point
    for k in range(combinations.shape[0]):
        idx = np.where((what_bin == combinations[k,:]).all(1))[0]
        if idx.shape[0] > 0:
            position_vect[idx, k] = 1
            
#%%-------Align the neural signals and grab the corresponding video frames---
#         Video was acquired faster than imaging.
    
    if signal_type == 'S':
        signal = gaussian_filter1d(miniscope_data[signal_type].T,1, axis=0, mode='constant', cval=0, truncate=4)
        #Pad the first and last samples with zeros in this condition
    else:
        signal = miniscope_data[signal_type].T #Transpose the original signal so that rows are timepoints and columns cells
    
    #Determine which neurons to include in the analysis - keep all here
    keep_neuron = np.arange(signal.shape[1])
    
    #Align signal to the respsective state and retrieve the data
    Y = []
    x_analog = []
    part_likelihood = [] #Also extrac the likelihood estimate for the chest point at the extracted frames
   
    for k in range(len(aligned_to)):
        state_start_frame, state_time_covered = chiCa.find_state_start_frame_imaging(aligned_to[k], trialdata, miniscope_data['frame_interval'], miniscope_data['trial_starts'], miniscope_data['trial_start_time_covered'])                                                                     
        zero_frame = np.array(state_start_frame[val_trials[k]] + time_frame[k][0], dtype=int) #The firts frame to consider
        
        for add_to in np.arange(time_frame[k][1] - time_frame[k][0]): #Start looping through all the frames for that trial phase
            matching_frames = []
            for q in range(zero_frame.shape[0]): #unfortunately need to loop through the trials, should be improved in the future...
                    tmp = chiCa.match_video_to_imaging(np.array([zero_frame[q] + add_to]), miniscope_data['trial_starts'][val_trials[k][q]],
                           miniscope_data['frame_interval'], video_alignment['trial_starts'][val_trials[k][q]], video_alignment['frame_interval'])[0].astype(int)
                    matching_frames.append(tmp)
            
            Y.append(signal[zero_frame + add_to,:][:, keep_neuron])
            #Here the head orientation angles have been acquired with the imaging
            #and are aligned the same way the miniscope data are, the stimulus
            #and poke events have been mapped to the imaging in the previous steps
            #and can now be retrieve similarly, the body position and video / motion energy SVD
            #are still in video frames and are mapped to the imaging frames via
            #matching frames.
            x_analog.append(np.concatenate((stim_matrix[zero_frame + add_to,:], response_matrix[zero_frame + add_to,:], pitch[zero_frame + add_to,:], roll[zero_frame + add_to,:], yaw[zero_frame + add_to,:], position_vect[matching_frames,:], video_svd[matching_frames,:], me_svd[matching_frames,:]), axis=1))
            part_likelihood.append(part_likelihood_estimate[matching_frames,:])
            
    #Back to array like, where columns are trials, rows time points and sheets cells   
    Y = np.squeeze(Y)
    x_analog = np.squeeze(x_analog)
    part_likelihood = np.squeeze(part_likelihood)
    
    #Reshape the arrays to match the design matrix
    Y = np.reshape(Y, (x_include.shape[0], Y.shape[2]), order = 'F') #x_include already has the desired number of elements in its rows
    x_analog = np.reshape(x_analog, (x_include.shape[0], x_analog.shape[2]), order = 'F')
    
    #Add the analog regressors to the cognitive regressor design matrix
    X = np.hstack((x_include, x_analog))  
    
    #%%---------Construct the groupin of the regressors within the design matrix X---------------
    # Unfortunately this is really manual. Ideally this would be implemented when the individual 
    # regressors are being created...
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
            
    #Dictionary of regressor idx for different variales
    variable_idx = dict()
    for k in range(len(regressor_labels)):
        variable_idx[regressor_labels[k]] = regressor_idx[k]              
  
    #Update info about the regressor indices, always remember uppper exclusive
    reg_group_idx = [np.arange(regressor_idx[1][0], regressor_idx[5][0])] #The task regressors (choice, outcome, trial history + stimulus events) exclusive the intercept
    reg_group_idx.append(np.arange(regressor_idx[5][0],regressor_idx[8][0])) #Instructed actions (center and side poking)
    reg_group_idx.append(np.arange(regressor_idx[8][0],regressor_idx[-1][-1])) #Uninstructed movements (head orientation tuning, chest position, video, video ME)
     
    standardize_reg = np.arange(variable_idx['video_svd'][0], variable_idx['video_me_svd'][-1]+1) #Only z-score the truly anlog video variables
    #Video_svd is the first truly analog regressor, and video_me_svd is the last one of that kind.
    #Here, I decided not to standardize the dummy variables because this would lead
    #to a conflation of the events themselves with their frequency of occurence.
    #In case of very rare events this would lead to very high values for the times
    #the event actually occurred which would be more strongly penalized.
    #In the implementation here where the dummies remain 0 and 1 the interpretation
    #of the effect is frequency independent.
    
    #While we have tracked the visual stimuli as event based regressors they are
    #also visible on the video frames and redundant information about them could
    #be driving some of the video SVD and video ME SVD fluctuations. We therefore
    #orthogonolize these two regressors agains the visual events regressors using
    #Q decomposition.
    Q_vid = np.linalg.qr(np.hstack((X[:,variable_idx['stim_events']], X[:,variable_idx['video_svd']])))[0][:,variable_idx['stim_events'].shape[0]:]
    Q_me =np.linalg.qr(np.hstack((X[:,variable_idx['stim_events']], X[:,variable_idx['video_me_svd']])))[0][:,variable_idx['stim_events'].shape[0]:]
        
    X[:, variable_idx['video_svd']] = Q_vid #Reassemble the design matrix
    X[:, variable_idx['video_me_svd']] = Q_me
    
   #%%-------Draw the training and testing splits-------------------
    
    #First get the splits for training and testing sets. These will be constant
    #throughout the runs, so that every regressor (group) is fitted with the
    #same data. Here, we also use stratified k-folds to make sure that for training
    #and testing we homogeneously sample over all the different timepoints in
    #in the trial.
    kf = StratifiedKFold(n_splits = k_folds, shuffle = True)
    timepoint_labels = np.tile(np.arange(block.shape[0]),valid_trials.shape[0])
    k_fold_generator = kf.split(X, timepoint_labels) #This returns a generator object that spits out a different split at each call
    training = []
    testing = []
    for draw_num in range(k_folds):
        tr, te = k_fold_generator.__next__()
        training.append(tr)
        testing.append(te)
        
    #%%------Based on the specification at the top of the script, determine
    # which regressors will have to be shuffled-----------
    
    shuffle_regressor = [None] #Always fit the full model with no shuffling
    shuffle_regressor.append(np.arange(block.shape[0], X.shape[1])) #Add a null model
    #where there is only information about the trial time regressor, whose fluctuations
    #are identical for every trial.
    
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
    
    #Add a set number of complete shuffles, except for the intercept term
    for k in range(add_complete_shuffles):
        shuffle_regressor.append(np.arange(block.shape[0], X.shape[1]).tolist())
    
    #%%---------Finally, perform the model fitting------------------------------
    
    all_betas = []
    all_alphas = []
    all_rsquared = []
    all_corr = []
    all_r_timecourse = []
    all_corr_timecourse = []
    
    #Specific parameters for ridgeCV
    alpha_range = 10**np.linspace(-3,5,9) #Search the best ridge penalty over and exponentially grwoing series of values
    fit_intercept = False #We have explicitly added the time regressor as our intercept (but it will be penalized too...)
    alpha_per_target = True #Allow every neuron to get its own best penalty
   
    for s_round in range(len(shuffle_regressor)):    
        success = 0
        while success == 0: #Some shuffles lead to non-converging svd for the OLS solutions that will throw errors occasionally
            try:
                alphas, betas, r_squared, corr, y_test, y_hat = chiCa.fit_ridge_cv_shuffles(X, Y, alpha_range, alpha_per_target, fit_intercept, shuffle_regressor[s_round], standardize_reg, training, testing)    
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
    
