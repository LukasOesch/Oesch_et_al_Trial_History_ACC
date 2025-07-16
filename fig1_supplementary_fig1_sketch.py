# -*- coding: utf-8 -*-
"""
Code to reproduce analyses and plots shown on Figure 1 and Supplementary Figure 1

Please make sure to run sequentially as some cells might depend on variables defined earlier.

Created on Fri Jan 17 12:32:43 2025

@author: Lukas Oesch
"""

from chiCa import * #Run inside chiCa path to directly import it. Otherwise use sys to add the path, see below.
import matplotlib.pyplot as plt
import matplotlib
plt.rcParams["font.family"] = "Arial" #Use arial font for plotting labels
import pandas as pd
from scipy.stats import pearsonr
from labdatatools import * #Make sure to have labdata-tools installed and the data directory set to where the data are stored
import os
from glob import glob
from scipy.ndimage import gaussian_filter1d
# Add the path to the location containing the code for psychometric fits - please adapt to your specific path
import sys
sys.path.append('/Users/loesch/Documents/Churchland_lab/fit_psychometric/fit_psychometric')
from analysis import *


# #%%---Find sessions that can be included into the analysis


# #Find some defining exclusion criteria to remove other animals from the analysis, for example:
# #Should have experienced all stimulus strengths, maybe also expert early withdrawal
# #rate should not be higher than 0.5 overall and average number of trials performed per session
# #should be higher than ~250?


# fetch_and_convert = False #Assumes the data were already downloaded before

# subjects = ['LO032', 'LO051', 'LY008', 'LO068', 'LO028', 'LO061', 'LO069', 'LO067', 'LO071', 'LO073', 'LO074']
# use_sessions = [np.hstack((np.arange(110), np.arange(245,246), np.arange(247,258,2), np.arange(260,261), np.arange(263,276,2))), #For LO032 the sessions are interleaved with observations, the mouse is learning the auditory
#                           np.arange(0, 158), #LO051
#                           np.hstack((np.arange(0,122), np.arange(123,162))), #LY008, All audio-visual sessions excluded, one sessions with missing data is also excluded
#                           np.hstack((np.arange(5,89), np.arange(90, 101))), #LO068, All audio-visual sessions excluded, exclude spotlight in the beginning
#                           np.hstack((np.arange(0,109), np.arange(160, 174), np.arange(175,176), np.arange(178, 182), np.arange(183, 206, 2))), #LO028, Exclude interleaved observer sessions
#                           np.hstack((np.arange(5, 117), np.arange(118,124))), #LO061, Exclude spotlight sessions in the beginning
#                           np.arange(4, 133), #LO069, Exclude spotlight sessions
#                           np.hstack((np.arange(5,53), np.arange(55,214))), #LO067, exclude spotlight
#                           np.arange(0, 68), #LO071, Exclude sessions with less than 1 s extended stim here
#                           np.arange(0, 69), #LO073, Exclclude sessions with less than 1 s extended stim time
#                           np.arange(0, 70) #LO074, Exclude sessions with less than 1 s extended stim time
#                                      ]

# all_animals_session_files = []
# for subj in range(len(subjects)):
#     ses = rclone_list_sessions(subjects[subj])
    
#     if fetch_and_convert:
#         session_matfiles = []
#         for curr_ses in ses[use_sessions[subj]]:
#             exec_str = 'labdata get ' + subjects[subj] + ' -s ' + curr_ses + ' -d chipmunk -i "*.mat"' # *.avi needs to be between double quotes!
#             subprocess.run(exec_str)
#             session_matfiles.append(glob(os.path.join('C:\\data', subjects[subj], curr_ses, 'chipmunk', '*.mat'))[0])
        
#         session_files = convert_specified_behavior_sessions(session_matfiles)
#     else:
#         session_files = []
#         for curr_ses in ses[use_sessions[subj]]:
#             session_files.append(glob(os.path.join('C:/data', subjects[subj], curr_ses, 'chipmunk', '*.h5'))[0])
    
#     all_animals_session_files.append(session_files)
    
# #%%---Function to extrac some (chipmunk) session metrics for specified subject

# def diagnose_performance(subject):
#     '''Get some performance metrics on the level of the subject and on individual
#     sessions to decide whether to keep this subject and include specific sessions.
    
#     ---------------------------------------------------------------------------'''
    
#     from chiCa import convert_specified_behavior_sessions
#     import pandas as pd
#     from labdatatools import get_labdata_preferences
#     import os
#     import subprocess
#     from glob import glob
#     from time import time
#     import numpy as np
    
#     datatype = 'chipmunk'
    
#     start = time()
#     #Get a list of all the chipmunk sessions for the subject
#     #res = subprocess.check_output(f'labdata sessions {subject} -f {datatype}').decode().split("\n") #List all chipmunk sessions for this subject
#     res = subprocess.check_output(['labdata', 'sessions', subject, '-f', datatype]).decode().split("\n")
#     ses_list = []
#     for x in res:
#         if len(x) > 0:
#             if x[0]==" " and x[1:2]!="\t":
#                 ses_list.append(x[1:])
    
#     #Go through the sessions, copy and convert if session is not dowloaded yet
#     #and extract some metrics
#     metrics = {'session': [],
#                'corrupted': [],
#                'modality': [],
#                'stim_strengths': [],
#                'revise_choice': [],
#                'wait_time': [],
#                'post_stim_delay': [],
#                'completed_trials': [],
#                'early_withdrawal_rate': [],
#                'performance_easy': [],
#                'extended_stim': []}
    
    
#     #Locate the data in folder
#     base_dir = get_labdata_preferences()['paths'][0]
#     for session in ses_list:
#         metrics['session'].append(session)
#         if len(glob(os.path.join(base_dir, subject, session, datatype, '*.mat'))) == 0: #Check if chipmunk file is there
#             subprocess.run(['labdata', 'get', subject, '-s', session, '-d', datatype, '-i', "*.mat"])
#         metrics['corrupted'].append(False) #Assume things are find with this mat file
#         if len(glob(os.path.join(base_dir, subject, session, datatype, '*.h5'))) == 0: #Check if converted h5 file is present
#             try:
#                 _ = convert_specified_behavior_sessions([glob(os.path.join(base_dir, subject, session, datatype, '*.mat'))[0]])
#             except:
#                 pass #Use pass here because if a matfile exists but there is some issue with the conversion it only throws a warning
#         if len(glob(os.path.join(base_dir, subject, session, datatype, '*.h5'))) == 0: #Check again whether a file exists now
#                 metrics['corrupted'][-1] = True
#                 metrics['modality'].append(None)
#                 metrics['stim_strengths'].append([None])
#                 metrics['revise_choice'].append(None)
#                 metrics['wait_time'].append(None)
#                 metrics['post_stim_delay'].append(None)
#                 metrics['completed_trials'].append(None)
#                 metrics['early_withdrawal_rate'].append(None)
#                 metrics['performance_easy'].append(None)
#                 metrics['extended_stim'].append(None)
#                 continue #Skip to the next iteration if the file is broken
        
#         #Extract some other metrics
#         trialdata = pd.read_hdf(glob(os.path.join(base_dir, subject, session, datatype, '*.h5'))[0])
#         metrics['revise_choice'].append(trialdata['revise_choice_flag'][0]) #Were animals allowed to change their mind?
        
#         mods = np.unique(trialdata['stimulus_modality'].tolist())
#         if mods.shape[0] == 1:
#             metrics['modality'].append(mods[0])
#         elif mods.shape[0] > 1:
#             metrics['modality'].append('mixed')
        
#         stim_rate = [x.shape[0] for x in trialdata['stimulus_event_timestamps']]
#         metrics['stim_strengths'].append(np.unique(stim_rate))
#         try:
#             metrics['wait_time'].append(np.mean(trialdata['waitTime'])) #This is the time animals are required to wait
#         except:
#             metrics['wait_time'].append(np.mean(trialdata['actual_wait_time']))
#         try:
#             metrics['post_stim_delay'].append(np.mean(trialdata['postStimDelay'])>0) #This is true when a random delay was added
#         except:
#             metrics['post_stim_delay'].append(False) #If it is not specified rely on the wait time criterion alone.
#         metrics['completed_trials'].append(np.sum(np.isnan(trialdata['response_side'])==0)) #Number of completed trials
#         metrics['early_withdrawal_rate'].append(np.sum(trialdata['outcome_record']==-1)/trialdata.shape[0]) #Early withdrawal rate
#         easy_str = [np.min(np.unique(stim_rate)), np.max(np.unique(stim_rate))]
#         consider = ((stim_rate == easy_str[0]) | (stim_rate == easy_str[1])) & (np.isnan(trialdata['response_side'])==0)
#         metrics['performance_easy'].append(np.mean(trialdata['outcome_record'][consider]))
#         try:
#             metrics['extended_stim'].append(np.mean(trialdata['ExtraStimulusDuration']))
#         except:
#             metrics['extended_stim'].append(0) #Be conservative here and assume no extended stim if it can't be found
#     session_metrics = pd.DataFrame(metrics)
#     print(f'Computed session metrics for {subject} in {time() - start} seconds.')
#     print('-------------------------------------------------------------------')
#     return session_metrics

#%%--Get the session metrics for a set (or all) the animals

#Here, hte function will query the google drive with all the stored data using 
#labdata-tools, download and convert the originally acquired .mat files for the 
#behavior and extract a set of performance and task parameters that will be used
#in the following to decide whether to include animals or not.
#
#Please be aware that this procedure is slow and might take several hours!
#It is advised to save the list of performance metrics 
#
#Alternatively, a file with the metrics is provided with this repo and can be 
#used to skip this step, see below

#-------This might take a little moment!--------
subjects = ['LO028', 'LO032', 'LO037', 'LO038', 'LO051', 'LO061', 'LO067',
            'LO068', 'LO069', 'LO071', 'LO073', 'LO074', 'LO090', 'LO091', 'LY007', 'LY008']

subject_metrics = []
for subj in subjects:
    print(f'Runing diagnostics on {subj}...')
    subject_metrics.append(diagnose_performance(subj))
#---Extract a list of the relevant sessions by first getting the metrics for
# all animals of interest and then filter their sessions 


#----Alternative: Load existing file with animal sessions--------
#A file with the subject metrics is provided within this repo (subject_metrics_dict) to allow users to
#skip over the diagnosis process and avoid using labdata-tools as long as the data
#are stored in the subject -> session -> datatype structure.

## Here's how:
#tmp = np.load('the path to my file', allow_pickle=True).tolist() # First load the file, it is a dictionary
#subject_metrics = [tmp[k] for k in tmp.keys()] #Just restructure dict keys into list entries


#%%---Apply the selection criteria
include_sessions = [None] * len(subjects)
for sub in range(len(subject_metrics)):
    session_metrics = subject_metrics[sub]
    #1) Animals needs to have experienced 4, 6, 8, 10, 14, 16, 18, 20 Hz stims during a visual session
    req = [4, 6, 8, 10, 14, 16, 18, 20]
    has_all_strengths = np.zeros([session_metrics.shape[0]])
    for k in range(session_metrics['stim_strengths'].shape[0]):
        has_all_strengths[k] = np.sum([x in session_metrics['stim_strengths'][k] for x in req]) == 8
    
    stim_req = np.sum(has_all_strengths & (session_metrics['modality']=='visual')) > 0
    
    #2) Animals need to have attained wait times of 1.1 s and larger
    wait_req = np.sum((session_metrics['wait_time'] >= 1.1)) > 1 #This is somewhat minimal because it might be another requirement later
    
    if stim_req and wait_req: #If this is fullfilled filter for sessions
        #a) At least two pairs of different difficulties
        has_two_strengths = np.zeros([session_metrics.shape[0]])
        for k in range(session_metrics['stim_strengths'].shape[0]):
            has_two_strengths[k] = np.sum([x in session_metrics['stim_strengths'][k] for x in req]) >= 4
        #b) Consider only visual sessions
        is_visual = session_metrics['modality'] == 'visual'
        #c) Minimal wait time of 1.1  or post stim delay
        long_wait = (session_metrics['wait_time'] >= 1.1) | (session_metrics['post_stim_delay'] > 0)
        #d) Only include sessions with >0.4s extended stim time
        has_extended = session_metrics['extended_stim'] >= 0.4
        #e) More than 100 completed trials per session
        trial_crit = session_metrics['completed_trials'] >= 100
        #f) The file must also be intact
        is_intact = session_metrics['corrupted']==False
        #g) Must at least once have had higher performance than 80%
        high_perf = np.sum(session_metrics['performance_easy']>0.8) > 0
        
        idx = np.where(has_two_strengths & is_visual & long_wait & has_extended & trial_crit & is_intact)[0]
        if (idx.shape[0] >= 5) & (np.mean(session_metrics['early_withdrawal_rate'][idx])<0.55): #Make sure to at least get 5 sessions for the analyses and that the early withdrawal rate is below 0.55 
            include_sessions[sub] = session_metrics['session'][idx].tolist()

#%%------Extrac session-wise choices, stim_strengths,etc...

# def get_chipmunk_behavior(data_source):
#     '''Retrieve animal behavior information including previous choices and
#     outcomes, stim strengths, easy stims, etc.
#     '''
    
#     import numpy as np
#     import pandas as pd
#     from glob import glob
#     import os
#     from chiCa import load_trialdata, determine_prior_variable
    
#     #Load the trialdata
#     if isinstance(data_source, str):
#         if len(glob(session_dir + '/chipmunk/*.h5')) == 1:
#             trialdata = pd.read_hdf(glob(session_dir + '/chipmunk/*.h5')[0], '/Data')
#         elif len(glob(session_dir + '/chipmunk/*.mat')) == 1:
#             trialdata = load_trialdata(glob(session_dir + '/chipmunk/*.mat')[0])
#         else:
#             raise RuntimeError(f"Can't find a behavioral file in {session_dir}")
#     elif isinstance(data_source, pd.DataFrame):
#         trialdata = data_source
     
#     #Get the data   
#     out_dict = dict()
#     out_dict['choice']  = np.array(trialdata['response_side'])
#     out_dict['category'] = np.array(trialdata['correct_side'])
#     out_dict['prior_choice'] =  determine_prior_variable(np.array(trialdata['response_side']), np.ones(len(trialdata)), 1, 'consecutive')
#     out_dict['prior_category'] =  determine_prior_variable(np.array(trialdata['correct_side']), np.ones(len(trialdata)), 1, 'consecutive')
    
#     out_dict['outcome'] = np.array(trialdata['response_side'] == trialdata['correct_side'], dtype=float) #Define as float so that nans (that are float) can be retained later
#     out_dict['outcome'][np.array(np.isnan(trialdata['response_side']))] = np.nan
#     out_dict['prior_outcome'] =  determine_prior_variable(out_dict['outcome'], np.ones(len(trialdata)), 1, 'consecutive')
       
#     modality = np.zeros([trialdata.shape[0]])
#     modality[trialdata['stimulus_modality'] == 'auditory'] = 1
#     modality[trialdata['stimulus_modality'] == 'audio-visual'] = 2
#     out_dict['modality'] = modality
#     out_dict['prior_modality'] = determine_prior_variable(modality, np.ones(len(trialdata)), 1, 'consecutive')
    
#     #Find stimulus strengths
#     tmp_stim_strengths = np.zeros([trialdata.shape[0]], dtype=int) #Filter to find easiest stim strengths
#     for k in range(trialdata.shape[0]):
#         tmp_stim_strengths[k] = trialdata['stimulus_event_timestamps'][k].shape[0]
#     out_dict['stim_strengths'] = tmp_stim_strengths
    
#     #Get category boundary to normalize the stim strengths
#     unique_freq = np.unique(tmp_stim_strengths)
#     category_boundary = (np.min(unique_freq) + np.max(unique_freq))/2
#     stim_strengths = (tmp_stim_strengths- category_boundary) / (np.max(unique_freq) - category_boundary)
#     if trialdata['stimulus_modality'][0] == 'auditory':
#         stim_strengths = stim_strengths * -1
#     out_dict['relative_stim_strengths'] = stim_strengths
#     out_dict['easy_stim'] = np.abs(stim_strengths) == 1 #Either extreme
    
#     out_dict['all_valid'] = np.isnan(out_dict['choice'])==0
#     out_dict['valid_past'] = (np.isnan(out_dict['choice'])==0) & (np.isnan(out_dict['prior_choice'])==0)
    
#     out_dict['session'] = [os.path.split(session_dir)[1]] * trialdata.shape[0]
#     return pd.DataFrame(out_dict)
    
#-----------------------------------------
#%%----Go through the list of useful sessions and get the data

subject_data = dict()
base_dir = get_labdata_preferences()['paths'][0] #Retrieve the base directory
## Alternatively set the base directory manualy:
#base_dir = 'my base directory'

for subj_idx in range(len(subjects)):
    if include_sessions[subj_idx] is not None:
        dat = pd.DataFrame()
        for ses in include_sessions[subj_idx]:
            tmp = get_chipmunk_behavior(os.path.join(base_dir, subjects[subj_idx], ses))
            dat = pd.concat([dat, tmp], axis=0)
        subject_data[subjects[subj_idx]] = dat
        
#%%---Fit overall psychometric curves on all animals with valid trials

##########--------Fig. 1c----------######################
psy_models = []
nx = np.linspace(4, 20, 100) #Specify the range of values for the curve reconstruction, 4 - 20 hz with 100 steps
curve = []
gray = '#858585'
dict_keys = subject_data.keys()
for subj in dict_keys:
    dat = subject_data[subj]
   
    ft = PsychometricRegression(dat['choice'][dat['all_valid']], exog = dat['stim_strengths'][dat['all_valid']])
    res = ft.fit(min_required_stim_values = 0, full_output=True) #Mode lfit with output reporting
    curve.append(cumulative_gaussian(res.params[0], res.params[1], res.params[2], res.params[3], nx)) #Store reconstruction
    psy_models.append(res) #Store psychometric parameters
    
#Now the plotting
all_curves = np.vstack(curve).T
fi = plt.figure(figsize= [3.8,4.8])
ax = fi.add_subplot(1,1,1)
ax.plot(nx, all_curves, color=gray, linewidth = 0.5)
ax.plot(nx, np.mean(all_curves,axis=1), color = 'k', linewidth=1.5)
ax.plot([4,20],[0.5,0.5], linewidth=0.5, color = 'k', linestyle='--') #horizontal line
ax.plot([12,12],[0,1], linewidth=0.5, color = 'k', linestyle='--') #horizontal line
ax.set_xticks([4,8,12,16,20])
ax.set_ylim([-0.03,1]) #Lower bound is just eyballed..
separate_axes(ax)
ax.set_xlabel('Stimulus rate (Hz)')
ax.set_ylabel('Fraction right side choices')  

#%%-----
# # #TODO: Put this function into chiCa


# #----Function for wilson score confidence interval----
# def calculateWilsonScoreInterval(data):
#     ''' lowerBound, upperBound = calculateWilsonScoreInerval(data);
    
#     % Calculate the Wilson score confidence interval for binomial
#     % distributions. This confidence intervall as asymmetric, bounded by 0 and
#     % 1 and "pulls" towards 0.5. Currently, this function only handles an alpha
#     % level of 0.05.
#     % Fromula was drawn from:
#     % https://www.ucl.ac.uk/english-usage/staff/sean/resources/binomialpoisson.pdf
#     %
#     % INPUT: -data: A vector of 0s and 1s containing the observed outcomes.
#     %               Nans can be tolerated.
#     %
#     % OUTPUTS: -lowerBound/upperBound: The lower and upper bound of the
#     %                                  confidence interval.
#     %
#     % LO, 5/4/2021
#     %
#     %--------------------------------------------------------------------------'''
#     import numpy as np
#     p = np.nanmean(data) #take the average fo all the specified values
#     n = np.sum(np.isnan(data)==0) #number of valid observations
#     z = 1.95996 #the critical value for an alpha level of 0.05
    
#     lowerBound = (p + z**2/(2*n) - z*np.sqrt(p*(1-p)/n + z**2/(4*n**2)))/(1 + z**2/n);
#     upperBound = (p + z**2/(2*n) + z*np.sqrt(p*(1-p)/n + z**2/(4*n**2)))/(1 + z**2/n);
    
#     return lowerBound, upperBound

#%%--- Fit trial history dependent psychometric curves for each animal,
# also show repsonse averages per stim strength with wilson score interval

#######-----Fig 1d, pick LO074-------##################

#Define a path to automatically save the plots - there will be one per animal
save_to = 'C:/Users/Lukas Oesch/Documents/ChurchlandLab/Trial-history_manuscript/Figures/Fig1d_plots'
cols = ['#606cbe', '#48a7b1', '#deb35e', '#de8468'] #Plotting color for different trial history contexts
#--order is previous correct left, prev incorrect left, prev incorrect right, correct right

psych_param = [[],[],[],[],[]]#Ugly!
nx = np.linspace(4, 20, 100)  #Specify the range of values for the curve reconstruction, 4 - 20 hz with 100 steps

dict_keys = subject_data.keys()
for subj in dict_keys:
    dat = subject_data[subj]

    plc = dat['valid_past'] & ((dat['prior_choice']==0) & (dat['prior_outcome']==1)) #Previous correct left
    pli = dat['valid_past'] & ((dat['prior_choice']==0) & (dat['prior_outcome']==0)) #Previous incorrect left
    pri = dat['valid_past'] & ((dat['prior_choice']==1) & (dat['prior_outcome']==0)) #Previous incorrect right
    prc = dat['valid_past'] & ((dat['prior_choice']==1) & (dat['prior_outcome']==1)) #Previous correct right
   
    conds = [plc, pli, pri, prc, dat['valid_past']] #All the different scenarios plus all the trials with a valid current and immediate past choice
    fi = plt.figure(figsize= [3.8,4.8])
    ax = fi.add_subplot(1,1,1)
    for k in range(len(conds)):
        #Fit the curve 
        ft = PsychometricRegression(dat['choice'][conds[k]], exog = dat['stim_strengths'][conds[k]])
        res = ft.fit(min_required_stim_values = 0, full_output=True)
        print(res.summary())
        psych_param[k].append(res.params)
        curve = cumulative_gaussian(res.params[0], res.params[1], res.params[2], res.params[3], nx)
        
        if k < 4: #This is for all the trial history contexts
            av = []
            lower_bound = []
            upper_bound = []
            rates = np.unique(dat['stim_strengths'][conds[k]])
            for n in rates:
                av.append(np.mean(dat['choice'][conds[k]][dat['stim_strengths'][conds[k]]==n]))
                l, h = calculateWilsonScoreInterval(dat['choice'][conds[k]][dat['stim_strengths'][conds[k]]==n])
                lower_bound.append(av[-1] - l)
                upper_bound.append(h- av[-1])
            
            #For some rare stim strengths the error estimate is singular and can become negative
            lower_bound = np.array(lower_bound)
            lower_bound[lower_bound < 0] = 0
            upper_bound = np.array(upper_bound)
            upper_bound[upper_bound < 0] = 0
            
            ax.plot(nx, curve, color = cols[k])
            ax.errorbar(rates, av, yerr=[lower_bound, upper_bound], color = cols[k], fmt='o', markerfacecolor = 'w', capsize=4)
        elif k==4: # This is for the overall curve
            ax.plot(nx, curve, color = 'k', linestyle='--')
   
    ax.plot([4,20],[0.5,0.5], linewidth=0.5, color = 'k', linestyle='--') #horizontal line
    ax.plot([12,12],[0,1], linewidth=0.5, color = 'k', linestyle='--') #horizontal line
    ax.set_xticks([4,8,12,16,20])
    ax.set_ylim([-0.03,1]) #Lower bound is just eyballed..
    separate_axes(ax)
    ax.set_xlabel('Stimulus rate (Hz)')
    ax.set_ylabel('Fraction right side choices')
    ax.set_title(subj)      
          
    fi.savefig(os.path.join(save_to, subj + '.svg'))

#%% Plot the psychometric parameter estimates for the different trial history contexts

#----------This is for supplementary figure 1a, b, c and d
psy_array = np.squeeze(psych_param).transpose(1,0,2) #Subjects x Trial history x parameter

#Jitter the dots representing individual subjects 
tmp = 0.5-np.random.rand(psy_array.shape[0])
jitter = np.sign(tmp) * tmp**2

titles = ['Sensory_bias',
          'Sensitivity',
          'Upper_lapse_rate',
          'Lower_lapse_rate']

for k in range(len(titles)):
    
    fi = plt.figure(figsize=[3.5, 4.8])
    ax = fi.add_subplot(111)
    for n in range(4): #the trial history
        av = np.mean(psy_array[:,n,k])
        sem = np.std(psy_array[:,n,k]) / np.sqrt(psy_array.shape[0])
        ax.scatter(n+0.85+jitter, psy_array[:,n,k], color = cols[n], s = 14)
        ax.errorbar(n+1.1, av, yerr=sem, color = cols[n], fmt='o', markerfacecolor = 'w', capsize=4)
        ax.set_title(titles[k])
        ax.set_xlim([0,5])
        separate_axes(ax)
        
#%%-------Arrange the data for lme_stats and save to .csv

save_to = '/Users/loesch/Documents/Churchland_lab/Trial-history_manuscript/Figures/Supp_fig1_panels_202050601'
file_names = ['perceptual_bias', 'sensitivity','upper_lapse_rate', 'lower_lapse_rate']
#psy_array dimensions are: Subjects x Trial history (plus one more condition whivch is all together) x parameter
for k in range(len(file_names)):
    param_estimate = np.reshape(psy_array[:,:4,k].T, (psy_array.shape[0]*4))
    subj_id = [x for x in subject_data.keys()] * 4 #The 4 are the different trial history conditions
    trial_history = ['Previous_correct_left'] * psy_array.shape[0] + ['Previous_incorrect_left'] * psy_array.shape[0] + ['Previous_incorrect_right'] * psy_array.shape[0] + ['Previous_correct_right'] * psy_array.shape[0] 
    out_d = {'param_estimate': param_estimate, 'subject': subj_id, 'trial_history': trial_history}
    out_f = pd.DataFrame(out_d)
    out_f.to_csv(os.path.join(save_to, file_names[k] + '_lme_table.csv'))
    
#%%--------Compute the behavioral models with the trial history terms-----

#Here we fit the animals' choices using a logistic regression of a bias term
# (actually trial history-free 12 Hz stimulus), the stimulus strength, and a 
# regressor for every trial history context. 

# Note: These fits from the logistic regression will be saved to the 'analysis'
# datatype folder within the current subject's session folder. Please make sure
# that an analysis folder exists beofre running this part


file_name = 'logreg_fig_one' #The name to the files for these models
fit_models = False #Set to True to fit and retain models, set to False if you want to load existing fits

model_params = {'penalty': 'l2', 'inverse_regularization_strength': 1, 'solver': 'liblinear', 'fit_intercept': False, 'normalize': False}
# Note that we use the same regularization strength for all the models and do not optimize this parameter.
# This is to make sure the weight magnitudes are similarly penalized between sessions and subjects and can be compared.
# Of further note: We add a column of 1s to the design matrix and hence set fit_intercept to false
subsampling_rounds = 100 #Number of repeated subsamplings from the majority class(es).
secondary_labels = None
k_folds = 10
subjects_logreg = dict()
insufficient_sessions = []
for subj in subject_data.keys():
    tmp_weights = []
    tmp_shuf_weights = []
    tmp_perf = []
    tmp_shuf_perf = []
    tmp_ses = []

    dat = subject_data[subj]
    sessions = np.unique(dat['session']).tolist()
    for ses in sessions:
        failed = False
        if fit_models:
            incl_idx = (dat['session'] == ses) & dat['valid_past']
            #Construct the design matrix in order: bias, relative stim strength, previous correct left, previous incorrect left, previous incorrect right, previous correct right
            design_matrix = np.vstack((np.ones(np.sum(incl_idx)), dat['relative_stim_strengths'][incl_idx],
                                       (dat['prior_choice'][incl_idx]==0) & (dat['prior_outcome'][incl_idx]==1),
                                       (dat['prior_choice'][incl_idx]==0) & (dat['prior_outcome'][incl_idx]==0),
                                       (dat['prior_choice'][incl_idx]==1) & (dat['prior_outcome'][incl_idx]==0),
                                       (dat['prior_choice'][incl_idx]==1) & (dat['prior_outcome'][incl_idx]==1))).T
            
            resp_matrix = np.array(dat['choice'][incl_idx]).reshape(-1,1)
            try: #Sometimes there are not enough trials to split...
                choice_strategy_models = balanced_logistic_model_training(design_matrix, resp_matrix, k_folds, subsampling_rounds, secondary_labels, model_params)
                if not os.path.isdir(os.path.join(base_dir,subj,ses,'analysis')):
                    os.mkdir(os.path.join(base_dir,subj,ses,'analysis'))
                choice_strategy_models.to_hdf(os.path.join(base_dir,subj,ses,'analysis',file_name + '.hdf5'), '/Data', complevel=9) #Use compression
            except:
                failed = True
                insufficient_sessions.append([subj,ses])
        else:
            try:
                choice_strategy_models = pd.read_hdf(os.path.join(base_dir,subj,ses,'analysis',file_name + '.hdf5'), '/Data')
            except:
                failed = True
                insufficient_sessions.append([subj,ses])
            
        if not failed:
            tmp_weights.append(np.mean(choice_strategy_models['model_coefficients'],axis=0))
            tmp_shuf_weights.append(np.mean(choice_strategy_models['shuffle_coefficients'],axis=0))          
            tmp_perf.append(np.mean(choice_strategy_models['model_accuracy'],axis=0))
            tmp_shuf_perf.append(np.mean(choice_strategy_models['shuffle_accuracy'],axis=0))
            tmp_ses.append(ses)
    
    tmp = {'logreg_weights': np.squeeze(tmp_weights),
           'shuffled_weights': np.squeeze(tmp_shuf_weights),
           'model_accuracy': np.squeeze(tmp_perf),
           'shuffled_accuracy': np.squeeze(tmp_shuf_perf),
           'session': tmp_ses}
    subjects_logreg[subj] = tmp
    
        
#%%-------Fit main effects models too and especially models with multiple trials back


#------------Supplementary figure 1 

file_names = ['main_effects_logreg_suppfig_one', 'main_effects_logreg_twoback_suppfig_one'] #The name to the files for these models
fit_models = False #Once they are computed they may also be loaded from their stored files

model_params = {'penalty': 'l2', 'inverse_regularization_strength': 1, 'solver': 'liblinear', 'fit_intercept': False, 'normalize': False}
subsampling_rounds = 100
secondary_labels = None
k_folds = 10

main_eff_logreg = [dict(),dict()]
insufficient_sessions = [[],[]]
for n in range(len(file_names)):
    for subj in subject_data.keys():
        tmp_weights = []
        tmp_shuf_weights = []
        tmp_perf = []
        tmp_shuf_perf = []
        tmp_ses = []
    
        dat = subject_data[subj]
        sessions = np.unique(dat['session']).tolist()
        for ses in sessions:
            failed = False
            if fit_models:
                if n == 0:
                    #Fit main effects model with one trial back first
                    incl_idx = (dat['session'] == ses) & dat['valid_past']
                    #Construct the design matrix in order: bias, relative stim strength, previous correct left, previous incorrect left, previous incorrect right, previous correct right
                    design_matrix = np.vstack((np.ones(np.sum(incl_idx)), dat['relative_stim_strengths'][incl_idx],
                                               dat['prior_choice'][incl_idx], dat['prior_category'][incl_idx], dat['prior_choice'][incl_idx] * dat['prior_category'][incl_idx])).T
                    
                    resp_matrix = np.array(dat['choice'][incl_idx])
                    try: #Sometimes there are not enough trials to split...
                        choice_strategy_models = balanced_logistic_model_training(design_matrix, resp_matrix, k_folds, subsampling_rounds, secondary_labels, model_params)
                        if not os.path.isdir(os.path.join(base_dir,subj,ses,'analysis')):
                            os.mkdir(os.path.join(base_dir,subj,ses,'analysis'))
                        choice_strategy_models.to_hdf(os.path.join(base_dir,subj,ses,'analysis', file_names[n] + '.hdf5'), '/Data', complevel=9) #Use compression
                    except:
                        failed = True
                        insufficient_sessions[n].append([subj,ses])
                elif n == 1:
                    #Now fit main effects model with one trial back and two trials back 
                    incl_idx = (dat['session'] == ses) & dat['valid_two_back']
                    #Construct the design matrix in order: bias, relative stim strength, previous correct left, previous incorrect left, previous incorrect right, previous correct right
                    design_matrix = np.vstack((np.ones(np.sum(incl_idx)), dat['relative_stim_strengths'][incl_idx],
                                               dat['prior_choice'][incl_idx], dat['prior_category'][incl_idx], dat['prior_choice'][incl_idx] * dat['prior_category'][incl_idx],
                                               dat['choice_two_back'][incl_idx], dat['category_two_back'][incl_idx], dat['choice_two_back'][incl_idx] * dat['category_two_back'][incl_idx])).T
                    
                    resp_matrix = np.array(dat['choice'][incl_idx])
                    try: #Sometimes there are not enough trials to split...
                        choice_strategy_models = balanced_logistic_model_training(design_matrix, resp_matrix, k_folds, subsampling_rounds, secondary_labels, model_params)
                        if not os.path.isdir(os.path.join(base_dir,subj,ses,'analysis')):
                            os.mkdir(os.path.join(base_dir,subj,ses,'analysis'))
                            choice_strategy_models.to_hdf(os.path.join(base_dir,subj,ses,'analysis',file_names[n] + '.hdf5'), '/Data', complevel=9) #Use compression
                    except:
                        failed = True
                        insufficient_sessions[n].append([subj,ses])
            else:
                try:
                    choice_strategy_models = pd.read_hdf(os.path.join(base_dir,subj,ses,'analysis',file_names[n] + '.hdf5'), '/Data')
                except:
                    failed = True
                    insufficient_sessions[n].append([subj,ses])
                
            if not failed:
                tmp_weights.append(np.mean(choice_strategy_models['model_coefficients'],axis=0))
                tmp_shuf_weights.append(np.mean(choice_strategy_models['shuffle_coefficients'],axis=0))          
                tmp_perf.append(np.mean(choice_strategy_models['model_accuracy'],axis=0))
                tmp_shuf_perf.append(np.mean(choice_strategy_models['shuffle_accuracy'],axis=0))
                tmp_ses.append(ses)
        
        tmp = {'logreg_weights': np.squeeze(tmp_weights),
               'shuffled_weights': np.squeeze(tmp_shuf_weights),
               'model_accuracy': np.squeeze(tmp_perf),
               'shuffled_accuracy': np.squeeze(tmp_shuf_perf),
               'session': tmp_ses}
        main_eff_logreg[n][subj] = tmp
    
#--------Do some ploting

#More for internal use plot the model performance
perf = []
perf.append([np.mean(subjects_logreg[subj]['model_accuracy']) for subj in subjects_logreg.keys()])
for k in main_eff_logreg:
    perf.append([np.mean(k[subj]['model_accuracy']) for subj in k.keys()])

perf = np.squeeze(perf).T
ax = plt.figure().add_subplot(111)
ax.plot(perf.T, linewidth=0.5, color = gray)
ax.plot(np.mean(perf,axis=0), color='k', marker='o')
ax.set_ylim([0.5,1])

#Plot the weights for two trials back
main_eff_weights = np.squeeze([np.mean(main_eff_logreg[1][subj]['logreg_weights'],axis=0) for subj in main_eff_logreg[1].keys()])
ax = plt.figure(figsize=[4.8,4.8]).add_subplot(111)
ax.plot(main_eff_weights.T, linewidth=0.5, color = gray)
av = np.mean(main_eff_weights,axis=0)
sem = np.std(main_eff_weights,axis=0) / np.sqrt(main_eff_weights.shape[0])
ax.errorbar(np.arange(main_eff_weights.shape[1]), av, yerr=sem,  fmt='o', capsize=4, color='k')
#ax.plot(np.mean(main_eff_weights,axis=0), color='k', marker='o')
ax.plot([0,7],[0,0], linewidth=0.3,linestyle ='--', color = 'gray')
separate_axes(ax)


#Make the data frame for stats
export_loc = '/Users/loesch/Documents/Churchland_lab/Trial-history_manuscript/Figures/Supp_fig1_panels_202050601'
tmp_dict = {'logreg_weights': [], 'abs_logreg_weights': [], 'regressor': [],'is_shuffle': [], 'subject':[]}
w_names = ['intercept', 'stimulus', 'previous_choice', 'previous_category', 'previous_interaction','choice_two_back', 'category_two_back', 'interaction_two_back']

for subj in main_eff_logreg[1].keys():
    dat = main_eff_logreg[1][subj]
    num = np.size(dat['logreg_weights'])
    tmp_dict['logreg_weights'] = tmp_dict['logreg_weights'] + dat['logreg_weights'].flatten().tolist() + dat['shuffled_weights'].flatten().tolist()
    tmp_dict['abs_logreg_weights'] = tmp_dict['abs_logreg_weights'] + np.abs(dat['logreg_weights'].flatten()).tolist() + np.abs(dat['shuffled_weights'].flatten()).tolist()
    tmp_dict['is_shuffle'] = tmp_dict['is_shuffle'] + num*['data'] + num*['shuffle'] 
    tmp_dict['regressor'] = tmp_dict['regressor'] + w_names * int((num/len(w_names))*2)
    tmp_dict['subject'] = tmp_dict['subject'] + [subj]*(num*2)
logreg_df = pd.DataFrame(tmp_dict)
logreg_df.to_csv(os.path.join(export_loc,'main_effects_logreg_weights_for_lme.csv'))


# #Plot the weight magnitudes for two trials back for model and shuffle
# main_eff_weight_mag = np.squeeze([np.mean(np.abs(k[subj]['logreg_weights']),axis=0) for subj in k.keys()])
# main_eff_shu_weight_mag = np.squeeze([np.mean(np.abs(k[subj]['shuffled_weights']),axis=0) for subj in k.keys()])
# tmp = [main_eff_shu_weight_mag, main_eff_weight_mag]
# cols = [gray, '#2986cc']
# ax = plt.figure().add_subplot(111)
# for k in range(len(tmp)):
#     av = np.mean(tmp[k],axis=0)
#     sem = np.std(tmp[k],axis=0) / np.sqrt(tmp[k].shape[0])
#     ax.fill_between(np.arange(tmp[k].shape[1]), av - sem, av + sem, color=cols[k], linewidth=None, alpha = 0.4)
#     ax.plot(av, color= cols[k], marker='o')
    
# ax.plot(main_eff_weights.T, linewidth=0.5, color = gray)
# ax.plot(np.mean(main_eff_weights,axis=0), color='k', marker='o')
# ax.plot([0,7],[0,0], linewidth=0.3,linestyle ='--', color = 'gray')



#%%-----Plot trial history weights for the chosen subject LO073-----##

###############----Figure 1e ---------#########################
save_to = '/Users/loesch/Documents/Churchland_lab/Trial-history_manuscript/Figures/Fig1_panels_20250601'
w_names = ['Bias', 'Stimuls strength', 'Previous correct left', 'Previous incorrect left', 'Previous incorrect right', 'Previous correct right']

fi = plt.figure(figsize= [4.8,4.8])
ax = fi.add_subplot(1,1,1)
ax.plot(w_names, subjects_logreg['LO074']['logreg_weights'].T,color=gray, linewidth=0.5)
av = np.mean(subjects_logreg['LO074']['logreg_weights'],axis=0)
sem = np.std(subjects_logreg['LO074']['logreg_weights'],axis=0) / np.sqrt(subjects_logreg['LO074']['logreg_weights'].shape[0])
ax.errorbar(w_names, av, yerr=sem, color = 'k', fmt='o', capsize=4, markerfacecolor='w')
ax.plot([0,5],[0,0],color='k', linewidth=0.5, linestyle='--')
separate_axes(ax)
ax.set_ylabel('Choice decoding Weight')
ax.set_xticklabels(w_names, rotation = 45)
fi.savefig(os.path.join(save_to, subj + 'all_sessinon_logreg_Fig1e.svg'))

#%%----Restructure data to pandas data frame with absolute weight

#abs_weight ~ regressor + is_shuffle + weight_category * is_shuffle (1|subject) #Maybe include sex at some point
export_loc = '/Users/loesch/Documents/Churchland_lab/Trial-history_manuscript/Figures/Fig1_panels_20250601'
tmp_dict = {'logreg_weights': [], 'abs_logreg_weights': [], 'regressor': [],'is_shuffle': [], 'subject':[]}

for subj in subjects_logreg.keys():
    dat = subjects_logreg[subj]
    num = np.size(dat['logreg_weights'])
    tmp_dict['logreg_weights'] = tmp_dict['logreg_weights'] + dat['logreg_weights'].flatten().tolist() + dat['shuffled_weights'].flatten().tolist()
    tmp_dict['abs_logreg_weights'] = tmp_dict['abs_logreg_weights'] + np.abs(dat['logreg_weights'].flatten()).tolist() + np.abs(dat['shuffled_weights'].flatten()).tolist()
    tmp_dict['is_shuffle'] = tmp_dict['is_shuffle'] + num*['data'] + num*['shuffle'] 
    tmp_dict['regressor'] = tmp_dict['regressor'] + w_names * int((num/6)*2)
    tmp_dict['subject'] = tmp_dict['subject'] + [subj]*(num*2)
logreg_df = pd.DataFrame(tmp_dict)
logreg_df.to_csv(os.path.join(export_loc,'logreg_weights_for_lme.csv'))

#%%----Plot boxplots of average logreg weights for each animal.
# abs_weights = logreg_df['logreg_weights'][logreg_df['is_shuffle']=='not']
# abs_weights = np.array(abs_weights).reshape(int(abs_weights.shape[0]/6),6)
# ax = plt.figure().add_subplot(111)
# fancy_boxplot(ax,abs_weights, None)
# plt.plot([1,2,3,4,5,6], abs_weights.T)

# signed_weights = logreg_df['raw_weight'][logreg_df['is_shuffle']=='not']
# signed_weights = np.array(signed_weights).reshape(int(signed_weights.shape[0]/6),6)
# ax = plt.figure().add_subplot(111)
# fancy_boxplot(ax,signed_weights,None)

# #Average for every mouse
# fi = plt.figure(figsize= [6.8,4.8])
# ax = fi.add_subplot(111)

w_names = ['Bias', 'Stimuls strength', 'Previous correct left', 'Previous incorrect left', 'Previous incorrect right', 'Previous correct right']
num_subj = len(subjects_logreg.keys())
shift = 0.5 / num_subj
average = []
average_shuffled = []
subjects = list(subjects_logreg.keys())
for k in range(len(subjects)):
    dat = subjects_logreg[subjects[k]]
    av = np.mean(dat['logreg_weights'],axis=0)
    sem = np.std(dat['logreg_weights'],axis=0) / np.sqrt(dat['logreg_weights'].shape[0])
    vect = np.arange(6) + (shift*k - 0.25)
    ax.errorbar(vect, av, yerr=sem, fmt='o', capsize=4)
    #ax.errorbar([1,2,3,4,5,6], av, yerr=sem, color = gray, fmt='o', capsize=4)
    #ax.scatter(w_names, av, c=gray, s=40, edgecolor='w')
    average.append(av)
    average_shuffled.append(np.mean(dat['shuffled_weights'],axis=0))
average = np.squeeze(average)
average_shuffled = np.squeeze(average_shuffled) 


fi = plt.figure(figsize=[5.5,4.8])
ax = fi.add_subplot(111)
widths = 0.6
fancy_boxplot(ax, average, None, widths = widths)
ax.plot([1,6],[0,0], linestyle='--', color=gray, linewidth=0.5)
ax.set_xticklabels(w_names,rotation = 45, ha="right")
jitter = (0.5 - np.random.rand(average.shape[0]))/(1/widths)
for k in range(average.shape[0]):
    ax.scatter(np.arange(1,7) + jitter[k], average[k,:], c=gray,s=14)
ax.set_ylim([-1, 2.5])
separate_axes(ax)


#Plot the variance of the parameter estimates
variance = []
variance_shuffled = []
subjects = list(subjects_logreg.keys())
for k in range(len(subjects)):
    dat = subjects_logreg[subjects[k]]
    variance.append(np.std(dat['logreg_weights'],axis=0)**2)
    variance_shuffled.append(np.std(dat['shuffled_weights'],axis=0)**2)
variance = np.squeeze(variance)
variance_shuffled = np.squeeze(variance_shuffled) 


fi = plt.figure(figsize=[4.8,4.8])
ax = fi.add_subplot(111)
widths = 0.6
fancy_boxplot(ax, variance, None, widths = widths)
# ax.axhline(0, linestyle='--', color=gray, linewidth=0.5)
ax.set_xticklabels(w_names,rotation = 45, ha="right")
#jitter = (0.5 - np.random.rand(average.shape[0]))/(1/widths)
for k in range(variance.shape[0]):
    ax.scatter(np.arange(1,7) + jitter[k], variance[k,:], c=gray,s=14)
ax.set_ylim([0, 0.5])
separate_axes(ax)

#Make another csv for the mixed-effects model
d_dict = dict()
d_dict['weight_variance'] = np.hstack((variance.T.flatten(), variance_shuffled.T.flatten()))
d_dict['regressor'] = np.repeat(w_names, variance.shape[0]).tolist() * 2
d_dict['condition'] = ['data'] * np.size(variance) + ['shuffle'] * np.size(variance)
d_dict['subject'] = list(subject_data.keys()) * (variance.shape[1] * 2)
df = pd.DataFrame(d_dict)
df.to_csv('/Users/loesch/Documents/Churchland_lab/Trial-history_manuscript/Figures/Supp_fig1_panels_202050601/logreg_weigth_variance.csv')


# fi = plt.figure()
# ax = fi.add_subplot(111)
# fancy_boxplot(ax, average, None, labels = None, x_label = None, y_label = None, widths = 0.25, positions=np.arange(average.shape[1])+0.8)
# fancy_boxplot(ax, average_shuffled, ['w']*average_shuffled.shape[1], labels = w_names, x_label = None, y_label = '\u03B2 weight magnitude', widths = 0.25, positions=np.arange(average.shape[1])+1.2)
# ax.set_xlim([0.5,average.shape[1]+0.5])
# ax.axhline(0, linestyle='--', color='k')
# separate_axes(ax)
# ax.set_xticklabels(w_names,rotation = 45, ha="right")


# jitter = (0.5 - np.random.rand(average.shape[0]))/2
# for k in range(average.shape[0]):
#     ax.scatter(np.arange(1,7) + jitter[k], average[k,:],c=gray,s=20)

#%%----Embed the model weights using umap and color code by subject


#----- Supplementary figure 1
import umap
import matplotlib.cm as cm
weights = [subjects_logreg[x]['logreg_weights'] for x in subjects_logreg.keys()]
reducer = umap.UMAP()
embedding = reducer.fit_transform(np.vstack(weights))

current_start = 0
ind = []
subj_idx = []
for k in range(len(weights)):
    stop = current_start + weights[k].shape[0]
    ind.append(np.arange(current_start,stop))
    current_start = stop
    subj_idx = subj_idx + list(np.zeros([ind[-1].shape[0]],dtype=int)+k)
ax = plt.figure(figsize=[4.8,4.8]).add_subplot(111)
# for k in ind:
#     plt.scatter(embedding[k,0], embedding[k,1], cmap=cm.Spectral)
ax.scatter(embedding[:,0], embedding[:,1], s=14, c = subj_idx, cmap=cm.Spectral)
separate_axes(ax)

#%%----Plot session performance against history strength---#####

#-----Figure 1H
save_to = 'C:/Users/Lukas Oesch/Documents/ChurchlandLab/Trial-history_manuscript/Figures'

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


#Extract aniaml performance, the strength of the trial history coefficients and te stim coefficients
session_performance = []
history_strength = []
stim_coef = []
tmp_subj = []
for subj in subject_data.keys():
    
    dat = subjects_logreg[subj]
    history_strength.append(np.sqrt(np.sum((dat['logreg_weights'][:,2:])**2,axis=1)))
    stim_coef.append(np.array(dat['logreg_weights'][:,1]))
    sessions = dat['session'] #Retrieve the sessions from the logreg beacause some failed to fit because of the lack of date for the corresponding splits
    tmp_subj = tmp_subj + [subj] * len(sessions)
    
    dat = subject_data[subj]
    tmp_perf = []
    for ses in sessions:
        tmp_perf.append(np.mean(dat['outcome'][dat['valid_past'] & (dat['session']==ses)]))
    session_performance.append(np.squeeze(tmp_perf))


#Prep data frame for LME
d_dict = dict()
d_dict['subject'] = tmp_subj
d_dict['performance'] = np.hstack(session_performance)
d_dict['hist_stim_coef_delta'] = np.hstack(history_strength) - np.hstack(stim_coef)
d_dict['history_strength'] = np.hstack(history_strength)
d_dict['stim_coef'] = np.hstack(stim_coef)
df = pd.DataFrame(d_dict)
df.to_csv('/Users/loesch/Documents/Churchland_lab/Trial-history_manuscript/Figures/Fig1_panels_20250601/hist_delta_vs_performance.csv')

# fi = plt.figure(figsize=[5.8,4.8])
# ax = fi.add_subplot(111)
# sc = ax.scatter(np.hstack(history_strength), np.hstack(session_performance), c=np.hstack(stim_coef), alpha=1, linewidth=0)
# #sc = ax.scatter(np.hstack(history_strength), np.hstack(session_performance), c=np.hstack(stim_coef), cmap = col_map, alpha=1, linewidth=0)
# cob = fi.colorbar(sc)
# ax.set_ylim([0.48, 1])
# ax.set_xlim([0, 2.653423907830812])
# separate_axes(ax)
# ax.set_xlabel('History strength')
# ax.set_ylabel('Performance')
# cob.set_label('Stimulus weight')

# corr, p_val = pearsonr(np.hstack(history_strength), np.hstack(session_performance))

# fi.savefig(os.path.join(save_to,'Fig1h_str_perf_corr.svg'))

# bin_id = np.digitize(np.hstack(history_strength), np.arange(0,np.max(np.hstack(history_strength)),0.1))
# maxima = np.squeeze([np.max(np.hstack(session_performance)[bin_id==x]) for x in np.unique(bin_id)])
# plt.figure()
# plt.plot(maxima)
# corr, p_val = pearsonr(np.arange(np.unique(bin_id).shape[0]), maxima)

#Reconstruct line from LME results run in R
lims = [np.min(df['hist_stim_coef_delta']), np.max(df['hist_stim_coef_delta'])]
lims = np.round(lims, decimals=1)
x_vect = np.arange(lims[0],lims[1],0.01)
regression_line = 0.744872 + (-0.055290 * x_vect)


gray = '#858585'
line_col = '#710909'
#Try plotting the difference between the stimulus weight and the trial history strength
fi = plt.figure(figsize=[4.8,4.8])
ax = fi.add_subplot(111)
sc = ax.scatter(np.hstack(history_strength) - np.hstack(stim_coef), np.hstack(session_performance), c=gray, edgecolor='w', linewidth=0.5)
ax.plot(x_vect, regression_line, color=line_col)
ax.set_ylim([0.5,1])
separate_axes(ax)
corr, p_val = pearsonr(np.hstack(history_strength) - np.hstack(stim_coef), np.hstack(session_performance))
