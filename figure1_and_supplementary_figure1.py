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
#A file with the subject metrics is provided within this repo (subject_metrics) to allow users to
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

#%%--- Fit trial history dependent psychometric curves for each animal,
# also show repsonse averages per stim strength with wilson score interval

#######-----Fig 1d, pick LO074-------##################

#Optional: define a path to automatically save the plots - there will be one per animal
#save_to = 'C:/Users/Lukas Oesch/Documents/ChurchlandLab/Trial-history_manuscript/Figures/Fig1d_plots'
save_to = None

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
    
    if save_to is not None:    
        fi.savefig(os.path.join(save_to, subj + '.svg'))

#%%------Plot the psychometric parameter estimates for the different trial history contexts

###############-------Supplementary figure 1a, b, c and d--------##############
psy_array = np.squeeze(psych_param).transpose(1,0,2) #Subjects x Trial history x parameter
#Please note that psy_array[:,4,:] represents the psychometric parameters fitted for all trials
#irrestpective of trial history.

#Jitter the dots representing individual subjects 
tmp = 0.5-np.random.rand(psy_array.shape[0])
jitter = np.sign(tmp) * tmp**2

titles = ['Sensory_bias',
          'Sensitivity',
          'Upper lapse rate',
          'Lower lapse rate']

x_tick_labels = ['Previous correct left', 'Previous incorrect left', 'Previous incorrect right', 'Previous correct right']

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
        ax.set_xticks(np.arange(1,5))
        ax.set_xticklabels(x_tick_labels, rotation=45, ha = 'right')
        ax.set_ylabel('Parameter value')
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

fit_models = False #Set to True to fit and retain models, set to False if you want to load existing fits


file_name = 'logreg_fig_one' #The name to the files for these models

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
    
        
#%%-------Fit main effects models andmodels with two trials  back

# Fit models where the different trial history regressors are not partitioned, that is,
# there is a previous choice regressor, a previous correct side regressor (WSLS) and one for 
# their interaction. Since previous right side choice and right side stimulus are modeled as ones
# the intercept in this model effectively reflectsthe animals biases after a previous correct left
# choice.
# The two trial back version of the model is an extended version of the main effects model also
# containing choice, correct side and their interaction form two trials back.
# Please note that due to early withdrawals the number of trials that have a consecutive history
# of valid trials decreases the farther in the past trials are included.

#######--------Supplementary figure 1g--------------##########################

fit_models = False #Once they are computed they may also be loaded from their stored files
 
file_names = ['main_effects_logreg_suppfig_one', 'main_effects_logreg_twoback_suppfig_one'] #The name to the files for these models

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
    
#%%---Plot the different regressor weights for the two-back logistic regression

#More for internal use: Compare the performance of the different models
perf = []
perf.append([np.mean(subjects_logreg[subj]['model_accuracy']) for subj in subjects_logreg.keys()])
for k in main_eff_logreg:
    perf.append([np.mean(k[subj]['model_accuracy']) for subj in k.keys()])

x_tick_labels = ['partitioned model', 'interaction model', 'two-back']
perf = np.squeeze(perf).T
ax = plt.figure().add_subplot(111)
ax.plot(perf.T, linewidth=0.5, color = gray)
ax.plot(np.mean(perf,axis=0), color='k', marker='o')
ax.set_xticks([0,1,2])
ax.set_xticklabels(x_tick_labels, rotation=45, ha='right')
ax.set_ylabel('Model accuracy')
ax.set_ylim([0.5,1])
separate_axes(ax)

######-----------Supplementary Figure 1g------------------####################

#Plot the weights for two trials back
w_names = ['intercept', 'stimulus', 'previous_choice', 'previous_category', 'previous_interaction','choice_two_back', 'category_two_back', 'interaction_two_back']
main_eff_weights = np.squeeze([np.mean(main_eff_logreg[1][subj]['logreg_weights'],axis=0) for subj in main_eff_logreg[1].keys()])
ax = plt.figure(figsize=[4.8,4.8]).add_subplot(111)
ax.plot(main_eff_weights.T, linewidth=0.5, color = gray)
av = np.mean(main_eff_weights,axis=0)
sem = np.std(main_eff_weights,axis=0) / np.sqrt(main_eff_weights.shape[0])
ax.errorbar(np.arange(main_eff_weights.shape[1]), av, yerr=sem,  fmt='o', capsize=4, color='k', markerfacecolor='w')
ax.plot([0,7],[0,0], linewidth=0.3,linestyle ='--', color = 'gray')
ax.set_xticks(np.arange(8))
ax.set_xticklabels(w_names, rotation=45, ha='right')
ax.set_ylabel('Choice decoding weight')
separate_axes(ax)

#Make the data frame for stats
export_loc = '/Users/loesch/Documents/Churchland_lab/Trial-history_manuscript/Figures/Supp_fig1_panels_202050601'
tmp_dict = {'logreg_weights': [], 'abs_logreg_weights': [], 'regressor': [],'is_shuffle': [], 'subject':[]}

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


#%%-----Plot trial history weights for the chosen subject LO073-----##

###############----Figure 1e ---------#########################
#save_to = '/Users/loesch/Documents/Churchland_lab/Trial-history_manuscript/Figures/Fig1_panels_20250601'
w_names = ['Bias', 'Stimuls strength', 'Previous correct left', 'Previous incorrect left', 'Previous incorrect right', 'Previous correct right']

fi = plt.figure(figsize= [4.8,4.8])
ax = fi.add_subplot(1,1,1)
ax.plot(w_names, subjects_logreg['LO074']['logreg_weights'].T,color=gray, linewidth=0.5)
av = np.mean(subjects_logreg['LO074']['logreg_weights'],axis=0)
sem = np.std(subjects_logreg['LO074']['logreg_weights'],axis=0) / np.sqrt(subjects_logreg['LO074']['logreg_weights'].shape[0])
ax.errorbar(w_names, av, yerr=sem, color = 'k', fmt='o', capsize=4, markerfacecolor='w')
ax.plot([0,5],[0,0],color='k', linewidth=0.5, linestyle='--')
separate_axes(ax)
ax.set_ylabel('Choice decoding weight')
ax.set_xticklabels(w_names, rotation = 45, ha = 'right')
#fi.savefig(os.path.join(save_to, subj + 'all_sessinon_logreg_Fig1e.svg'))

#%%----Restructure data to pandas data frame with absolute weight

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

######--------Figure 1f---------###############

w_names = ['Bias', 'Stimuls strength', 'Previous correct left', 'Previous incorrect left', 'Previous incorrect right', 'Previous correct right']
average = []
average_shuffled = []
subjects = list(subjects_logreg.keys())
for k in range(len(subjects)): #Get the average over sessions for each subject
    dat = subjects_logreg[subjects[k]]
    average.append(np.mean(dat['logreg_weights'],axis=0))
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
ax.set_ylabel('Choice decoding weight')
separate_axes(ax)

#%%Plot the variance of the parameter estimates

######----------Supplementary figure 1e-------------####################

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
ax.set_ylabel('Choice decoding weight variance')
separate_axes(ax)

#Make another csv for the mixed-effects model
d_dict = dict()
d_dict['weight_variance'] = np.hstack((variance.T.flatten(), variance_shuffled.T.flatten()))
d_dict['regressor'] = np.repeat(w_names, variance.shape[0]).tolist() * 2
d_dict['condition'] = ['data'] * np.size(variance) + ['shuffle'] * np.size(variance)
d_dict['subject'] = list(subject_data.keys()) * (variance.shape[1] * 2)
df = pd.DataFrame(d_dict)
df.to_csv('/Users/loesch/Documents/Churchland_lab/Trial-history_manuscript/Figures/Supp_fig1_panels_202050601/logreg_weigth_variance.csv')

#%%----Embed the model weights using umap and color code by subject

#############------Supplementary figure 1f-------------########################

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
ax.scatter(embedding[:,0], embedding[:,1], s=14, c = subj_idx, cmap=cm.Spectral)
separate_axes(ax)

#%%----Plot session performance against history strength---#####

#######################-----Figure 1h-------##################################
#save_to = 'C:/Users/Lukas Oesch/Documents/ChurchlandLab/Trial-history_manuscript/Figures'

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

#Extract aniaml performance, the strength of the trial history coefficients and the stim coefficients
session_performance = []
history_strength = []
stim_coef = []
tmp_subj = []
includes_difficult = []
for subj in subject_data.keys():
    
    dat = subjects_logreg[subj]
    history_strength.append(np.sqrt(np.sum((dat['logreg_weights'][:,2:])**2,axis=1)))
    stim_coef.append(np.array(dat['logreg_weights'][:,1]))
    sessions = dat['session'] #Retrieve the sessions from the logreg beacause some failed to fit because of the lack of date for the corresponding splits
    tmp_subj = tmp_subj + [subj] * len(sessions)
    
    dat = subject_data[subj]
    tmp_perf = []
    tmp_difficult = []
    for ses in sessions:
        tmp_perf.append(np.mean(dat['outcome'][dat['valid_past'] & (dat['session']==ses)]))
        subj_idx = [x for x in range(len(subject_metrics)) if subject_metrics[x]['subject_id'][0]==subj][0] #Arrgh, redefined what the subjects are in the middle of this script, very confusing...
        tmp_difficult.append(np.sum([(x >= 8) and (x <= 16) for x in subject_metrics[subj_idx]['stim_strengths'][subject_metrics[subj_idx]['session']==ses].tolist()[0]]) > 0)
    session_performance.append(np.squeeze(tmp_perf))
    includes_difficult.append(tmp_difficult)

#Prep data frame for LME
d_dict = dict()
d_dict['subject'] = tmp_subj
d_dict['performance'] = np.hstack(session_performance)
d_dict['hist_stim_coef_delta'] = np.hstack(history_strength) - np.hstack(stim_coef)
d_dict['history_strength'] = np.hstack(history_strength)
d_dict['stim_coef'] = np.hstack(stim_coef)
df = pd.DataFrame(d_dict)
df.to_csv('/Users/loesch/Documents/Churchland_lab/Trial-history_manuscript/Figures/Fig1_panels_20250601/hist_delta_vs_performance.csv')

#Reconstruct line from LME results run in R
lims = [np.min(df['hist_stim_coef_delta']), np.max(df['hist_stim_coef_delta'])]
lims = np.round(lims, decimals=1)
x_vect = np.arange(lims[0],lims[1],0.01)
regression_line = 0.744872 + (-0.055290 * x_vect)

#-----Grey scatter version
# gray = '#858585'
# line_col = '#710909'
# #Try plotting the difference between the stimulus weight and the trial history strength
# fi = plt.figure(figsize=[4.8,4.8])
# ax = fi.add_subplot(111)
# sc = ax.scatter(np.hstack(history_strength) - np.hstack(stim_coef), np.hstack(session_performance), c=gray, edgecolor='w', linewidth=0.5)
# ax.plot(x_vect, regression_line, color=line_col)
# ax.set_ylim([0.5,1])
# separate_axes(ax)
# corr, p_val = pearsonr(np.hstack(history_strength) - np.hstack(stim_coef), np.hstack(session_performance))

###########-----------Figure 1g-----------######################

gray = '#858585'
line_col = 'k'
subj_code = np.zeros([df.shape[0]]) * np.nan
for k in range(np.unique(df['subject']).shape[0]):
    subj_code[df['subject'] == np.unique(df['subject'])[k]] = k
tmp_colors = [cm.Spectral(x / (len(subject_data.keys()) - 1)) for x in range(len(subject_data.keys()))]
subj_colors = np.vstack([[tmp_colors[x]] * len(history_strength[x]) for x in range(len(history_strength))])

fi = plt.figure(figsize=[4.8,4.8])
ax = fi.add_subplot(111)
sc = ax.scatter(np.hstack(history_strength) - np.hstack(stim_coef), np.hstack(session_performance), c=subj_colors, cmap=cm.Spectral, edgecolor='w', linewidth=0.5)
ax.plot(x_vect, regression_line, color=line_col)
ax.set_ylim([0.5,1])
separate_axes(ax)
corr, p_val = pearsonr(np.hstack(history_strength) - np.hstack(stim_coef), np.hstack(session_performance))

######---------Reviewer Figure 3a, b, and c-----------################

#Reviewer comment: exclude high hist strength mouse
anim_idx = np.argmax([np.mean(x) for x in history_strength]) #The mouse is LY008 the last in the list
#Since the mouse is the last one we can simply index into the list
no_LY008 = subj_code!=anim_idx

#Use the regression coefficients from the lme to fit the line
lims = [np.min(df['hist_stim_coef_delta']), np.max(df['hist_stim_coef_delta'])]
lims = np.round(lims, decimals=1)
x_vect = np.arange(lims[0],lims[1],0.01)
regression_line = 0.734097 + (-0.065992 * x_vect)

#Plot the difference between the stimulus weight and the trial history strength
fi = plt.figure(figsize=[4.8,4.8])
ax = fi.add_subplot(111)
sc = ax.scatter(np.hstack(history_strength)[no_LY008] - np.hstack(stim_coef)[no_LY008], np.hstack(session_performance)[no_LY008], c=subj_colors[no_LY008], edgecolor='w', linewidth=0.5)
ax.plot(x_vect, regression_line, color=line_col)
ax.set_ylim([0.5,1])
separate_axes(ax)
corr, p_val = pearsonr(np.hstack(history_strength)[no_LY008] - np.hstack(stim_coef)[no_LY008], np.hstack(session_performance)[no_LY008])

#Now also look at sessions where the mice saw more stim strengths vs relatively easy only sessions
difficult_sessions = np.hstack(includes_difficult)
titles = ['Only easy', 'Includes difficult']
subj_code = np.zeros([df.shape[0]]) * np.nan
for k in range(np.unique(df['subject']).shape[0]):
    subj_code[df['subject'] == np.unique(df['subject'])[k]] = k

axes = []
for k in [0,1]:
    fi = plt.figure(figsize=[4.8,4.8])
    axes.append(fi.add_subplot(111))
    sc = axes[k].scatter(np.hstack(history_strength)[difficult_sessions==k] - np.hstack(stim_coef)[difficult_sessions==k], np.hstack(session_performance)[difficult_sessions==k], c=subj_colors[difficult_sessions==k], edgecolor='w', linewidth=0.5)
    axes[k].set_ylim([0.5,1])
    axes[k].set_title(titles[k])
    corr, p_val = pearsonr(np.hstack(history_strength)[difficult_sessions==k] - np.hstack(stim_coef)[difficult_sessions==k], np.hstack(session_performance)[difficult_sessions==k])
    print(f'{titles[k]} : r = {corr}, p = {p_val}')

x_min, x_max = np.min([x.get_xlim()[0] for x in axes]), np.max([x.get_xlim()[1] for x in axes])
[x.set_xlim([x_min, x_max]) for x in axes]
[separate_axes(x) for x in axes]


#%%----Get the general stats for these behavioral sessions for supplementary table 1

#######------------Supplementary Table 1-----------######################

#This is dirty: Redefine the subject list with the original animals, so you can use include_session
#and extract all the metrics from subject_metrics....
subjects = ['LO028', 'LO032', 'LO037', 'LO038', 'LO051', 'LO061', 'LO067',
            'LO068', 'LO069', 'LO071', 'LO073', 'LO074', 'LO090', 'LO091', 'LY007', 'LY008']

supp_table_one = {'subject': [],
          'stimulus_modality': [],
          'session_number': [],
          'number_with_full_stim': [],
          'performance_on_easy': [],
          'valid_trials': [],
          'early_withdrawal_rate': [],
          'no_choice_trials': [],
          'history_strength': []
          }
mod_ids = ['visual', 'auditory', 'multisensory'] #The corresponding codes are 0 = visual, 1 = auditory, 2 = multisensory
full_stim_set = [4, 6 ,8, 10, 14, 16, 18, 20] # These stims have to be present for the session to count as a full

for n in range(len(include_sessions)): #Loop through all the subjects
    if include_sessions[n] is not None:
        tmp_ses_full_stim = []
        tmp_perf = []
        tmp_val = []
        tmp_early = []
        tmp_no_choice = []
        tmp_hist_str = []
       
        for session in include_sessions[n]:
            
            # #First confirm that the modality is the same
            # tmp_modality.append(subject_metrics[n]['modality'][subject_metrics[n]['session']==session].tolist()[0])
            #Check for the full stim set in the list of ocurring stm
            tmp_ses_full_stim.append(np.sum([x not in subject_metrics[n]['stim_strengths'][subject_metrics[n]['session']==session].tolist()[0].tolist() for x in full_stim_set]) == 0)
            tmp_perf.append(subject_metrics[n]['performance_easy'][subject_metrics[n]['session']==session].tolist()[0])
            tmp_early.append(subject_metrics[n]['early_withdrawal_rate'][subject_metrics[n]['session']==session].tolist()[0])
            
            beh = get_chipmunk_behavior(os.path.join(base_dir,subjects[n], session))
            tmp_val.append(np.sum(beh['valid_past'])) #This is the number of trials we actually consider ofr the analyses
            tmp_no_choice.append(np.sum(beh['no_choice_trial']))
            tmp_hist_str.append(np.sqrt(np.sum((subjects_logreg[subjects[n]]['logreg_weights'][:,2:])**2,axis=1)))
            
        supp_table_one['subject'].append(subjects[n])
        supp_table_one['stimulus_modality'].append(subject_metrics[n]['modality'][subject_metrics[n]['session']==include_sessions[n][0]].tolist()[0])
        supp_table_one['session_number'].append(len(include_sessions[n]))
        supp_table_one['number_with_full_stim'].append(np.sum(tmp_ses_full_stim))
        supp_table_one['performance_on_easy'].append(np.mean(tmp_perf))
        supp_table_one['valid_trials'].append(np.mean(tmp_val))
        supp_table_one['early_withdrawal_rate'].append(np.mean(tmp_early))
        supp_table_one['no_choice_trials'].append(np.mean(tmp_no_choice))
        supp_table_one['history_strength'].append(np.mean(np.sqrt(np.sum((subjects_logreg[subjects[n]]['logreg_weights'][:,2:])**2,axis=1))))

supp_table_one = pd.DataFrame(supp_table_one)        
