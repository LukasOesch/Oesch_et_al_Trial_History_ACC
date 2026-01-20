# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 15:34:09 2025

@author: Lukas Oesch
"""

from chiCa import *
import chiCa
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from labdatatools import *
import os
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Arial"
import matplotlib

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

#%%-----Load the data from the behavioral models, and the neural encoding and decoding--------

#--Load logistic regression modeling animals' choices using stimulus and trial history
file_name = 'logreg_fig_one' #The name to the files for these models
logreg_weights = []
history_strength = []
for ses in sessions:
    choice_strategy_models = pd.read_hdf(os.path.join(ses,'analysis',file_name + '.hdf5'), '/Data')
    logreg_weights.append(np.mean(choice_strategy_models['model_coefficients'],axis=0))
    history_strength.append(np.sqrt(np.sum((logreg_weights[-1][0,2:])**2)))

behavioral_weights = np.squeeze(logreg_weights)
history_strength = np.squeeze(history_strength)

#--Load trial history decoding models
use_name = 'Fig2_decoding_previous_choice_outcome_combination_spr.npy'
classes = 4
model_accuracy = []
label_wise_accuracy_list = []
confusion_matrices = [] # Use the confusion matrix to obtain the label-wise accuracy initialized above
for session_dir in sessions:
    decoding_data = np.load(os.path.join(session_dir, 'analysis', use_name), allow_pickle=True).tolist()
    label_wise_accuracy = np.zeros([len(decoding_data['confusion_matrix']),classes]) *np.nan
    for k in range(len(decoding_data['confusion_matrix'])):
        for n in range(classes): #Get the main diagonal of the confusion matrix at the training index position
            label_wise_accuracy[k,n] = decoding_data['confusion_matrix'][k][n,n,k]
    label_wise_accuracy_list.append(label_wise_accuracy)
    
    #Overall accuray and shuffle plus model coefficients
    model_accuracy.append(np.squeeze([np.mean(x['model_accuracy'],axis=0) for x in decoding_data['decoding_models']]))
    confusion_matrices.append(np.squeeze(decoding_data['confusion_matrix']))
decoder_accuracy = np.squeeze(model_accuracy)
label_wise_accuracy = np.squeeze(label_wise_accuracy_list)

#----Get the encoding model data, this is a little bulky...
single_var = []
from_session = [] #Retain session during which the neuron was recorded
original_neuron_id = [] #The id of the neuron inside its session
encoding_model_name = 'RidgeCV_encoding_models_ITI_partitioned_Fig3.npy'
counter = 0
for session_dir in sessions:
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
for k in range(2, model_num + 2): #Model 0 is the full model, model 1 the time only model
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

#--Identify the sessions for each animal
subj_ses = np.array([os.path.split(os.path.split(ses)[0])[1] for ses in sessions])
unique_subj = np.unique(subj_ses)
subj_code = np.zeros([len(subj_ses)])
for k in range(unique_subj.shape[0]):
    subj_code[np.where(subj_ses == unique_subj[k])[0]] = int(k)
subj_string = [unique_subj[int(x)] for x in subj_code]  

#%%----Plot the correlations between the trial history strength and the neural
# decoding accuracy or the explained neural variance

#######--------Supplementary Figure 4a------------#####################

# Define the colors for individual mice
cols = ['#3f2951', '#272d68', '#315881', '#759f9a', '#b7cd89', '#c3bc69', '#cebb97', '#e7c388', '#fade89' ]
ax = plt.figure().add_subplot(111)
for subj in range(unique_subj.shape[0]):
    ax.scatter(history_strength[subj_code==subj], np.mean(decoder_accuracy[subj_code==subj,:],axis=1), c = cols[subj], s = 65, edgecolor='k', linewidth = 0.5, label = unique_subj[subj])
ax.set_xlabel('Behavioral history strength')
ax.set_ylabel('Trial history decoding accuracy')
ax.set_ylim([0.2, 1])
r, p = pearsonr(history_strength, np.mean(decoder_accuracy,axis=1))
ax.text(1.2, 0.4, f'r = {np.round(r,3)}, p = {np.round(p,3)}')
separate_axes(ax)
ax.legend()

#######--------Supplementary Figure 4b------------######################

history_cols = ['#606cbe', '#48a7b1', '#deb35e', '#de8468']
ax = plt.figure().add_subplot(111)
for k in range(label_wise_accuracy.shape[2]):
    ax.scatter(np.abs(behavioral_weights[:,2+k]), np.mean(label_wise_accuracy[:,:,k],axis=1), s = 30, c = history_cols[k], edgecolor = 'k', linewidth=0.5)
    r, p = pearsonr(np.abs(behavioral_weights[:,2+k]), np.mean(label_wise_accuracy[:,:,k],axis=1))
    ax.text(1.8, 0.6 - (k*0.05), f'r = {np.round(r,3)}, p = {np.round(p,3)}', color = history_cols[k])
    print(f'Correlation is: {r}, with p-val: {p}')
tmp_hist_weights = np.abs(behavioral_weights[:,2:].flatten())
tmp_dec_acc = np.mean(label_wise_accuracy,axis=1).flatten()
ax.set_xlabel('Behavioral decoding weight')
ax.set_ylabel('Trial history decoding accuracy')
ax.set_ylim([0.2, 1])
separate_axes(ax)
# Please note that these p values are not adjusted for the four comparisons performed

# Prepare a table with the class-wise decoding accuracy and the weight for each
# trial history context from the behavioral model
output_loc = '/Users/loesch/Documents/Churchland_lab/Trial-history_manuscript/Figures/Supplementary_figure4'
d_dict = dict()
d_dict['behavioral_weights'] = np.abs(behavioral_weights[:,2:].T.flatten())
d_dict['label_wise_accuracy'] = np.mean(label_wise_accuracy,axis=1).T.flatten()
d_dict['subject'] = subj_string * 4
d_dict['trial_history'] = np.repeat(['Previous correct left', 'Previous incorrect left', 'Previous incorrect right', 'Previous correct right'],subj_code.shape[0])
df = pd.DataFrame(d_dict)
df.to_csv(os.path.join(output_loc, 'history_betas_vs_label_wise_accuracy.csv'))

####-------------Supplementary Figure 4c--------------##################

cvR = [mod['r_squared'][4] - mod['r_squared'][1] for mod in single_var] #Index 4 are the trial history only models
ax = plt.figure().add_subplot(111)
for subj in range(unique_subj.shape[0]):
    av = np.array([np.mean(cvR[k]) for k in np.where(subj_code==subj)[0]])
    sem = np.array([np.std(cvR[k])/np.sqrt(cvR[k].shape[0]) for k in np.where(subj_code==subj)[0]])
    hist = np.array([history_strength[k] for k in np.where(subj_code==subj)[0]])
    ax.errorbar(hist, av, yerr=sem, color = cols[subj], fmt='o', capsize=4, markersize=7.5, label=unique_subj[subj])
ax.set_xlabel('Behavioral history strength')
ax.set_ylabel('Maximal variance explained by trial history')
ax.set_ylim([0,0.25])
separate_axes(ax)
r, p = pearsonr([np.mean(x) for x in cvR], history_strength)
ax.text(0.5, 0.02, f'r = {np.round(r,3)}, p = {np.round(p,3)}')
ax.legend(loc='lower left', bbox_to_anchor = [0.75, 0.6]) #bbox_to_anchor specifies the anchor point within the axes object

#######---------Supplementary Figure 4d--------------#####################

dR = [mod['r_squared'][0] - mod['r_squared'][17] for mod in single_var]
ax = plt.figure().add_subplot(111)
for subj in range(unique_subj.shape[0]):
    av = np.array([np.mean(dR[k]) for k in np.where(subj_code==subj)[0]])
    sem = np.array([np.std(dR[k])/np.sqrt(dR[k].shape[0]) for k in np.where(subj_code==subj)[0]])
    hist = np.array([history_strength[k] for k in np.where(subj_code==subj)[0]])
    ax.errorbar(hist, av, yerr=sem, color = cols[subj], fmt='o', capsize=4, markersize=7.5,label=unique_subj[subj])
ax.set_xlabel('Behavioral history strength')
ax.set_ylabel('Unique variance explained by trial history')
ax.set_ylim([0,0.04])
separate_axes(ax)
r, p = pearsonr([np.mean(x) for x in dR], history_strength)
ax.text(0.5, 0.004, f'r = {np.round(r,3)}, p = {np.round(p,3)}')
ax.legend(loc='lower left', bbox_to_anchor = [0.59, 0.01])

#%%----Plot dR and cvR as a funtion of anatomical position of th GRIN lens
 
#---Define the images that were used to approximate the lens position
lens_center_image = ['lo0028_lens_site2_bright.png',
                     'lo32_3.tiff',
                     'lo051_4.tif',
                     'lo067_slide2_grinlense_new_20x.tif',
                     'LO068_slide1_20x_c4.tif',
                     'lo069_slide1_20x_c3.tif',
                     'lo073_slide1_20x_c5.tif',
                     'lo074_slide1_20x_c3.tif',
                     'ly008.tif']

#----Enter the found coordinates manually
lens_AP = [0.37, 1.21, 1.93, 1.69, 1.97, 1.77, 1.21, 1.69, 1.69,]
lens_DV = [-2,  -1.9, -1.9 ,-1.7, -1.5, -1.75, -1.7, -1.75,  -2]
lens_ML = [0.5, 0.5, 0.4, 0.5, 0.3, 0.5, 0.8, 0.2, 0.6]

#---Associate the sex with the respective animals
sex = ['F', 'M', 'F', 'M', 'M', 'F', 'M', 'M', 'M' ]

dec_acc = np.mean(decoder_accuracy,axis=1) #Compute average decoder performance per session
# Also retrieve the number of valid trials for each session to check if that might be the reason for the sex difference
n_valid = np.zeros([len(sessions)]) * np.nan
for k in range(len(sessions)):
    tmp = c_data = get_chipmunk_behavior(sessions[k])
    n_valid[k] = np.sum(tmp['valid_past'])

#Generate table for lme
d_dict = dict({'decoding_accuracy': dec_acc, 'cvR': [np.mean(x) for x in cvR], 'dR': [np.mean(x) for x in dR]}) # Use the pre computed cvR and dR from above
d_dict['subject'] = subj_string
d_dict['sex'] = [sex[int(x)] for x in subj_code]
d_dict['lens_AP'] = [lens_AP[int(x)] for x in subj_code]
d_dict['lens_DV'] = [lens_DV[int(x)] for x in subj_code]
d_dict['lens_ML'] = [lens_DV[int(x)] for x in subj_code]
d_dict['valid_trials'] = n_valid
df = pd.DataFrame(d_dict)
output_loc = '/Users/loesch/Documents/Churchland_lab/Trial-history_manuscript/Figures/Supplementary_figure5'
df.to_csv(os.path.join(output_loc, 'lens_position_vs_neural_measures.csv'))

#%%---Optional: Scatter plot of lens coordinates (AP and DV) with scatters
# color-coded by the neural measures of trial history (decoding accuracy,
# maximal- and unique explained variance).

# # Define yellow-green-blue color map
# hex_cols = ['#ffffd6',
#  '#ecf8af',
#  '#ddf1b1',
#  '#c7e9b4',
#  '#7fccba',
#  '#44b6c5',
#  '#1b91c0',
#  '#1e77b3',
#  '#205fa7',
#  '#253393',
#  '#070d55']
# col_map = matplotlib.colors.LinearSegmentedColormap.from_list("", hex_cols)

# # Plot neural decoding as a function of AP and DV position
# av = []
# # sem = [] # sem is not used here
# for subj in range(np.unique(subj_code).shape[0]):
#     av.append(np.mean(dec_acc[subj_code == subj]))
#     #sem.append(np.std(dec_acc[subj_code == subj]) / np.sqrt(np.sum(subj_code == subj)))
# ax = plt.figure().add_subplot(111)
# #for subj in range(unique_subj.shape[0]):
# scat  = ax.scatter(lens_AP, lens_DV, c = av, s = 60, edgecolor='k', linewidth = 0.5, label = unique_subj[subj], cmap=col_map)
# ax.set_ylim([-2.5, 0])
# ax.set_xlim([2.2, 0])
# ax.set_xlabel('AP position')
# ax.set_ylabel('DV position')
# ax.set_title('Decoding accuracy by lens position')
# cbar = plt.colorbar(scat)
# cbar.set_label('Trial history decoding accuracy')

# # Plot plot maximal neural variance explained by trial history as a function of AP and DV position
# tmp = np.array([np.mean(x) for x in cvR])
# av = []
# for subj in range(np.unique(subj_code).shape[0]):
#     av.append(np.mean(tmp[subj_code == subj]))
# ax = plt.figure().add_subplot(111)
# scat  = ax.scatter(lens_AP, lens_DV, c = av, s = 60, edgecolor='k', linewidth = 0.5, label = unique_subj[subj], cmap=col_map)
# ax.set_ylim([-2.5, 0])
# ax.set_xlim([2.2, 0])
# ax.set_xlabel('AP position')
# ax.set_ylabel('DV position')
# ax.set_title('Maximal neural variance explained by trial history by lens position')
# cbar = plt.colorbar(scat)
# cbar.set_label('Maximal explained variance')


# tmp = np.array([np.mean(x) for x in dR])
# av = []
# for subj in range(np.unique(subj_code).shape[0]):
#     av.append(np.mean(tmp[subj_code == subj]))
# ax = plt.figure().add_subplot(111)
# scat  = ax.scatter(lens_AP, lens_DV, c = av, s = 60, edgecolor='k', linewidth = 0.5, label = unique_subj[subj], cmap=col_map)
# ax.set_ylim([-2.5, 0])
# ax.set_xlim([2.2, 0])
# ax.set_title('Unique neural variance explained by trial history by lens position')
# cbar = plt.colorbar(scat)
# cbar.set_label('Unique explained variance')

#%%----Plot average of neural measures against lens AP and DV position and 
# differentially represent different sexes.

symbol = ['o','^'] #Males are circles, females triangles
s_cols = ['#3aa630', '#f3c81b'] # Males are green, females yellow
coordinates = [lens_AP, lens_DV, lens_ML]
titles = ['AP', 'DV', 'ML'] # The coordinate directions
sex_bin = [x =='F' for x in sex] # Convert to boolean to durectly index into symbol and s_col

#####---------Supplementary Figure 5b and e---------#############

for co in range(len(coordinates)): # Plot the decoding accuracy for each cardinal direction
    ax = plt.figure().add_subplot(111)
    for subj in range(unique_subj.shape[0]):
        av = np.mean(dec_acc[subj_code == subj])
        sem = np.std(dec_acc[subj_code == subj]) / np.sqrt(np.sum(subj_code == subj))
        loc = coordinates[co][subj]
        ax.errorbar(loc, av, yerr=sem, color = s_cols[sex_bin[subj]], fmt=symbol[sex_bin[subj]], capsize=4, markersize=9, label=unique_subj[subj],  markerfacecolor = 'w', elinewidth=1.5, capthick=1.5)
        ax.set_ylim([0.18, 1])
        ax.set_xlabel(titles[co] + ' position')
        ax.set_ylabel('Decoding accuracy')
    separate_axes(ax)
    ax.set_title('Decoding accuracy by ' + titles[co] + ' position')       
    if co == 0: # Set minimum AP position to Bregma, this block was used earlier to flip the direction of the axis
        ax.set_xlim([0, ax.get_xlim()[1]])
        separate_axes(ax)


#######-------Supplementary Figure 5c and f, d and g--------################

vari = [cvR , dR] # loop through cvR and dR
var_title = ['Maximal', 'Unique']
lims = [[0, 0.25], [0, 0.04]] # Pre-define limits for cvR and dR
for curr_var in range(len(vari)): # Loop through the maximal and unique variances
    for co in range(len(coordinates)): #Loop through the directions 
        fi = plt.figure()
        ax = fi.add_subplot(111)
        for subj in range(unique_subj.shape[0]):
            means = np.array([np.mean(vari[curr_var][k]) for k in np.where(subj_code==subj)[0]])
            av = np.mean(means)
            sem = np.std(means)/np.sqrt(means.shape[0])
            loc = coordinates[co][subj]
            ax.errorbar(loc, av, yerr=sem, color = s_cols[sex_bin[subj]], fmt=symbol[sex_bin[subj]], capsize=4, markersize=9, label=unique_subj[subj],  markerfacecolor = 'w', elinewidth=1.5, capthick=1.5)
            ax.set_ylim(lims[curr_var])
            ax.set_xlabel(titles[co] + ' position')
            ax.set_ylabel(var_title[curr_var] + ' explained variance')
        separate_axes(ax)
        ax.set_title(var_title[curr_var] + ' explained neural variance by ' + titles[co] + ' position')       
        if co == 0:
            ax.set_xlim([0, ax.get_xlim()[1]])
            separate_axes(ax)
           

#%%-----Collect the data for supplementary table 2 

#Get a chronological sorting for the sessions within subjects
ordering = []
session_dates = np.array([os.path.split(x)[1] for x in sessions])
for k in np.unique(subj_code):
    target_ses = np.array(session_dates)[subj_code==k]
    for n in target_ses:
        ordering.append(np.where(session_dates == n)[0][0])

d_dict = {'subject': [],
          'session': [],
        'neuron_number': [],
          'stim_rates': [],
          'performance_easy': [],
          'valid_trials': [],
          'early_withdrawals': [],
          'no_choice': [],
          'hist_strength': []}

for k in ordering:
    d_dict['subject'].append(subj_string[k])
    d_dict['session'].append(session_dates[k])
    
    tmp = get_chipmunk_behavior(sessions[k])
    d_dict['stim_rates'].append(np.unique(tmp['stim_strengths']).tolist())
    d_dict['performance_easy'].append(np.mean(tmp['outcome'][tmp['easy_stim'] & tmp['valid_past']]))
    d_dict['valid_trials'].append(np.sum(tmp['valid_past']))
    d_dict['early_withdrawals'].append(np.mean(tmp['early_withdrawal']))
    d_dict['no_choice'].append(np.sum(tmp['no_choice_trial']))
    
    d_dict['hist_strength'].append(history_strength[k])
    
    enc = np.load(os.path.join(sessions[k], 'analysis', encoding_model_name),allow_pickle = True).tolist()
    d_dict['neuron_number'].append(enc['r_squared'][0].shape[0])
    
df = pd.DataFrame(d_dict)