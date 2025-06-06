# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 15:34:09 2025

@author: Lukas Oesch
"""


from chiCa import *
import chiCa
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
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

#%%

# sessions = ['C:/data/LO028/20220616_145438',
#  'C:/data/LO032/20220830_121448',
#  'C:/data/LO032/20220905_123313',
#  'C:/data/LO032/20220907_150825',
#  'C:/data/LO032/20220909_144008',
#  'C:/data/LO032/20220923_135753',
#  'C:/data/LO051/20230322_161052',
#  'C:/data/LO051/20230329_161431',
#  'C:/data/LO051/20230427_163356',
#  'C:/data/LY008/20230405_172520',
#  #'C:/data/LY008/20230421_123107',
#  'C:/data/LY008/20230425_105313',
#  'C:/data/LO068/20230831_132717',
#  'C:/data/LO068/20230905_115256',
#  'C:/data/LO068/20230906_140350',
#  #'C:/data/LO068/20230911_154649',
#  'C:/data/LO069/20240102_125134']



# sessions_dec = [#'C:/data/LO028/20220616_145438', # This is actually an auditory session with low performance
#  #'C:/data/LO032/20220830_121448', #These LO032 sessions are also auditroy and low performance
#  #'C:/data/LO032/20220905_123313',
#  #'C:/data/LO032/20220907_150825',
#  'C:/data/LO032/20220909_144008',
#  'C:/data/LO032/20220923_135753',
#  'C:/data/LO051/20230322_161052',
#  'C:/data/LO051/20230329_161431',
#  'C:/data/LO051/20230427_163356',
#  'C:/data/LY008/20230405_172520',
#  #'C:/data/LY008/20230421_123107',
#  'C:/data/LY008/20230425_105313',
#  'C:/data/LO068/20230831_132717',
#  'C:/data/LO068/20230905_115256',
#  'C:/data/LO068/20230906_140350',
#  'C:/data/LO068/20230911_154649',
#  'C:/data/LO069/20240102_125134']


# sessions_enc = [#'C:/data/LO028/20220616_145438', # This is actually an auditory session with low performance
#  #'C:/data/LO032/20220830_121448', #These LO032 sessions are also auditroy and low performance
#  #'C:/data/LO032/20220905_123313',
#  #'C:/data/LO032/20220907_150825',
#  'C:/data/LO032/20220909_144008',
#  'C:/data/LO032/20220923_135753',
#  'C:/data/LO051/20230322_161052',
#  'C:/data/LO051/20230329_161431',
#  'C:/data/LO051/20230427_163356',
#  'C:/data/LY008/20230405_172520',
#  'C:/data/LY008/20230421_123107',
#  'C:/data/LY008/20230425_105313',
#  'C:/data/LO068/20230831_132717',
#  'C:/data/LO068/20230905_115256',
#  'C:/data/LO068/20230906_140350',
#  'C:/data/LO068/20230911_154649',
#  'C:/data/LO069/20240102_125134']



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
#---First load behavioral models
file_name = 'logreg_fig_one' #The name to the files for these models

logreg_weights_dec = []
history_strength_dec = []
for ses in sessions:
    choice_strategy_models = pd.read_hdf(os.path.join(ses,'analysis',file_name + '.hdf5'), '/Data')
    logreg_weights_dec.append(np.mean(choice_strategy_models['model_coefficients'],axis=0))
    history_strength_dec.append(np.sqrt(np.sum((logreg_weights_dec[-1][0,2:])**2)))

behavioral_weights_dec = np.squeeze(logreg_weights_dec)
history_strength_dec = np.squeeze(history_strength_dec)

#--Now load trial history decoders
use_name = 'Fig2_decoding_previous_choice_outcome_combination_spr.npy'
classes = 4
model_accuracy = []
label_wise_accuracy_list = []
confusion_matrices = []
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


#
logreg_weights_enc = []
history_strength_enc = []
for ses in sessions:
    choice_strategy_models = pd.read_hdf(os.path.join(ses,'analysis',file_name + '.hdf5'), '/Data')
    logreg_weights_enc.append(np.mean(choice_strategy_models['model_coefficients'],axis=0))
    history_strength_enc.append(np.sqrt(np.sum((logreg_weights_enc[-1][0,2:])**2)))

behavioral_weights_enc = np.squeeze(logreg_weights_enc)
history_strength_enc = np.squeeze(history_strength_enc)

#----Finally get the encoding model data
single_var = []
from_session = []
original_neuron_id = []
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

#Just do some input checkig here
unlikely_r = [] #Session id, shuffle round id
for k in range(len(single_var)):
    for n in range(len(single_var[k]['r_squared_timecourse'])):
        if np.max(single_var[k]['r_squared_timecourse'][n]) > 1:
            unlikely_r.append(np.array([k,n]))


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


#%%--------

#First identify the animals in the sessions
subj_ses = np.array([os.path.split(os.path.split(ses)[0])[1] for ses in sessions])
unique_subj = np.unique(subj_ses)

subj_code = np.zeros([len(subj_ses)])
for k in range(unique_subj.shape[0]):
    subj_code[np.where(subj_ses == unique_subj[k])[0]] = int(k)
subj_string = [unique_subj[int(x)] for x in subj_code]  
   
frame_rate = 30

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
#Now do all the correlations

# #decdoer accuracy vs strategy usage
# phase = ['Early ITI','Late ITI', 'Stimulus','Action']
# for k in range(len(phase)):
#     ax = plt.figure().add_subplot(111)
#     for subj in range(unique_subj.shape[0]):
#         ax.scatter(history_strength[subj_code==subj], np.mean(decoder_accuracy[:,idx_list[k]][subj_code==subj,:],axis=1))
#     ax.set_title(phase[k])
#     ax.set_xlabel('Behavioral history strength')
#     ax.set_ylabel('Neural histroy decoding')
#     ax.set_ylim([0, 1])
#     r, p = pearsonr(history_strength, np.mean(decoder_accuracy[:,idx_list[k]],axis=1))
#     ax.text(1, 0.5, f'r = {np.round(r,3)}, p = {np.round(p,3)}')
#     separate_axes(ax)
  
input_cols = ['#2c3b68', '#357097', '#639a9c', '#b3ae8f', '#ffc380', '#fbdc8c'] #The color progression for individual animals
col_map = matplotlib.colors.LinearSegmentedColormap.from_list("", input_cols)
cols = []
for k in range(unique_subj.shape[0]):
    cols.append(col_map(k / unique_subj.shape[0] + (1/unique_subj.shape[0])))

cols = ['#3f2951', '#272d68', '#315881', '#759f9a', '#b7cd89', '#c3bc69', '#cebb97', '#e7c388', '#fade89' ]

ax = plt.figure().add_subplot(111)
for subj in range(unique_subj.shape[0]):
    ax.scatter(history_strength_dec[subj_code==subj], np.mean(decoder_accuracy[subj_code==subj,:],axis=1), c = cols[subj], s = 65, edgecolor='k', linewidth = 0.5, label = unique_subj[subj])
ax.set_xlabel('Behavioral history strength')
ax.set_ylabel('Neural histroy decoding')
ax.set_ylim([0.2, 1])
r, p = pearsonr(history_strength_dec, np.mean(decoder_accuracy,axis=1))
ax.text(1.2, 0.4, f'r = {np.round(r,3)}, p = {np.round(p,3)}')
separate_axes(ax)
ax.legend()

history_cols = ['#606cbe', '#48a7b1', '#deb35e', '#de8468']
ax = plt.figure().add_subplot(111)
for k in range(label_wise_accuracy.shape[2]):
    ax.scatter(np.abs(behavioral_weights_dec[:,2+k]), np.mean(label_wise_accuracy[:,:,k],axis=1), s = 30, c = history_cols[k], edgecolor = 'k', linewidth=0.5)
    r, p = pearsonr(np.abs(behavioral_weights_dec[:,2+k]), np.mean(label_wise_accuracy[:,:,k],axis=1))
    print(f'Correlation is: {r}, with p-val: {p}')
tmp_hist_weights = np.abs(behavioral_weights_dec[:,2:].flatten())
tmp_dec_acc = np.mean(label_wise_accuracy,axis=1).flatten()
#r, p = pearsonr(tmp_hist_weights, tmp_dec_acc)
ax.text(1.2, 0.4, f'r = {np.round(r,3)}, p = {np.round(p,3)}')
ax.set_xlabel('Behavioral decoding weight')
ax.set_ylabel('Neural histroy decoding')
ax.set_ylim([0.2, 1])
separate_axes(ax)


#Do some additional lme testing for decoder weights because thez are multivariate
d_dict = dict()
d_dict['behavioral_weights'] = np.abs(behavioral_weights_dec[:,2:].T.flatten())
d_dict['label_wise_accuracy'] = np.mean(label_wise_accuracy,axis=1).T.flatten()
d_dict['subject'] = subj_string * 4
d_dict['trial_history'] = np.repeat(['Previous correct left', 'Previous incorrect left', 'Previous incorrect right', 'Previous correct right'],subj_code.shape[0])
df = pd.DataFrame(d_dict)
df.to_csv('/Users/loesch/Documents/Churchland_lab/Trial-history_manuscript/Figures/Fig4_panels/history_betas_vs_label_wise_accuracy.svg')

#%%

#First identify the animals in the sessions
subj_ses = np.array([os.path.split(os.path.split(ses)[0])[1] for ses in sessions])
unique_subj = np.unique(subj_ses)

subj_code = np.zeros([len(subj_ses)],dtype=int)
for k in range(unique_subj.shape[0]):
    subj_code[np.where(subj_ses == unique_subj[k])[0]] = int(k)


# #--Now the encoding models
# cvR = np.array([np.mean(mod['r_squared'][4] - mod['r_squared'][1]) for mod in single_var])
# ax = plt.figure().add_subplot(111)
# for subj in range(unique_subj.shape[0]):
#     ax.scatter(history_strength_enc[subj_code==subj], cvR[subj_code==subj], c = cols[subj], s = 60, edgecolor='k', linewidth = 0.5, label = unique_subj[subj])
# r, p = pearsonr(history_strength_enc, cvR)
# ax.text(1.2, 0.02, f'r = {np.round(r,3)}, p = {np.round(p,3)}')
# ax.set_xlabel('Behavioral history strength')
# ax.set_ylabel('Maximal explained neural variance')
# ax.set_ylim([-0.002,0.16])
# separate_axes(ax)
# ax.legend()

# #--Now the encoding models
# dR = np.array([np.mean(mod['r_squared'][0] - mod['r_squared'][17]) for mod in single_var])
# ax = plt.figure().add_subplot(111)
# for subj in range(unique_subj.shape[0]):
#     ax.scatter(history_strength_enc[subj_code==subj], dR[subj_code==subj], c = cols[subj], s = 60, edgecolor='k', linewidth = 0.5, label = unique_subj[subj])

# r, p = pearsonr(history_strength_enc, dR)
# ax.text(1.2, 0.01, f'r = {np.round(r,3)}, p = {np.round(p,3)}')
# ax.set_xlabel('Behavioral history strength')
# ax.set_ylabel('Unique neural variance')
# ax.set_ylim([-0.002,0.04])
# separate_axes(ax)
# ax.legend()
#---------------Alternate plotting

#--Now the encoding models
cvR = [mod['r_squared'][4] - mod['r_squared'][1] for mod in single_var]
ax = plt.figure().add_subplot(111)
for subj in range(unique_subj.shape[0]):
    av = np.array([np.mean(cvR[k]) for k in np.where(subj_code==subj)[0]])
    sem = np.array([np.std(cvR[k])/np.sqrt(cvR[k].shape[0]) for k in np.where(subj_code==subj)[0]])
    hist = np.array([history_strength_enc[k] for k in np.where(subj_code==subj)[0]])
    ax.errorbar(hist, av, yerr=sem, color = cols[subj], fmt='o', capsize=4, markersize=7.5, label=unique_subj[subj])

ax.set_xlabel('Behavioral history strength')
ax.set_ylabel('Maximal explained neural variance')
ax.set_ylim([0,0.25])
separate_axes(ax)
ax.legend()
r, p = pearsonr([np.mean(x) for x in cvR], history_strength_enc)


dR = [mod['r_squared'][0] - mod['r_squared'][17] for mod in single_var]
ax = plt.figure().add_subplot(111)
for subj in range(unique_subj.shape[0]):
    av = np.array([np.mean(dR[k]) for k in np.where(subj_code==subj)[0]])
    sem = np.array([np.std(dR[k])/np.sqrt(dR[k].shape[0]) for k in np.where(subj_code==subj)[0]])
    hist = np.array([history_strength_enc[k] for k in np.where(subj_code==subj)[0]])
    ax.errorbar(hist, av, yerr=sem, color = cols[subj], fmt='o', capsize=4, markersize=7.5,label=unique_subj[subj])

ax.set_xlabel('Behavioral history strength')
ax.set_ylabel('UNique explained neural variance')
ax.set_ylim([0,0.04])
#ax.set_yticks(np.arange(0,0.05,0.01))
separate_axes(ax)
r, p = pearsonr([np.mean(x) for x in dR], history_strength_enc)

#%%---Play



# #--Now the encoding models
# cvR = np.zeros([len(single_var)])
# for k in range(len(single_var)):
#     tmp = single_var[k]['r_squared'][4] - single_var[k]['r_squared'][1]
#     cvR[k] = np.mean(tmp[tmp>0.02])
    
# ax = plt.figure().add_subplot(111)
# for subj in range(unique_subj.shape[0]):
#     ax.scatter(history_strength_enc[subj_code==subj], cvR[subj_code==subj], c = cols[subj], s = 60, edgecolor='k', linewidth = 0.5, label = unique_subj[subj])

# r, p = pearsonr(history_strength_enc, cvR)
# ax.text(1.2, 0.1, f'r = {np.round(r,3)}, p = {np.round(p,3)}')
# ax.set_xlabel('Behavioral history strength')
# ax.set_ylabel('Unique neural variance')
# separate_axes(ax)
# ax.legend()



# #--Now the encoding models
# dR = np.zeros([len(single_var)])
# for k in range(len(single_var)):
#     tmp = single_var[k]['r_squared'][0] - single_var[k]['r_squared'][17]
#     dR[k] = np.mean(tmp[tmp>0.01])
    
# ax = plt.figure().add_subplot(111)
# for subj in range(unique_subj.shape[0]):
#     ax.scatter(history_strength_enc[subj_code==subj], dR[subj_code==subj], c = cols[subj], s = 60, edgecolor='k', linewidth = 0.5, label = unique_subj[subj])

# r, p = pearsonr(history_strength_enc, dR)
# ax.text(1.2, 0.03, f'r = {np.round(r,3)}, p = {np.round(p,3)}')
# ax.set_xlabel('Behavioral history strength')
# ax.set_ylabel('Unique neural variance')
# separate_axes(ax)
# ax.legend()



#%%----Plot dR and cvR as a funtion of anatomical position of th GRIN lens

#-----Supplementary figure 5
lens_center_image = ['lo0028_lens_site2_bright.png', 'lo32_3.tiff', 'lo051_4.tif', 'lo067_slide2_grinlense_new_20x.tif',
                       'LO068_slide1_20x_c4.tif',  'lo069_slide1_20x_c3.tif','lo073_slide1_20x_c5.tif',
                       'lo074_slide1_20x_c3.tif', 'ly008.tif']
lens_AP = [0.37, 1.21, 1.93, 1.69, 1.97, 1.77, 1.21, 1.69, 1.69,]
lens_DV = [-2,  -1.9, -1.9 ,-1.7, -1.5, -1.75, -1.7, -1.75,  -2]
lens_ML = [0.5, 0.5, 0.4, 0.5, 0.3, 0.5, 0.8, 0.2, 0.6]


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



#
sex = ['F', 'M', 'F', 'M', 'M', 'F', 'M', 'M', 'M' ]   #Sorted sexes of the subject
sex_expanded = [sex[x] for x in subj_code]


dec_acc = np.mean(decoder_accuracy,axis=1)

#Also retrieve the number of valid trials for each session to check if that might be the reason for the sex difference
n_valid = np.zeros([len(sessions)]) * np.nan
for k in range(len(sessions)):
    tmp = c_data = get_chipmunk_behavior(sessions[k])
    n_valid[k] = np.sum(tmp['valid_past'])

#Generate table for lme
d_dict = dict({'decoding_accuracy': dec_acc, 'cvR': [np.mean(x) for x in cvR], 'dR': [np.mean(x) for x in dR]})
d_dict['subject'] = subj_string
d_dict['sex'] = [sex[int(x)] for x in subj_code]
d_dict['lens_AP'] = [lens_AP[int(x)] for x in subj_code]
d_dict['lens_DV'] = [lens_DV[int(x)] for x in subj_code]
d_dict['lens_ML'] = [lens_DV[int(x)] for x in subj_code]
d_dict['valid_trials'] = n_valid
df = pd.DataFrame(d_dict)
df.to_csv('/Users/loesch/Documents/Churchland_lab/Trial-history_manuscript/Supplementary_figure_5_histology_function/lens_position_vs_neural_measures.csv')

#%%--------

av = []
sem = []
for subj in range(np.unique(subj_code).shape[0]):
    av.append(np.mean(dec_acc[subj_code == subj]))
    sem.append(np.std(dec_acc[subj_code == subj]) / np.sqrt(np.sum(subj_code == subj)))

ax = plt.figure().add_subplot(111)
#for subj in range(unique_subj.shape[0]):
scat  = ax.scatter(lens_AP, lens_DV, c = av, s = 60, edgecolor='k', linewidth = 0.5, label = unique_subj[subj], cmap=col_map)
ax.set_ylim([-2.5, 0])
ax.set_xlim([2.2, 0])
ax.set_title('Decoding accuracy')
plt.colorbar(scat)


tmp = np.array([np.mean(x) for x in cvR])
av = []
for subj in range(np.unique(subj_code).shape[0]):
    av.append(np.mean(tmp[subj_code == subj]))
ax = plt.figure().add_subplot(111)
#for subj in range(unique_subj.shape[0]):
scat  = ax.scatter(lens_AP, lens_DV, c = av, s = 60, edgecolor='k', linewidth = 0.5, label = unique_subj[subj], cmap=col_map)
ax.set_ylim([-2.5, 0])
ax.set_xlim([2.2, 0])
ax.set_title('Maximal explained variance')
plt.colorbar(scat)


tmp = np.array([np.mean(x) for x in dR])
av = []
for subj in range(np.unique(subj_code).shape[0]):
    av.append(np.mean(tmp[subj_code == subj]))
ax = plt.figure().add_subplot(111)
#for subj in range(unique_subj.shape[0]):
scat  = ax.scatter(lens_AP, lens_DV, c = av, s = 60, edgecolor='k', linewidth = 0.5, label = unique_subj[subj], cmap=col_map)
ax.set_ylim([-2.5, 0])
ax.set_xlim([2.2, 0])
ax.set_title('Unique variance')
plt.colorbar(scat)






#%%

cols = ['#3f2951', '#272d68', '#315881', '#759f9a', '#b7cd89', '#c3bc69', '#cebb97', '#e7c388', '#fade89' ]

sex = ['F', 'M', 'F', 'M', 'M', 'F', 'M', 'M', 'M' ]   #Sorted sexes of the subject
sex_bin = [x =='F' for x in sex]
symbol = ['o','^']

coordinates = [lens_AP, lens_DV, lens_ML] #Plot cvR and dR for the three coordinates
vari = [cvR , dR] # loop through cvR and dR
titles = ['AP', 'DV', 'ML']
var_title = ['cvR', 'dR']
lims = [[0, 0.25], [0, 0.04]]
# for curr_var in range(len(vari)):
#     for co in range(len(coordinates)):
#         ax = plt.figure().add_subplot(111)
#         for subj in range(unique_subj.shape[0]):
#             av = np.array([np.mean(vari[curr_var][k]) for k in np.where(subj_code==subj)[0]])
#             sem = np.array([np.std(vari[curr_var][k])/np.sqrt(vari[curr_var][k].shape[0]) for k in np.where(subj_code==subj)[0]])
#             loc = [coordinates[co][subj]] * np.sum(subj_code==subj)
            
#             if co == 1:
#                 ax.errorbar(av, loc, xerr=sem, color = cols[subj], fmt=symbol[sex_bin[subj]], capsize=4, markersize=7.5, label=unique_subj[subj])
#             else:
#                 ax.errorbar(loc, av, yerr=sem, color = cols[subj], fmt=symbol[sex_bin[subj]], capsize=4, markersize=7.5, label=unique_subj[subj])
#         separate_axes(ax)
#         ax.set_title(titles[co] + ' ' + var_title[curr_var])       
#         if co == 0:
#             ax.invert_xaxis()
out_dir = '/Users/loesch/Documents/Churchland_lab/Trial-history_manuscript/Supplementary_figure_5_histology_function'
for curr_var in range(len(vari)):
    for co in range(len(coordinates)):
        fi = plt.figure()
        ax = fi.add_subplot(111)
        for subj in range(unique_subj.shape[0]):
            means = np.array([np.mean(vari[curr_var][k]) for k in np.where(subj_code==subj)[0]])
            av = np.mean(means)
            sem = np.std(means)/np.sqrt(means.shape[0])
            loc = coordinates[co][subj]
            
            if co == 1:
                ax.errorbar(av, loc, xerr=sem, color = cols[subj], fmt=symbol[sex_bin[subj]], capsize=4, markersize=9, label=unique_subj[subj], markerfacecolor = 'w', elinewidth=1.5, capthick=1.5)
            else:
                ax.errorbar(loc, av, yerr=sem, color = cols[subj], fmt=symbol[sex_bin[subj]], capsize=4, markersize=9, label=unique_subj[subj], markerfacecolor = 'w', elinewidth=1.5, capthick=1.5)
        if co==1:
            ax.set_xlim(lims[curr_var])
        else:
            ax.set_ylim(lims[curr_var])
        separate_axes(ax)
        ax.set_title(titles[co] + ' ' + var_title[curr_var])       
        if co == 0:
            ax.set_xlim([0, ax.get_xlim()[1]])
            separate_axes(ax)
            #ax.invert_xaxis()
        fi.savefig(os.path.join(out_dir, titles[co] + '_for_' + var_title[curr_var] + '.svg'))



for co in range(len(coordinates)):
    ax = plt.figure().add_subplot(111)
    for subj in range(unique_subj.shape[0]):
        av = np.mean(dec_acc[subj_code == subj])
        sem = np.std(dec_acc[subj_code == subj]) / np.sqrt(np.sum(subj_code == subj))
        loc = coordinates[co][subj]
        
        if co == 1:
            ax.errorbar(av, loc, xerr= sem, color = cols[subj], fmt=symbol[sex_bin[subj]], capsize=4, markersize=9, label=unique_subj[subj], markerfacecolor = 'w', elinewidth=1.5, capthick=1.5)
            ax.set_xlim([0.2, 1])
        else:
            ax.errorbar(loc, av, yerr=sem, color = cols[subj], fmt=symbol[sex_bin[subj]], capsize=4, markersize=9, label=unique_subj[subj],  markerfacecolor = 'w', elinewidth=1.5, capthick=1.5)
            ax.set_ylim([0.18, 1])
    separate_axes(ax)
    ax.set_title(titles[co])       
    if co == 0:
        ax.set_xlim([0, ax.get_xlim()[1]])
        separate_axes(ax)
        #ax.invert_xaxis()
    
    
#%%----Now use different marker and color for different sex but drop subject info

s_cols = ['#3aa630', '#f3c81b']


out_dir = '/Users/loesch/Documents/Churchland_lab/Trial-history_manuscript/Supplementary_figure_5_histology_function'
for curr_var in range(len(vari)):
    for co in range(len(coordinates)):
        fi = plt.figure()
        ax = fi.add_subplot(111)
        for subj in range(unique_subj.shape[0]):
            means = np.array([np.mean(vari[curr_var][k]) for k in np.where(subj_code==subj)[0]])
            av = np.mean(means)
            sem = np.std(means)/np.sqrt(means.shape[0])
            loc = coordinates[co][subj]
            
          
            ax.errorbar(loc, av, yerr=sem, color = s_cols[sex_bin[subj]], fmt=symbol[sex_bin[subj]], capsize=4, markersize=9, label=unique_subj[subj],  markerfacecolor = 'w', elinewidth=1.5, capthick=1.5)
            ax.set_ylim(lims[curr_var])
        separate_axes(ax)
        ax.set_title(titles[co] + ' ' + var_title[curr_var])       
        if co == 0:
            ax.set_xlim([0, ax.get_xlim()[1]])
            separate_axes(ax)
            #ax.invert_xaxis()
        fi.savefig(os.path.join(out_dir, titles[co] + '_for_' + var_title[curr_var] + 'two_colors.svg'))



for co in range(len(coordinates)):
    ax = plt.figure().add_subplot(111)
    for subj in range(unique_subj.shape[0]):
        av = np.mean(dec_acc[subj_code == subj])
        sem = np.std(dec_acc[subj_code == subj]) / np.sqrt(np.sum(subj_code == subj))
        loc = coordinates[co][subj]
        
        
        ax.errorbar(loc, av, yerr=sem, color = s_cols[sex_bin[subj]], fmt=symbol[sex_bin[subj]], capsize=4, markersize=9, label=unique_subj[subj],  markerfacecolor = 'w', elinewidth=1.5, capthick=1.5)
        ax.set_ylim([0.18, 1])
    separate_axes(ax)
    ax.set_title(titles[co])       
    if co == 0:
        ax.set_xlim([0, ax.get_xlim()[1]])
        separate_axes(ax)
        #ax.invert_xaxis()
    


