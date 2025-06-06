# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 23:05:13 2025

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




# file_name ='encoding_models_ITI_partitioned_Fig3.npy'
# single_var = []
# for session_dir in sessions:
#     single_var.append(np.load(os.path.join(session_dir, 'analysis', file_name), allow_pickle=True).tolist())
    
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
     
#------------------Now the single variable models

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
# labels = list(single_var[0]['variable_index'].keys())
# labels.pop(0) #Remove the intercept weight
labels = ['Choice', 'Outcome', 'Previous choice x outcome', 'Stimulus events',
          'Center poke', 'Left poke', 'Right poke',
          'Pitch', 'Roll', 'Yaw', 'Chest position', 'Video SVD', 'Video motion energy SVD']

##################################################3
#%%----


# sort_idx = np.flip(np.argsort(full_single))

# colors = ['#444444', '#e26f2e', '#9f89dd', '#2a99cd','#5a8c45']
# data = [full_single] + cvR_single
# line_labels = ['full model'] + labels
# ax = plt.figure(figsize=[10, 4.8]).add_subplot()
# model_list = [0,1,3,4,12,13]
# for k in range(len(model_list)):
#     ax.plot(data[model_list[k]][sort_idx], linewidth=1, alpha=0.8, label = line_labels[model_list[k]])
# ax.set_ylim([-0.05, 1])
# ax.set_xlabel('Neuron number')
# ax.set_ylabel('cvR\u00b2')
# separate_axes(ax)
# ax.spines['left'].set_bounds([0,1])
# plt.legend()


# sort_idx = np.flip(np.argsort(full_single))

# colors = ['#e26f2e', '#9f89dd', '#2a99cd','#5a8c45']
# data = dR_single
# ax = plt.figure(figsize=[10, 4.8]).add_subplot()
# model_list = [0,2,3,11,12]
# for k in range(len(model_list)):
#     ax.plot(data[model_list[k]][sort_idx], linewidth=1, alpha=0.8, label = labels[model_list[k]])
# #ax.set_ylim([-0.05, 1])
# ax.set_xlabel('Neuron number')
# ax.set_ylabel('\u0394R\u00b2')
# separate_axes(ax)
# #ax.spines['left'].set_bounds([0,1])
# plt.legend()


# #Boxplots ofcvR2 and dR2
# colors = ['w'] * model_num
# ax = plt.figure(figsize=[10, 4.8]).add_subplot()
# fancy_boxplot(ax, cvR_single, colors, labels = labels, y_label = 'cvR\u00b2', widths = 0.7) 
# lims = ax.get_ylim()
# ax.set_ylim([lims[0], 0.4])
# ax.set_yticks([0,0.1,0.2,0.3,0.4])
# separate_axes(ax)
# ax.plot([0.5, len(cvR_single)+0.5],[0,0], color='k', linestyle='--', linewidth = ax.spines['top'].get_linewidth())
# ax.spines['bottom'].set_bounds([0.5, len(cvR_single)+0.5])


# ax = plt.figure(figsize=[10, 4.8]).add_subplot()
# fancy_boxplot(ax, dR_single, colors, labels = labels, y_label = '\u0394R\u00b2', widths = 0.7) 
# #ax.set_ylim([-0.05, 0.28])
# separate_axes(ax)
# ax.plot([0.5, len(cvR_single)+0.5],[0,0], color='k', linestyle='--', linewidth = ax.spines['top'].get_linewidth())
# ax.spines['bottom'].set_bounds([0.5, len(dR_single)+0.5])

# #Try only coloring the outlines
# outline_cols =  ['#303961'] * 4 +  ['#255753'] * 3 +  ['#265c2b'] * 6
# box_cols = model_num * ['w']
# ax = plt.figure(figsize=[10, 4.8]).add_subplot()
# positions = None
# widths = 0.7
# medianprops = dict({'color': 'k'})
# meanprops = dict(marker='o', markeredgecolor='black', markerfacecolor='white')
# bp = ax.boxplot(dR_single, positions=positions, labels=labels, widths = widths,
#                 showmeans=True, showfliers = False, patch_artist=True, medianprops = medianprops, meanprops = meanprops)
# import matplotlib
# for k in range(len(box_colors)):
#     # bp['boxes'][k].set_alpha(None)
#     # bp['boxes'][k].set_edgecolor(outline_cols[k])
#     # bp['boxes'][k].set_linewidth(2)
#     #bp['boxes'][k].set_facecolor(matplotlib.colors.to_rgba(outline_cols[k], alpha=0.5))
#     bp['boxes'][k].set_facecolor('w')
# ax.set_xlabel(x_label)
# ax.set_ylabel(y_label)
# if labels is not None:
#     axis.set_xticks(np.arange(1,len(labels)+1))
#     axis.set_xticklabels(labels)
#     axis.set_xlim(0.25, len(labels) + 0.75)


# #Plot unique variance for each neuron sorted by trial history
# sort_idx = np.flip(np.argsort(dR_single[2]))

# colors = ['#e26f2e', '#9f89dd', '#2a99cd','#5a8c45']
# data = dR_single
# ax = plt.figure(figsize=[10, 4.8]).add_subplot()
# model_list = [2,0,10,11]
# for k in range(len(model_list)):
#     ax.plot(data[model_list[k]][sort_idx], linewidth=1, alpha=0.8, label = labels[model_list[k]])
# #ax.set_ylim([-0.05, 1])
# ax.set_xlabel('Neuron number')
# ax.set_ylabel('\u0394R\u00b2')
# separate_axes(ax)
# #ax.spines['left'].set_bounds([0,1])
# plt.legend()


#Do the same averageing within each animal
#find sessions from a single subject
subj = [os.path.split(os.path.split(session_dir)[0])[1] for session_dir in sessions]
uni = np.unique(subj)
subj_ses = []
for k in uni:
    subj_ses.append(np.array([x for x in range(len(subj)) if subj[x] == k]))

#Calculate mean ans sem mdodel accuracy
ses_av = np.array([np.mean(x['r_squared'][0]) for x in single_var])
subj_av = np.array([np.mean(ses_av[subj_code==x]) for x in np.unique(subj_code)])
subj_sem = np.std(subj_av) / np.sqrt(np.unique(subj_code).shape[0])
print(f'Mean +/- sem full model variance is: {np.mean(subj_av)} +/- {subj_sem}')

mean_cvR = []
mean_dR = []
for k in range(len(uni)):
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
        
        
#light_gray = '#c8c8c8'
gray = '#858585' 
ax = plt.figure(figsize=[10, 4.8]).add_subplot()
widths = 0.7
fancy_boxplot(ax, np.squeeze(mean_cvR), None, widths = widths, labels = labels, y_label = 'cvR\u00b2')
ax.axhline(0, linestyle='--', color='k', linewidth=0.5)
jitter = (0.5 - np.random.rand(len(mean_cvR))) * (widths*0.8)
for k in range(len(mean_cvR)):
    ax.scatter(np.arange(1, model_num + 1) + jitter[k], np.squeeze(mean_cvR)[k,:],c=gray,s=10)
separate_axes(ax)
ax.spines['bottom'].set_bounds([0.5, len(cvR_single)+0.5])

ax = plt.figure(figsize=[10, 4.8]).add_subplot()
widths = 0.7
fancy_boxplot(ax, np.squeeze(mean_dR), None, widths = widths, labels = labels, y_label = 'cvR\u00b2')
ax.axhline(0, linestyle='--', color='k', linewidth=0.5)
jitter = (0.5 - np.random.rand(len(mean_cvR))) * (widths*0.8)
for k in range(len(mean_dR)):
    ax.scatter(np.arange(1, model_num + 1) + jitter[k], np.squeeze(mean_dR)[k,:],c=gray,s=10)
separate_axes(ax)
ax.spines['bottom'].set_bounds([0.5, len(cvR_single)+0.5])


#Boxplots ofcvR2 and dR2
colors = ['w'] * model_num
ax = plt.figure(figsize=[10, 4.8]).add_subplot()
fancy_boxplot(ax, np.squeeze(mean_cvR), colors, labels = labels, y_label = 'cvR\u00b2', widths = 0.7) 
separate_axes(ax)
ax.plot([0.5, len(cvR_single)+0.5],[0,0], color='k', linestyle='--', linewidth = ax.spines['top'].get_linewidth())
ax.spines['bottom'].set_bounds([0.5, len(cvR_single)+0.5])

colors = ['w'] * model_num
ax = plt.figure(figsize=[10, 4.8]).add_subplot()
fancy_boxplot(ax, np.squeeze(mean_dR), colors, labels = labels, y_label = '\u0394R\u00b2', widths = 0.7) 
separate_axes(ax)
ax.plot([0.5, len(cvR_single)+0.5],[0,0], color='k', linestyle='--', linewidth = ax.spines['top'].get_linewidth())
ax.spines['bottom'].set_bounds([0.5, len(cvR_single)+0.5])


#-------

time_cvR = []
time_dR = []
for k in range(len(uni)):
    tmp = []
    tmp_d = []
    for n in subj_ses[k]:
        tmp.append(np.mean(np.squeeze(cvR_timecourse_single)[:,:,from_session == n],axis=-1))
        tmp_d.append(np.mean(np.squeeze(dR_timecourse_single)[:,:,from_session == n],axis=-1))
    if len(tmp) > 1:
        time_cvR.append(np.mean(np.squeeze(tmp),axis=0))
        time_dR.append(np.mean(np.squeeze(tmp_d),axis=0))
    else:
        time_cvR.append(np.squeeze(tmp))
        time_dR.append(np.squeeze(tmp_d))
        
plot_models = [2,11,10,0]
cols = ['#765e9c', '#629bc6',  '#98446f', '#bc6238']
line_labels = ['trial history', 'video svd', 'chest position', 'choice']
for dat in [time_cvR, time_dR]:
    data = [np.squeeze(dat)[:,k,:].T for k in plot_models]
    ax = plt.figure(figsize=[10, 4.8]).add_subplot()
    plot_timecourse(ax, data, frame_rate, idx_list, spacer=4, colors=cols, line_labels = line_labels, x_label = 'Time (s)', y_label = 'cvR\u00b2')
    separate_axes(ax)

#%%-----Plot 
#--------------
ses_idx = [0,1,7,9,14,17,18,20,22] #Include the session with most recorded neurons for each mouse
use_neuron = np.array([x for x in range(from_session.shape[0]) if from_session[x] in ses_idx])

sort_idx = np.flip(np.argsort(full_single[use_neuron]))

colors = ['#e26f2e', '#9f89dd', '#2a99cd','#5a8c45']
data = dR_single
ax = plt.figure(figsize=[10, 4.8]).add_subplot()
model_list = [2,0,10,11]
for k in range(len(model_list)):
    ax.plot(data[model_list[k]][use_neuron][sort_idx], linewidth=1, alpha=0.8, label = labels[model_list[k]])
#ax.set_ylim([-0.05, 1])
ax.set_xlabel('Neuron number')
ax.set_ylabel('cvR\u00b2')
separate_axes(ax)
#ax.spines['left'].set_bounds([0,1])
plt.legend()




# # #Do this for true R2
# # #Plot timecourse of cognitive vars
# # colors = ['#933f39', '#31618c', '#e0b329', '#5a8c45']
# data = []
# line_labels = []
# for k in [0,2,10,11]:
#     data.append(cvR_timecourse_single[k])
#     line_labels.append(labels[k])
# ax = plt.figure(figsize=[10, 4.8]).add_subplot()
# plot_timecourse(ax, data, frame_rate, idx_list, spacer=4, colors=colors, line_labels = line_labels, x_label = 'Time (s)', y_label = 'cvR\u00b2')
# separate_axes(ax)

# #Plot time course for different variable groups
# data = []
# for k in [0,2,10,11]:
#     data.append(dR_timecourse_single[k])
# ax = plt.figure(figsize=[10, 4.8]).add_subplot()
# plot_timecourse(ax, data, frame_rate, idx_list, spacer=4, colors=colors, line_labels = labels, x_label = 'Time (s)', y_label ='\u0394R\u00b2')
# #ax.set_ylim([0,0.032])
# separate_axes(ax)

#%%-

#L


# from sklearn.decomposition import PCA

# history_data = betas[enc['variable_index']['previous_choice_outcome_combination'],:]

# pc = PCA(50).fit(history_data)
# exp_var = np.cumsum(pc.explained_variance_ratio_)

# #Perform hisotry permutations and compare explained variance increase
# rounds = 100
# shuffled_var = np.zeros([exp_var.shape[0], rounds])
# for ro in range(rounds):
#     block = int(history_data.shape[0] / 4)
#     shuffle = np.array(history_data)
#     ind = [np.arange(block), np.arange(block,block*2), np.arange(block*2, block*3), np.arange(block*3,block*4)]
#     for k in range(shuffle.shape[1]):
#         perm = np.hstack([ind[i] for i in np.random.permutation(4)])
#         shuffle[:,k] = shuffle[perm, k]
#     ps = PCA(50).fit(shuffle)
#     shuffled_var[:,ro] = np.cumsum(ps.explained_variance_ratio_)
    
    
# conf_interval = np.std(shuffled_var,axis=1) / np.sqrt(shuffled_var.shape[1])
   
    
