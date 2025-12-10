#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 23 15:14:58 2025

@author: loesch
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
from sklearn.decomposition import PCA
from scipy.spatial import procrustes
from scipy.ndimage import zoom

#%%-------Define the sessions for the dimensionality and procrustes analysis---

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

#%%-------Load the encoding model betas and run PCA-----------------

file_name = 'RidgeCV_encoding_models_ITI_partitioned_Fig3.npy'
regressors = ['previous_choice_outcome_combination','chest_point','video_svd'] #Only include chest point as a control

cum_variance = {key: [] for key in regressors} # cool, dict comprehension!
dimensionality = {key: [] for key in regressors}
reduction_dims = []
n_neurons = []

#Load and extract dimensionality
for session_dir in sessions:
    enc = np.load(os.path.join(session_dir, 'analysis', file_name), allow_pickle=True).tolist()
    n_neurons.append(enc['betas'][0].shape[0])
    if n_neurons[-1] > 200: #Determine maximum number of PCA components, the 200 is the limit because of the 200 video components
        dim = 200
    else:
        dim = n_neurons[-1]
    reduction_dims.append(dim)
    
    for reg in regressors:
        betas = enc['betas'][0][:, enc['variable_index'][reg]]
        pc = PCA(dim).fit(betas.T)
        tmp = np.ones(200)*np.nan
        tmp[:dim] = np.cumsum(pc.explained_variance_ratio_)
        cum_variance[reg].append(tmp)
        dimensionality[reg].append(np.min(np.where(tmp > 0.9)[0]))
    
#%%--------Figure 4C: Plot the dimensionality averaged over subjects

# Identify the subject for every sessions
subj = np.unique([os.path.split(os.path.split(session_dir)[0])[1] for session_dir in sessions])
subj_code = np.zeros([len(sessions)],dtype=int)
subj_string = []
for n in range(len(sessions)):
    for k in range(subj.shape[0]):
        if subj[k] in sessions[n]:
            subj_code[n] = k
            subj_string.append(str(subj[k]))

cols = ['#765e9c', '#98446f','#629bc6']
gray = '#858585'

#Plot the cumulative explained variance per regressor first, figure 4C
ninty_reg = []
fi = plt.figure(figsize=[4.8,4.8])
ax = fi.add_subplot(111)
for n in range(len(regressors)):
    #Now average first within subjectgs
    tmp = []
    for k in range(subj.shape[0]):
        tmp.append(np.nanmean(np.squeeze(cum_variance[regressors[n]])[subj_code==k,:],axis=0))
    sub_cum_var = np.vstack(tmp).T
    
    av = np.nanmean(sub_cum_var,axis=1)
    sem = np.nanstd(sub_cum_var,axis=1) / np.sqrt(sub_cum_var.shape[1])
    
    ax.fill_between(np.arange(av.shape[0]), av-sem, av+sem, color=cols[n], linewidth=0, alpha = 0.4)
    ax.plot(np.arange(av.shape[0]), av, color=cols[n])
    
    ninty_point = np.min(np.where(av > 0.9)[0])
    ninty_reg.append(ninty_point)
    ax.plot([-5, ninty_point], [av[ninty_point], av[ninty_point]], linewidth=0.5, linestyle = '--', color=cols[n])
    ax.plot([ninty_point, ninty_point], [0, av[ninty_point]], linewidth=0.5, linestyle = '--', color=cols[n])
lims = ax.get_ylim()
ax.set_ylim([0,lims[1]])
ax.set_xlim([-5,200])
ax.set_xlabel('Number of dimensions')
ax.set_ylabel('Cumulative explained variance')
separate_axes(ax)
    
# Plot the number of components required to explain 90% of the beta weight variance
# Figure 4D
fi = plt.figure(figsize=[2.5,4.8])
ax = fi.add_subplot(111)
tmp_dims = []
for n in range(len(regressors)):
    #Now average first within subjects
    tmp = []
    for k in range(subj.shape[0]):
        tmp.append(np.nanmean(np.squeeze(dimensionality[regressors[n]])[subj_code==k]))
    sub_dim = np.squeeze(tmp)
    tmp_dims.append(sub_dim)
    av = np.mean(sub_dim)
    sem = np.std(sub_dim) / np.sqrt(sub_dim.shape[0])
    ax.errorbar(n, av, yerr=sem, color = cols[n], fmt='o', markerfacecolor = 'w', capsize=4)
for k in range(len(regressors)-1):
    ax.plot([0.12+ k , 0.88 +k], [tmp_dims[k], tmp_dims[k+1]], linewidth=0.5, color=gray, marker='.', markersize=4)
ax.set_ylim([0,100])
ax.set_xticks(np.arange(len(regressors)))
ax.set_xticklabels(regressors, ha = 'right', rotation=45)
ax.set_ylabel('Number of dimensions')
separate_axes(ax)

#Construct csv for testing
av_dim = np.zeros([subj.shape[0], len(regressors)]) * np.nan
for k in range(len(regressors)):
    for n in np.unique(subj_code):
        av_dim[n, k] = np.mean(np.array(dimensionality[regressors[k]])[subj_code==n])
sem_dim = np.std(av_dim,axis=0) / np.sqrt(av_dim.shape[0])
ses_dim = []
for x in regressors:
    ses_dim = ses_dim + dimensionality[x]
reg_id = np.repeat(regressors, len(n_neurons)).tolist()
subject = subj_string * len(regressors)
neuron_num = n_neurons * len(regressors)
session_id = [os.path.split(x)[1] for x in sessions] * len(regressors)      

d = {'dimensionality': ses_dim, 'regressor': reg_id, 'subject': subject, 'session': session_id, 'neuron_number': neuron_num}
dim_data = pd.DataFrame(d)
#dim_data.to_csv(os.path.join('/Users/loesch/Documents/Churchland_lab/Trial-history_manuscript/Figures/Fig5_panels','encoding_weight_dimensionality.csv'))

#%%----Figure 4G: Match the trial history dynamics, the chest point tuning motifs,
# and the video SVD motifs between sessions

#Use the number of neural dimensions that explain 90% of the encoding weight
#variance for every variable.
#Note that the dimensionality of the encoding does not directly influence the 
#procrustes distance because the average chest point tuning dimensionality is
#similar to the trial history but the procrustes distance is significantly 
#different. Also, even if the numbers of neural dimensions are different for the
#different variables we make sure to include at least 90% of the variance, so
#adding more dimensions to the lower dimensional variables does only marginally
#alter the procrustes results because the extra dimensions are extremely low-variance.
use_dims = np.max(dimensionality[regressors[0]]) #Find the session with maximum required dimensions
ref_session = np.argmax(np.squeeze(cum_variance[regressors[0]])[:,35]) #Determine which session has most explained variance with 35 components and set as reference

all_projections = [] #Encoding weights projected into PC space 
procrustes_distance = [] #Procrustes distance between sessions
matched_matrices = [] #The target matrices matched to the reference session (after translation, scalingand rotation)

for reg in range(len(regressors)):
    use_dims = np.max(dimensionality[regressors[reg]]) #Include the minimum amount of dimensions to cover 90 of the variance of every mouse
    projection = [] #Compute the pca and project encoding weights into that space
    for session_dir in sessions:
        enc = np.load(os.path.join(session_dir, 'analysis', file_name), allow_pickle=True).tolist()
        betas = enc['betas'][0][:, enc['variable_index'][regressors[reg]]]
        pc = PCA(use_dims).fit(betas.T)
        projection.append(pc.transform(betas.T))
        
    #Preform procrustes analysis for every session
    matched_mat = [] #Retain the transformed matrices
    procrustes_dist = np.zeros([len(projection),len(projection)]) *np.nan #Calculate the procrustes distance
    for n in range(len(projection)):
        tmp = []
        for k in range(len(projection)):
            _, tmp_mat, tmp_dist = procrustes(projection[n], projection[k])
            tmp.append(tmp_mat)
            procrustes_dist[k,n] = tmp_dist
        matched_mat.append(tmp)
        
    all_projections.append(projection)
    procrustes_distance.append(procrustes_dist)
    matched_matrices.append(matched_mat)
    
#Compare within subject across session distance and across subject distance
row_indices, column_indices = np.triu_indices_from(procrustes_distance[0], k=1) #Get indices of upper triangular excluding diagonal
#Do this because the distance matries are symmetric and so we want to avoid duplication of results

ref_subj = np.zeros(row_indices.shape) * np.nan
test_subj = np.zeros(row_indices.shape) * np.nan
distance = np.zeros([row_indices.shape[0], len(regressors)]) * np.nan
for reg in range(len(regressors)):
    tmp = np.zeros(row_indices.shape) * np.nan
    for k in range(row_indices.shape[0]):
        if reg==0:
            test_subj[k] = subj_code[column_indices[k]]
            ref_subj[k] = subj_code[row_indices[k]]
        distance[k,reg] = procrustes_distance[reg][row_indices[k],column_indices[k]]

within_dist = []
across_dist = [] 
no_subj = np.unique(subj_code).shape[0]
for n in range(no_subj):
    #Retrieve the distances for within-subject sessions
    within_dist.append(np.mean(distance[(ref_subj == n) & (test_subj == n)],axis=0)) #Compute the within average
    #Now get across subjects distance
    #Caution: This only works if the last subject has more than one session in
    #the dataset. If there is only one session for that subject all the comparisons
    #are already covered by the other subjects' comparisons with that session
    tmp = []
    for k in range(no_subj):
        tmp.append(np.mean(distance[(ref_subj == n) & (test_subj == k)],axis=0))
    across_dist.append(np.nanmean(np.squeeze(tmp),axis=0))

combined = [np.vstack(within_dist), np.vstack(across_dist)]

#Figure 4G: Plot boxplot of procrustes distance
no_nan = np.isnan(combined[0][:,0]) == 0 #For some mice only one session is included,
#so no within comparison can be performed but all across mice comparisons are possible
fi =plt.figure(figsize=[4.8,4.8])
ax = fi.add_subplot(111)
fancy_boxplot(ax,combined[0][no_nan,:],cols, widths = 0.2, positions =  [0.75, 1, 1.25])
fancy_boxplot(ax,combined[1],cols, widths = 0.2, positions = [1.75,2, 2.25])

tmp = (0.5-np.random.rand(combined[0].shape[0])) * 0.2
jitter = tmp
tmp = [combined[0][:,0], combined[0][:,1],combined[0][:,2], combined[1][:,0], combined[1][:,1], combined[1][:,2]]
pos = [0.75, 1, 1.25, 1.75, 2, 2.25]
for k in range(len(pos)):
    no_nan = np.isnan(tmp[k]) == 0
    posi = np.repeat(pos[k], np.sum(no_nan))
    ax.scatter(posi + jitter[no_nan], tmp[k][no_nan], color = gray, s = 14)
ax.set_xlim([0.5, 2.5])
ax.set_ylim([0, 1])
ax.set_xticks([1,2])
ax.set_xticklabels=(['within', 'across'])
separate_axes(ax)

#Prepare csv for lme
d_dict = dict()
d_dict['procrustes_distance'] = np.hstack(([x.T.flatten() for x in combined]))
d_dict['regressor'] = np.repeat(regressors, no_subj).tolist() * len(combined)
d_dict['condition'] = ['within'] * no_subj * len(regressors) + ['across'] * no_subj * len(regressors)
d_dict['subject'] = np.unique(subj_string).tolist() * 2 * len(regressors)
dist_data = pd.DataFrame(d_dict)
#dist_data.to_csv(os.path.join('/Users/loesch/Documents/Churchland_lab/Trial-history_manuscript/Figures/Fig5_panels','procrustes_distance.csv'))

#%%---Figure 4A and B: Example trial history encoding dynamics and chest tuning motifs

#Start plotting the trial history dynamics
hist_cols = np.array(['#606cbe', '#48a7b1', '#deb35e', '#de8468'])
hist_cols = hist_cols[[1,0,2,3]]

# Optional: Plot the trial history dynamics projected into the pc space for every session
# pro = all_projections[0] #Use the stored projected dynamics for trial history (regressor 0)
# for n in range(len(all_projections[0])):
#     history_dynamics = np.reshape(all_projections[0][n].T, (all_projections[0][n].shape[1], 4, 133)).transpose(2,1,0)
    
#     ax = plt.figure().add_subplot(111, projection='3d')
#     for k in range(4):
#         ax.plot(history_dynamics[:,k,0], history_dynamics[:,k,1], history_dynamics[:,k,2], hist_cols[k])
#         ax.scatter(history_dynamics[0,k,0], history_dynamics[0,k,1], history_dynamics[0,k,2], s = 30, c = 'k', marker='^' )
#         ax.scatter(history_dynamics[132,k,0], history_dynamics[132,k,1], history_dynamics[132,k,2] ,s = 30, c='k', marker='o' )    
#     ax.set_xlabel('PC1')
#     ax.set_ylabel('PC2')
#     ax.set_zlabel('PC3')
#     ax.set_xticklabels([])
#     ax.set_yticklabels([])
#     ax.set_zticklabels([])
#     ax.set_title(sessions[n])

target_session = 11
history_dynamics = np.reshape(all_projections[0][target_session].T, (all_projections[0][target_session].shape[1], 4, 133)).transpose(2,1,0)
ax = plt.figure(figsize=[4.8,4.8]).add_subplot(111,aspect='equal')
for k in range(4):
    ax.plot(history_dynamics[:,k,0], history_dynamics[:,k,1], hist_cols[k])
    ax.scatter(history_dynamics[0,k,0], history_dynamics[0,k,1], s = 30, c = 'k', marker='^' )
    ax.scatter(history_dynamics[132,k,0], history_dynamics[132,k,1],s = 30, c='k', marker='o' )    
    ax.set_xlabel('Trail history encoding dim 1')
    ax.set_ylabel('Trail history encoding dim 2')
    ax.grid('on')
ymin = ax.get_ylim()[0]
ax.set_ylim([ymin, 6])

#Figure 4B: Plot first dim of spatial chest tuning for 
target_session = 18 #Actually the reference session
colormap_str = 'BrBG_r'
#Recover the spatial binning of the video that was used for the chest position tuning
video_dimensions = [512,640] #Unfortunately hard-coded here, but could easily be changed if necessary
pixels_per_bin = 32 #At 25 px/cm, so that each place field is 1.28 cm**2
x_bins = np.arange(0, video_dimensions[1] + pixels_per_bin, pixels_per_bin) #Divide video width in bins
y_bins = np.arange(0, video_dimensions[0] + pixels_per_bin, pixels_per_bin)

cb = matched_matrices[1][target_session][target_session] #Retrieve the chest tuning motifs matched to the reference session (target)
#Reshape the data to reconstruct the image
beta_image = np.stack([np.reshape(cb[:,x], (x_bins.shape[0]-1, y_bins.shape[0]-1)).T for x in range(cb.shape[1])], axis=2)
#Interpolate to get a smooth tuning map
full_interpolated = np.stack([zoom(beta_image[:,:,x], 32, order = 3) for x in range(beta_image.shape[2])],axis=2)
  
minmax = [np.min(full_interpolated[:,:,:3]), np.max(full_interpolated[:,:,:3])]
limit = np.max(np.abs(minmax)) #Draw symmetrical limit
pick_comp = [0,2]
ylabels = ['Chest tuning dim 1', 'Chest tuning dim 3']
fi, ax = plt.subplots(nrows=len(pick_comp), ncols=1)
for n in range(len(pick_comp)):
    im = ax[n].imshow(np.flip(full_interpolated[:,:,pick_comp[n]]), vmin=-1*limit, vmax=limit, cmap = colormap_str)
    ax[n].set_xticks([])
    ax[n].set_yticks([])
    ax[n].set_ylabel(ylabels[n])
cbar = fi.colorbar(im, ax=ax.ravel().tolist())
cbar.set_label('A.U.')

#####################################################################
# #Optional: Do the reconstruction for the video svd dimensions
# session_dir = sessions[target_session]
# video_spatial, video_temporal = np.load(glob(session_dir + '/analysis/*video_SVD.npy')[0], allow_pickle = True).tolist()
# video_std = np.std(video_temporal,axis=1) #Get the standard deviation of the original traces to scale the PCA of encoding model weights 
# video_projection = all_projections[2][target_session]
# video_components = video_spatial @ video_projection

# #Plot the spatial components of the video projected into the pc space
# lower = np.min(video_components[:,:,0])
# upper= np.max(video_components[:,:,0])
# fi, ax = plt.subplots(nrows=1, ncols=2)
# for k in range(2):
#     image = ax[k].imshow(video_components[:,:,k],vmin=lower,vmax=upper, cmap='cividis',aspect='equal')
# cax,kw = matplotlib.colorbar.make_axes([x for x in ax.flat])
# fi.colorbar(image, cax=cax, **kw)
##############################################################################

#%%----------Figure F: Plot matched encoding dynamics or motifs

#Figure 4F, top: Plot matched within and across trial history dynamics
# Find all the sessions matched to the target session
target_session = 11
matched_dyn = []
for mat in matched_matrices[0][target_session]:
    matched_dyn.append(np.reshape(mat.T, (mat.shape[1], 4, 133)).transpose(2,1,0))
matched_dyn = np.stack(matched_dyn,axis=3)
inds = list(set(np.arange(matched_dyn.shape[3],dtype=int)) - set([target_session]))
# Plot the matched dynamics
ax = plt.figure().add_subplot(111, aspect='equal')
for k in inds:
    ax.plot(matched_dyn[:,:,0,k], matched_dyn[:,:,1,k], color =gray, linewidth=0.5)
ax.plot(matched_dyn[:,:,0,target_session], matched_dyn[:,:,1,target_session], color ='k')
ax.set_xlabel('Trial history dynamics dim 1')
ax.set_ylabel('Trial history dynamics dim 2')

#Figure 4F, bottom: Plot first dim of spatial chest tuning for two sessions
target_session = 18 #Actually the reference session
compare = 22 #The test session to be compared with the above
colormap_str = 'BrBG_r'
#Recover the spatial binning of the video that was used for the chest position tuning
video_dimensions = [512,640] #Unfortunately hard-coded here, but could easily be changed if necessary
pixels_per_bin = 32 #At 25 px/cm, so that each place field is 1.28 cm**2
x_bins = np.arange(0, video_dimensions[1] + pixels_per_bin, pixels_per_bin) #Divide video width in bins
y_bins = np.arange(0, video_dimensions[0] + pixels_per_bin, pixels_per_bin)

matched_tuning = []
for k in range(len(sessions)):
    cb = matched_matrices[1][target_session][k] #Retrieve the chest tuning motifs matched to the reference session (target)
    #Reshape the data to reconstruct the image
    beta_image = np.stack([np.reshape(cb[:,x], (x_bins.shape[0]-1, y_bins.shape[0]-1)).T for x in range(cb.shape[1])], axis=2)
    #Interpolate to get a smooth tuning map
    full_interpolated = np.stack([zoom(beta_image[:,:,x], 32, order = 3) for x in range(beta_image.shape[2])],axis=2)
    matched_tuning.append(full_interpolated)
matched_tuning = np.stack(matched_tuning,axis=3)

minmax = [np.min([matched_tuning[:,:,:3,target_session], matched_tuning[:,:,:3,compare]]), np.max([matched_tuning[:,:,:3,target_session], matched_tuning[:,:,:3,compare]])]
limit = np.max(np.abs(minmax)) #Draw symmetrical limit
for k in [target_session, compare]:
    for n in [0,1,2]: #First three pc dimensions
        plt.figure()
        plt.imshow(np.flip(matched_tuning[:,:,n,k]), vmin=-1*limit, vmax=limit, cmap = colormap_str)
        plt.colorbar()
        plt.xticks([])
        plt.yticks([])
        plt.title(f'Session {k}, dimension {n}')
  