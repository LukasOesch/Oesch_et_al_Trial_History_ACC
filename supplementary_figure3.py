#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 30 09:58:35 2025

@author: loesch
"""

from chiCa import *
import chiCa
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Arial"
from labdatatools import *
import os
from glob import glob
from scipy.ndimage import zoom

#%%---Load the data from the example session in panes a-d and compute cvR and dR

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

#%%---- Compare the regressor weights for either the trial history variable,
# the head yaw or the chest position tuning only models with the weights
# obtained from the full model fit. This representations help to build
# a better intuition about how much the presence of all the other variables
# as opposed to only the variable of interest impacts the model weights. The
# larger the difference between the weights in these two scenarios, the more
# the variable of interest might be correlated or co-encoded with one or more
# of the other variabels.

###############-------- Supplementary Figure 3a ---------######################

n_id = [296] #The example neuron shown in supplementary figure 3a
#n_id = [275, 78, 296] #Alternatively, plot three different neurons that encode
# trial history during different trial phases.

cols = ['#8a2525', '#393838'] #Dark red for individual models, dark grey for full

for neuron_id in n_id:
    
    #---Make plots for the weights of the different trial history regressors
    single_b = np.reshape(betas_single[2][neuron_id,enc['variable_index']['previous_choice_outcome_combination']], (4,133)).T
    full_b = np.reshape(full_betas[neuron_id,enc['variable_index']['previous_choice_outcome_combination']], (4,133)).T
    
    order = [[0,1],[0,0],[1,0],[1,1]] #History order is incor left, corr left, incorr right, corr right, which will be on top right, top left, bottom left and bottom right
    fi, ax = plt.subplots(nrows = 2, ncols=2, sharex=True, sharey=True) 
    for k in range(len(order)):
        plot_timecourse(ax[order[k][0], order[k][1]], [single_b[:,k], full_b[:,k]], 30, idx_list, colors = cols)       
    
    #-----Plot head orientation tuning
    orientation_angle = 'yaw'
    line_color = '#99733e'
    #Get the bin definitions as used in the encoding model fitting
    bin_number = 60 #6° per bin
    bins = np.arange(-np.pi, np.pi+np.pi/(bin_number/2), np.pi/(bin_number/2))[:bin_number+1]
    full_tuning = full_betas[neuron_id, enc['variable_index'][orientation_angle]]
    single_tuning = betas_single[9][neuron_id, enc['variable_index'][orientation_angle]]
    
    fi = plt.figure()
    ax = fi.add_subplot(111, projection='polar')
    ax.plot(bins, np.zeros(bins.shape[0]), color='k', linewidth= ax.spines['inner'].get_linewidth(), linestyle='--')
    ax.plot(bins[:-1], single_tuning, color = cols[0]) # Plot the single variable tuning model
    ax.plot([bins[-2], bins[-1]], [single_tuning[-1], single_tuning[0]], color = cols[0]) #Close the circle
    ax.plot(bins[:-1], full_tuning, color = cols[1]) # Plot the full model weights
    ax.plot([bins[-2], bins[-1]], [full_tuning[-1], full_tuning[0]], color = cols[1]) #Close the circle
    
    #------Now chest point tuning
    colormap_str = 'BrBG_r'
    video_dimensions = [512,640]
    pixels_per_bin = 32 #At 25 px/cm, so that each place field is 1.28 cm**2
    x_bins = np.arange(0, video_dimensions[1] + pixels_per_bin, pixels_per_bin) #Divide video width in bins
    y_bins = np.arange(0, video_dimensions[0] + pixels_per_bin, pixels_per_bin)
    
    #Get the weights, reshape to image dimensions and interpolate to actual pixel values
    cb = full_betas[neuron_id,  enc['variable_index']['chest_point']] # The full model weights
    beta_image = np.reshape(cb, (x_bins.shape[0]-1, y_bins.shape[0]-1)).T
    full_interpolated = zoom(beta_image, 32, order = 3)
    
    cb = betas_single[10][neuron_id,  enc['variable_index']['chest_point']] # The chest-only model weights
    beta_image = np.reshape(cb, (x_bins.shape[0]-1, y_bins.shape[0]-1)).T
    single_interpolated = zoom(beta_image, 32, order = 3)
    
    #Find symmetric boundaries for color-coding
    a_min = np.min(np.vstack((full_interpolated, single_interpolated)))
    b_max = np.max(np.vstack((full_interpolated, single_interpolated)))
    bound = np.max([b_max, np.abs(a_min)])
    
    fi = plt.figure()
    ax = [fi.add_subplot(2,1,k+1, aspect='equal') for k in range(2)]
    tuning_im = [single_interpolated, full_interpolated]
    titles = ['Chest point only', 'Full model']
    for k in range(len(tuning_im)):
        im = ax[k].imshow(np.flipud(tuning_im[k]), cmap = colormap_str, vmin= -bound, vmax=bound)
        ax[k].set_xticks([])
        ax[k].set_yticks([])
        ax[k].set_title(titles[k])
    cbar = fi.colorbar(im, ax=ax)
    cbar.set_label('Encoding weight (A.U.)')
    
#%%-----Correlate the maximally explained variances of all the single-variable
# models with each other and do this also for the unique variances with the 
# one-removed models. 

tmp  = [np.vstack(cvR), np.vstack(dR)]
titles = ['Single variable models', 'One-removed models']
labels = ['Choice', 'Outcome', 'Trial history', 'Stimulus events',
          'Center poke', 'Left poke', 'Right poke',
          'Pitch', 'Roll', 'Yaw', 'Chest position', 'Video SVD', 'Video motion energy SVD']

for k in range(len(tmp)):
    R = np.corrcoef(tmp[k])
    np.fill_diagonal(R,0) #Works within array without reassigning new variable

    fi = plt.figure()
    ax =fi.add_subplot(111, aspect='equal')
    im = ax.matshow(R, vmin=-1, vmax=1, cmap='RdBu_r')
    cbar = fi.colorbar(im)
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xticks(np.arange(R.shape[0]))
    ax.set_xticklabels(labels, ha = 'right', rotation=45)
    ax.set_yticks(np.arange(R.shape[0]))
    ax.set_yticklabels(labels)
    ax.set_title(titles[k])
    cbar.set_label("Pearson's correlation coeeficient")
    
#%%----To generate the timecourses of the explained variance from the different
# models please see 'figure3.py', line 258 - 283
  