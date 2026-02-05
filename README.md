# Oesch_et_al_Trial_History_ACC #
Analyses from "Anterior cingulate neurons combine outcome monitoring of past decisions with ongoing movement signals", Lukas T. Oesch, Makenna C. Thomas, Davis Sandberg, João Couto and Anne K. Churchland: https://doi.org/10.1101/2025.04.11.648398.

This repository contains the analysis scripts to reproduce the different figure panels and the code to run the statistics reported in the manuscript. Scripts are structured based on the the figures, whose panels they reproduce.

## System requirements and dependencies ##
The scripts were tested on MacOs Sequoia 15.5 running Python 3.12 and RStudio 2024.12.1+563. The following python toolboxes were used:
- numpy 1.26.4
- pandas 2.2.2
- scipy 1.13.1
- matplotlib 3.9.2
- sklearn 1.5.1
- umap-learn 0.5.11

Other standard toolboxes include os, sys, glob, time, and multiprocessing.

Please make sure to also clone the following repositories and to add their paths to your python path where necessary:
https://github.com/churchlandlab/chiCa - containing code for most of the imaging and behavioral analyses
https://github.com/jcouto/fit_psychometric - to estimate parameters of psychometric functions.

The original code also requires interactions with a remote to search through the data of individual subjects. This is performed using the following toolbox: https://github.com/jcouto/labdata-tools. As of the date of publication of this work this version of labdata-tools is depreciated.

All data included in the analyses can be manually downloaded here: https://doi.org/10.6084/m9.figshare.30670382
The scripts contain instructions on how to execute the code without needing labdata-tools.

## Installation guide ##
Simply clone this repository:

code(https://github.com/LukasOesch/Oesch_et_al_Trial_History_ACC.git)

## Instructions for use ##
As of the date of publication of this work, the version of labdata-tools is not supported anymore (pleaes see: https://jcouto.github.io/labdata-docs/ for the latest version of labdata). To run the analyses please download the subject data to your computer and retain the file directory as your base_dir (see specific instructions inside the scripts). It is recommended to run the different parts of the scripts in order as individual sections may rely on variables created earlier. While all the analysis files to plot the results are already created in the data provided, the code to produce these results can be also be found in the respective scripts.

Please do reach out to the corresponding author of this study if you have questions or comments. We highly value your input!

LO, February 4th, 2026
