# Oesch_et_al_Trial_History_ACC #
Analyses from "Anterior cingulate cortex mixes retrospective cognitive signals and ongoing movement signatures during decision-making" Lukas T. Oesch, Makenna C. Thomas1, Davis Sandberg, João Couto and Anne K. Churchland

This repository contains the analysis scripts to reproduce the different figure panels and the code to run the statistics reported in the manuscript. Scripts are structured based on the the figures, whose panels they reproduce.

## System requirements and dependencies ##
The scripts were tested on MacOs Sequoia 15.5 running Python 3.12 and RStudio 2024.12.1+563. The following python toolboxes were used:
- numpy 1.26.4
- pandas 2.2.2
- scipy 1.13.1
- matplotlib 3.9.2
- sklearn 1.5.1
Other standard toolboxes include os, sys and glob, time, multiprocessing.

Please make sure to also clone the following repositories:
https://github.com/churchlandlab/chiCa - containing code for most of the imaging and behavioral analyses
https://github.com/jcouto/fit_psychometric - to estimate parameters of psychometric functions

The original code also requires interactions with a remote to search through the data of individual subjects. This is performed using the following toolbox: https://github.com/jcouto/labdata-tools
However, the data included in the analyses can also be manually downloaded here: https://uclahs.box.com/s/n732k7tiy9qjyrmk39yznn2ugsqbneo7 (320 GB)
All the scripts contain instructions for how to execute the code without needing labdata-tools.

## Installation guide ##
Simply clone this repository:

code(https://github.com/LukasOesch/Oesch_et_al_Trial_History_ACC.git)

## Instructions for use ##
Please make sure to have labdata-tools installed if you need the scripts to fetch data from remote. Otherwise please download the subject data to your computer and retain the file directory as your base_dir (see specific instructions inside the scripts). It is recommended to run the different parts of the scripts in order as individual sections may rely on variables created earlier. Please make sure to first run the scripts to create the analyses for the main figures before running the supplementary figure scripts, so the respective files get created and are available.

LO, July 15th, 2025
