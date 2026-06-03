# MASTERTHESIS---SECUREL-DYNAMIC-GRID-MODEL
This repository constains a dynamic model in TOPS based on the securEL refrence grid. It was created for the purposes of Henrik Røren and Ole Henrik Løvgrens's master thesis in the spring of 2026. 

The repo is structured into several subfolders. 

In the dyn_sim folder the scripts of executing the dynamic simulation and plotting is located.
plot2 plots several different grid models with changing converter biases and plots in same plot.
plotting is a single simulation of a three phase fault that results in the grid going into island operation. 

The FCR folder contain scripts for checking both FCR-N and FCR-D as well as a sweep script that finds the minimum converter filter constant to pass the requirements. 

ps_data contains several versjons of the secureEL grid model. They correspond to different converter biases and load imbalances. The bias2,5 etc are all using the maxiumu import load scenario. Import 50 and 75 is 50% and 75% load import imbalance of the maximport scenario. Sixtyfiveimport is a import of 65 MW, while export case is an export of 134 MW. 

The dynamic models for the system components are located in src. Here the gen.py code is modified from the original TOPS. hygov2 and avr2 are both newly made for the thesis. In hygov2 there is some modification needed to ensure that the generators remain at zero mechanical power once this level is reached. This is marked in the script. For normal operation this is not on by default. 

If there are any questions regarding the code please send an email to henrik.roren@gmail.com
