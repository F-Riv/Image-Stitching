### Structure: 

Image_stiching.ipynb is a jupyter notebook with the main steps for image stitching 
modules:
- Distances.py contains functions related to calculating the difference between keypoints according to their descriptor
- Graphs.py contains functions for plotting graphs 
- Patches.py contains functions for generating, plotting and compare  patches created around the inliers and their transformed version, they are compared by SSIM and Norm. MSE
- Visualization.py contrains function for plotting the images with keypoints and the stiched image with inliers
- Utils_RANSAC.py contains other functions 
- json_read: contains the class for creating a json object where it is possible to read and write the best parameters of a run 

Json files:
- best_param contains best parameters from previous iterations 
- custom_param permit to set new parameters to explore 


The whole sequence of steps can be run by cells in the main section after running the other cells 
