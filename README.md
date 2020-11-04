# andi_challenge

Analysis of single particle tracking data for characterisation of anomalous diffusion, using a deep learning approach. Submitted to the Anomalous Diffusion Challenge: http://www.andi-challenge.org/

Performs three tasks on 1D and 2D tracks:
-	Task 1: inference of anomalous exponent
-	Task 2: classification of diffusion model
-	Task 3: segmentation of trajectories

Contains pre-trained models, and the code used to train these models


### Instructions for analysing competition data

Repository includes three scripts for analysing competition data in the ‘Competition’ folder, named Task1.py, Task2.py and Task3.py for the three tasks. To perform analysis, run these scripts, specifying the path to the data folder at the top of the file. Python package requirements listed in requirements.txt.
