"""
SINGLE ANIMAL (5/2/2024)
DeepOF v.0.5.1
activate_conda env deepof2
Analyze with DLC multia_bed_14_keyp-mcanela-2023-11-16
"""

import os
import pandas as pd
import pickle
import deepof.data

import deepof.visuals
import matplotlib.pyplot as plt
import seaborn as sns

# Define directories
directory_output = '/home/sie/Desktop/marc/brain_01a02/'
directory_dlc = '/home/sie/Desktop/marc/brain_01a02/csv_corrected/'
directory_videos = '/home/sie/Desktop/marc/brain_01a02/avi_corrected/'

# Convert CSV-multi-animal into CSV-single-animal
for file in os.listdir(directory_dlc):
    if file.endswith('.csv'):
        file_path = os.path.join(directory_dlc, file)
        df = pd.read_csv(file_path, header=None)
        df_modified = df.drop(1)
        df_modified.to_csv(file_path[:-7] + '.csv', index=False, header=False)

# Prepare the project
my_deepof_project_raw = deepof.data.Project(
                project_path=os.path.join(directory_output),
                video_path=os.path.join(directory_videos),
                table_path=os.path.join(directory_dlc),
                project_name="deepof_tutorial_project",
                arena="polygonal-manual",
                # animal_ids=['mouse'],
                table_format=".csv",
                video_format=".avi",
                bodypart_graph='deepof_14',
                # exclude_bodyparts=["Tail_1", "Tail_2", "Tail_tip"],
                video_scale=200,
                smooth_alpha=1,
                exp_conditions=None,
)

# Create the project
my_deepof_project = my_deepof_project_raw.create(force=True)

# Edit wrong arenas
my_deepof_project.edit_arenas(
    videos=['20240202_Marc_ERC SOC tone_Males_box ab_240003812_02_01_1'],
    arena_type="polygonal-manual",
)

# Load conditions
my_deepof_project.load_exp_conditions(directory_output + 'conditions.csv')

# Check conditions
coords = my_deepof_project.get_coords()
print("The original dataset has {} videos".format(len(coords)))
coords = coords.filter_condition({"protocol": "paired"})
print("The filtered dataset has only {} videos".format(len(coords)))

# Load a previously saved project
my_deepof_project = deepof.data.load_project(directory_output + "/tutorial_project")

# Perform a supervised analysis
supervised_annotation = my_deepof_project.supervised_annotation()
with open(directory_output + 'supervised_annotation.pkl', 'wb') as file:
    pickle.dump(supervised_annotation, file)



























