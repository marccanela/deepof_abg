"""
SINGLE ANIMAL
DeepOF v.0.5.1
conda activate deepof2
Analyze with DLC multia_bed_14_keyp-mcanela-2023-11-16
"""

import os
import pandas as pd
import pickle
import deepof.data
import copy
import numpy as np
import deepof.visuals
import matplotlib.pyplot as plt
import seaborn as sns
from IPython import display
from networkx import Graph, draw

# Define directories
directory_output = '/home/sie/Desktop/marc/dual videos'
directory_dlc = '/home/sie/Desktop/marc/dual videos/h5'
directory_videos = '/home/sie/Desktop/marc/dual videos/mp4'

# Prepare the project
my_deepof_project_raw = deepof.data.Project(
                project_path=os.path.join(directory_output),
                video_path=os.path.join(directory_videos),
                table_path=os.path.join(directory_dlc),
                project_name="deepof_tutorial_project",
                arena="polygonal-manual",
                animal_ids=['colortail','nocolor'],
                table_format=".h5",
                video_format=".mp4",
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
    videos=['20240119_Marc_ERC SOC light_Males_box de_06_01_1'],
    arena_type="polygonal-manual",
)

# Load conditions
my_deepof_project.load_exp_conditions(directory_output + '/conditions.csv')

# Check conditions
coords = my_deepof_project.get_coords()
print("The original dataset has {} videos".format(len(coords)))
coords = coords.filter_condition({"protocol": "hc_ind"})
print("The filtered dataset has only {} videos".format(len(coords)))

# Perform a supervised analysis
supervised_annotation = my_deepof_project.supervised_annotation()
with open('/home/sie/Desktop/marc/dual videos/supervised_annotation.pkl', 'wb') as file:
    pickle.dump(supervised_annotation, file)
    
# Perform an unsupervised analysis
graph_preprocessed_coords, adj_matrix, to_preprocess, global_scaler = my_deepof_project.get_graph_dataset(
    animal_id="nocolor", # Comment out for multi-animal embeddings
    center="Center",
    align="Spine_1",
    window_size=25,
    window_step=1,
    test_videos=1,
    preprocess=True,
    scale="standard",
)

trained_model = my_deepof_project.deep_unsupervised_embedding(
    preprocessed_object=graph_preprocessed_coords, # Change to preprocessed_coords to use non-graph embeddings
    adjacency_matrix=adj_matrix,
    embedding_model="VaDE", # Can also be set to 'VQVAE' and 'Contrastive'
    epochs=10,
    encoder_type="recurrent", # Can also be set to 'TCN' and 'transformer'
    n_components=10,
    latent_dim=4,
    batch_size=1024,
    verbose=False, # Set to True to follow the training loop
    interaction_regularization=0.0,
    pretrained=False, # Set to False to train a new model!
)

# Get embeddings, soft_counts, and breaks per video
embeddings, soft_counts, breaks = deepof.model_utils.embedding_per_video(
    coordinates=my_deepof_project,
    to_preprocess=to_preprocess,
    model=trained_model,
    animal_id='nocolor', # Comment out for multi-animal embeddings
    global_scaler=global_scaler,
)

# =============================================================================
# Load a previously saved project and supervised analysis
my_deepof_project = deepof.data.load_project(directory_output + "/deepof_tutorial_project")
my_deepof_project.load_exp_conditions(directory_output + '/conditions.csv')
with open('/home/sie/Desktop/marc/dual videos/supervised_annotation.pkl', 'rb') as file:
    supervised_annotation = pickle.load(file)

# =============================================================================
# Heatmaps
# =============================================================================

sns.set_context("notebook")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

deepof.visuals.plot_heatmaps(
    my_deepof_project,
    ["colortail_Nose"],
    center="arena",
    exp_condition="protocol",
    condition_value="hc_ee",
    ax=ax1,
    show=False,
    display_arena=True,
    experiment_id="average",
)

deepof.visuals.plot_heatmaps(
    my_deepof_project,
    ["nocolor_Nose"],
    center="arena",
    exp_condition="protocol",
    condition_value="hc_ee",
    ax=ax2,
    show=False,
    display_arena=True,
    experiment_id="average",
)

plt.tight_layout()
plt.show()

# =============================================================================
# Animated skeleton
# =============================================================================
import io
import base64

video = deepof.visuals.animate_skeleton(
    my_deepof_project,
    experiment_id="SOC INT IGM 05092023 HC A1-EE C1",
    frame_limit=500,
    dpi=60,
)

html = display.HTML(video)
display.display(html)
plt.close()

# =============================================================================
# Supervised enrichment
# =============================================================================

fig = plt.figure(figsize=(14, 5)).subplot_mosaic(
    mosaic="""
           AAAAB
           AAAAB
           """,
)

deepof.visuals.plot_enrichment(
    my_deepof_project,
    supervised_annotations=supervised_annotation,
    add_stats="Mann-Whitney",
    plot_proportions=True,
    bin_index=0,
    bin_size=120,
    ax = fig["A"],
)

deepof.visuals.plot_enrichment(
    my_deepof_project,
    supervised_annotations=supervised_annotation,
    add_stats="Mann-Whitney",
    plot_proportions=False,
    bin_index=0,
    bin_size=120,
    ax = fig["B"],
)

for ax in fig:
    fig[ax].set_xticklabels(fig[ax].get_xticklabels(), rotation=45, ha='right')
    fig[ax].set_title("")
    fig[ax].set_xlabel("")

fig["A"].get_legend().remove()

plt.tight_layout()
plt.show()

# =============================================================================
# PCA embedding
# =============================================================================

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

deepof.visuals.plot_embeddings(
    my_deepof_project,
    supervised_annotations=supervised_annotation,
    ax=ax1,
)
deepof.visuals.plot_embeddings(
    my_deepof_project,
    supervised_annotations=supervised_annotation,
    bin_size=120,
    bin_index=0,
    ax=ax2,
)

ax1.set_title("supervised embeddings of full videos")
ax2.set_title("supervised embeddings of first two minutes")

plt.tight_layout()
plt.show()

# =============================================================================
# Visualizing temporal and global embeddings
# =============================================================================

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

deepof.visuals.plot_embeddings(
    my_deepof_project,
    embeddings,
    soft_counts,
    breaks,
    aggregate_experiments=False,
    samples=100,
    ax=ax1,
    save=False, # Set to True, or give a custom name, to save the plot
)

deepof.visuals.plot_embeddings(
    my_deepof_project,
    embeddings,
    soft_counts,
    breaks,
    aggregate_experiments="time on cluster", # Can also be set to 'mean' and 'median'
    exp_condition="protocol",
    show_aggregated_density=False,
    ax=ax2,
    save=False, # Set to True, or give a custom name, to save the plot,
)
ax2.legend(
    bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0
)

plt.tight_layout()
plt.show()

# =============================================================================
# Generating Gantt charts with all clusters
# =============================================================================

fig = plt.figure(figsize=(12, 6))

deepof.visuals.plot_gantt(
    my_deepof_project,
    soft_counts=soft_counts,
    experiment_id="SOC INT IGM 05092023 HC A1-EE C1",
)

# =============================================================================
# Global separation dynamics
# =============================================================================

fig, ax = plt.subplots(1, 1, figsize=(12, 4))

deepof.visuals.plot_distance_between_conditions(
    my_deepof_project,
    embeddings,
    soft_counts,
    breaks,
    "protocol",
    distance_metric="wasserstein",
    n_jobs=1,
)

plt.show()



fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

deepof.visuals.plot_embeddings(
    my_deepof_project,
    embeddings,
    soft_counts,
    breaks,
    aggregate_experiments="time on cluster",
    bin_size=10, # This parameter controls the size of the time bins. We set it to match the optimum reported above
    bin_index=0, # This parameter controls the index of the bins to select, we take the first one here
    ax=ax1,
)

deepof.visuals.plot_embeddings(
    my_deepof_project,
    embeddings,
    soft_counts,
    breaks,
    aggregate_experiments="time on cluster",
    exp_condition="protocol",
    show_aggregated_density=True,
    bin_size=10, # This parameter controls the size of the time bins. We set it to match the optimum reported above
    bin_index=3, # This parameter controls the index of the bins to select, we take the fourth one here
    ax=ax2,
)
ax2.legend(
    bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0
)

ax1.legend().remove()
plt.tight_layout()
plt.show()

# =============================================================================
# Exploring cluster enrichment across conditions
# =============================================================================

fig, (ax, ax2) = plt.subplots(2, 1, figsize=(12, 5))

deepof.visuals.plot_enrichment(
    my_deepof_project,
    embeddings,
    soft_counts,
    breaks,
    normalize=True,
    add_stats="Mann-Whitney",
    exp_condition="protocol",
    verbose=False,
    ax=ax,
)

deepof.visuals.plot_enrichment(
    my_deepof_project,
    embeddings,
    soft_counts,
    breaks,
    normalize=True,
    bin_size=125,
    bin_index=0,
    add_stats="Mann-Whitney",
    exp_condition="protocol",
    verbose=False,
    ax=ax2,
)
ax.tick_params(
    axis='x',
    which='both',
    bottom=False,
    top=False,
    labelbottom=False)
ax.set_xlabel("")
ax2.legend().remove()
plt.title("")
plt.tight_layout()
plt.show()

# =============================================================================
# Exploring cluster dynamics across conditions
# =============================================================================

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Transition matrices and heatmaps
deepof.visuals.plot_transitions(
    my_deepof_project,
    embeddings.filter_videos(my_deepof_project.get_exp_conditions.keys()),
    soft_counts.filter_videos(my_deepof_project.get_exp_conditions.keys()),
    breaks.filter_videos(my_deepof_project.get_exp_conditions.keys()),
   # cluster=False,
    visualization="heatmaps",
    bin_size=125,
    bin_index=0,
    exp_condition="protocol",
    axes=axes,
)

plt.tight_layout()
plt.show()



fig, axes = plt.subplots(1, 2, figsize=(12, 5))

deepof.visuals.plot_transitions(
    my_deepof_project,
    embeddings.filter_videos(my_deepof_project.get_exp_conditions.keys()),
    soft_counts.filter_videos(my_deepof_project.get_exp_conditions.keys()),
    breaks.filter_videos(my_deepof_project.get_exp_conditions.keys()),
    visualization="networks",
    silence_diagonal=True,
    bin_size=125,
    bin_index=0,
    exp_condition="protocol",
    axes=axes,
)

plt.tight_layout()
plt.show()


fig, ax = plt.subplots(1, 1, figsize=(12, 2))

deepof.visuals.plot_stationary_entropy(
    my_deepof_project,
    embeddings,
    soft_counts,
    breaks,
    exp_condition="protocol",
    ax=ax,
)














































