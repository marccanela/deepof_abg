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
import numpy as np
import deepof.visuals
import matplotlib.pyplot as plt
import seaborn as sns
from IPython import display
import matplotlib.cm as cm
from sklearn.metrics import silhouette_samples, silhouette_score

# Define directories
directory_output = '/home/sie/Desktop/marc/brain_01a02/'
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
my_deepof_project.load_exp_conditions(directory_output + 'conditions.csv')

# Check conditions
coords = my_deepof_project.get_coords()
print("The original dataset has {} videos".format(len(coords)))
coords = coords.filter_condition({"protocol": "paired"})
print("The filtered dataset has only {} videos".format(len(coords)))

# Perform a supervised analysis
supervised_annotation = my_deepof_project.supervised_annotation()
with open('/home/sie/Desktop/marc/dual videos/supervised_annotation.pkl', 'wb') as file:
    pickle.dump(supervised_annotation, file)
    
# Perform an unsupervised analysis
graph_preprocessed_coords, adj_matrix, to_preprocess, global_scaler = my_deepof_project.get_graph_dataset(
    # animal_id="nocolor", # Comment out for multi-animal embeddings
    center="Center",
    align="Spine_1",
    window_size=25, # Adjust to frame rate
    window_step=1,
    test_videos=1,
    preprocess=True,
    scale="standard",
)

def silhouette_score_unsupervised(my_deepof_project, graph_preprocessed_coords, adj_matrix, to_preprocess, global_scaler):
    # start_time = time.time()
    silhouette_score_dict = {}
    num_clusters = list(range(2, 16))
    # num_clusters = [8]
    
    for num in num_clusters:
        trained_model = my_deepof_project.deep_unsupervised_embedding(
            preprocessed_object=graph_preprocessed_coords, # Change to preprocessed_coords to use non-graph embeddings
            adjacency_matrix=adj_matrix,
            embedding_model="VaDE", # Can also be set to 'VQVAE' and 'Contrastive'
            epochs=10, # (Default 10)
            encoder_type="recurrent", # Can also be set to 'TCN' and 'transformer'
            n_components=num, # (Default 10)
            latent_dim=4, # Dimention size of the latent space (aka, number of embeddings) (Default 4)
            batch_size=1024, # (Default 1024)
            verbose=True, # Set to True to follow the training loop
            interaction_regularization=0.0, # Change to 0.25 when multianimal
            pretrained=False, # Set to False to train a new model!
        )
        embeddings, soft_counts, breaks = deepof.model_utils.embedding_per_video(
            coordinates=my_deepof_project,
            to_preprocess=to_preprocess,
            model=trained_model,
            # animal_id='nocolor', # Comment out for multi-animal embeddings
            global_scaler=global_scaler,
        )
        
        silhouette_score_dict[num] = [embeddings, soft_counts, breaks]
        
    return silhouette_score_dict

silhouette_score_dict = silhouette_score_unsupervised(my_deepof_project, graph_preprocessed_coords, adj_matrix, to_preprocess, global_scaler)

with open('/home/sie/Desktop/marc/brain_01a02/silhouette_score_dict.pkl', 'wb') as file:
    pickle.dump(silhouette_score_dict, file)
with open('/home/sie/Desktop/marc/brain_01a02/silhouette_score_dict.pkl', 'rb') as file:
    silhouette_score_dict = pickle.load(file)

def create_silhouette_score_dict(silhouette_score_dict):
    new_silhouette_score_dict = {}
    
    # for n in [10,11,12,18]:
    #     del silhouette_score_dict[n]

    for key, value in silhouette_score_dict.items():
        print('Analyzing ' + str(key))
        embeddings = value[0]
        soft_counts = value[1]
        breaks = value[2]
        
        hard_counts = {}
        for individual, clusters_probabilities in soft_counts.items():
            # Find the index of the cluster with the maximum probability for each column
            hard_count_indices = [max(range(len(cluster)), key=lambda i: cluster[i]) for cluster in clusters_probabilities]
            # Create a list of hard counts for the individual
            hard_counts[individual] = hard_count_indices
        # Combine all individual hard counts into a single list
        combined_hard_counts = [hard_count for individual_hard_counts in hard_counts.values() for hard_count in individual_hard_counts]
        combined_embeddings = np.concatenate(list(embeddings.values()), axis=0)
        score = silhouette_score(combined_embeddings, combined_hard_counts,
                                 sample_size=10000, random_state=42)

        new_silhouette_score_dict[key] = [score, combined_hard_counts, combined_embeddings, embeddings, soft_counts, breaks]
    
    return new_silhouette_score_dict

new_silhouette_score_dict = create_silhouette_score_dict(silhouette_score_dict)

with open(directory_output + 'new_silhouette_score_dict.pkl', 'wb') as file:
    pickle.dump(new_silhouette_score_dict, file)
with open(directory_output + 'new_silhouette_score_dict.pkl', 'rb') as file:
    new_silhouette_score_dict = pickle.load(file)



        

# If epochs=10, latent_dim=4, batch_size=1024 -> max=0.39 (num_clusters=8)
# If epochs=50, latent_dim=4, batch_size=1024 -> irregular silhouettes, total_loss = ~50
# If epochs=10, latent_dim=6, batch_size=1024 -> 
# If epochs=150, latent_dim=8, batch_size=64 -> just identifies one cluster
# silhouette_score_dict: epochs=150, latent_dim=8, batch_size=1024 (2h/num) -> negative numbers
# silhouette_score_dict_2: epochs=10, latent_dim=8, batch_size=1024 (50min/num) -> max=0.13 (cluster 8)
# silhouette_score_dict_3: epochs=20, latent_dim=8, batch_size=1024 (70min/num) -> 0.03 (cluster 8)

# =============================================================================
# Load a previously saved project and supervised/unsupervised analysis
my_deepof_project = deepof.data.load_project(directory_output + "deepof_tutorial_project")
my_deepof_project.load_exp_conditions(directory_output + 'conditions.csv')

with open(directory_output + 'supervised_annotation.pkl', 'rb') as file:
    supervised_annotation = pickle.load(file)

# =============================================================================
# 
# =============================================================================

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














































