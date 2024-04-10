"""
Version number 1: 02/11/2023
@author: mcanela
DeepOF SUPERVISED ANALYSIS TIMESERIES PLOTS
"""

import deepof.data
import copy
import numpy as np
import pandas as pd
import os
import pickle
import pingouin as pg

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.collections import PatchCollection
import matplotlib.ticker as mtick

from scipy.stats import ttest_rel, wilcoxon

blue = '#194680'
red = '#801946'
grey = '#636466'

# directory_output = '/home/sie/Desktop/'
# with open(directory_output + 'supervised_annotation.pkl', 'rb') as file:
#     supervised_annotation = pickle.load(file)

# =============================================================================
# Defining functions for future plotting
# =============================================================================


def filter_out_other_behavior(supervised_annotation, filter_out):
    '''
    Parameters
    ----------
    supervised_annotation: data.TableDict from DeepOF
    filter_out: str (huddle, lookaround, etc.)
    '''

    copy_supervised_annotation = copy.deepcopy(supervised_annotation)

    for key, df in copy_supervised_annotation.items():
        mask = df[filter_out] == 1
        df[mask] = np.nan
        copy_supervised_annotation[key] = df

    return copy_supervised_annotation


def data_to_plot_mask(directory_output, conditions_cols, specific_conditions):
    '''
    Parameters
    ----------
    directory_output: str
    conditions: list (names of the columns from the conditions.csv)
    specific_conditions: list (names of the specific value of the conditions)
    '''
    conditions = pd.read_csv(directory_output + "conditions.csv")

    conditions_dict = dict(zip(conditions_cols, specific_conditions))

    mask_list = []
    for key, value in conditions_dict.items():
        mask = np.array(conditions[key] == value)
        mask_list.append(mask)

    global_mask = mask_list[0]
    for mask in mask_list[1:]:
        global_mask &= mask

    experiment_ids = conditions[global_mask]['experiment_id'].to_list()

    return experiment_ids


def data_set_to_plot(supervised_annotation, directory_output, conditions_cols, specific_conditions, behavior, length=360, bin_size=10, filter_out=''):
    '''
    Parameters
    ----------
    supervised_annotation: data.TableDict from DeepOF
    directory_output: str
    conditions: list (names of the columns from the conditions.csv)
    specific_conditions: list (names of the specific value of the conditions)
    behavior: str (huddle, lookaround, etc.)
    length: int (expected size of the video in seconds)
    bin_size: int (bin size in seconds)
    filter_out: str (do you want to filter out? yes or no)
    '''

    if filter_out != '':
        copy_supervised_annotation = filter_out_other_behavior(
            supervised_annotation, filter_out)
    else:
        copy_supervised_annotation = copy.deepcopy(supervised_annotation)

    experiment_ids = data_to_plot_mask(
        directory_output, conditions_cols, specific_conditions)
    dict_of_dataframes = {
        key: value for key, value in copy_supervised_annotation.items() if key in experiment_ids}

    num_of_bins = {}
    for key, value in dict_of_dataframes.items():
        factor = int(length/bin_size)

        value.reset_index(inplace=True)
        value.drop('index', axis=1, inplace=True)
        value.reset_index(inplace=True)

        bin_length = int(len(value) // factor)
        cutoffs = [i * bin_length for i in range(1, factor)]

        # Determine the minimum and maximum of the 'index' column
        min_value = value['index'].min()
        max_value = value['index'].max() + 1

        # Add the leftmost and rightmost edges for the first and last bins
        cutoffs = [min_value] + cutoffs + [max_value]

        value['bin'] = pd.cut(value['index'], bins=cutoffs,
                              labels=False, right=False, include_lowest=True)

        num_of_bins[key] = value

    df = pd.concat(num_of_bins.values(), keys=num_of_bins.keys()).reset_index()

    mean_values = df.groupby(['bin', 'level_0'])[behavior].mean()
    mean_values = mean_values.reset_index()
    mean_values['bin'] = mean_values['bin'] + 1

    # The numbers of the Y axis will be expressed as %
    mean_values[behavior] = mean_values[behavior] * 100

    return mean_values


def calculate_discrimination(data1, data2):

    if len(data1) == len(data2):
        subtraction = [x - y for x, y in zip(data2, data1)]
        addition = [x + y for x, y in zip(data2, data1)]
        discrimination = [x / y for x, y in zip(subtraction, addition)]

    return discrimination


def calculate_global_discrimination(data1, data2, data3, data4):

    if len(data1) == len(data2):
        subtraction = [x - y for x, y in zip(data2, data1)]
        addition = [x + y for x, y in zip(data2, data1)]
        discrimination_1 = [x / y for x, y in zip(subtraction, addition)]

    if len(data2) == len(data3):
        subtraction = [x - y for x, y in zip(data2, data3)]
        addition = [x + y for x, y in zip(data2, data3)]
        discrimination_2 = [x / y for x, y in zip(subtraction, addition)]

    if len(data3) == len(data4):
        subtraction = [x - y for x, y in zip(data4, data3)]
        addition = [x + y for x, y in zip(data4, data3)]
        discrimination_3 = [x / y for x, y in zip(subtraction, addition)]

    combined_lists = zip(discrimination_1, discrimination_2, discrimination_3)
    mean_list = [sum(x) / len(x) for x in combined_lists]

    return mean_list


def count_function(my_list):

    total = sum(1 for num in my_list)
    count_non = sum(1 for num in my_list if -1 <= num < 0.1)
    count_poor = sum(1 for num in my_list if 0.1 <= num < 0.2)
    count_average = sum(1 for num in my_list if 0.2 <= num < 0.3)
    count_good = sum(1 for num in my_list if 0.3 <= num < 0.4)
    count_excellent = sum(1 for num in my_list if 0.4 <= num < 1)
    # values = [count_non, count_poor, count_average, count_good, count_excellent]
    values = [count_excellent, count_good,
              count_average, count_poor, count_non]
    values = [x/total for x in values]
    values = [x*100 for x in values]

    # values_dict = dict(zip(tags, values))
    # for key, value in list(values_dict.items()):
    #     if value == 0:
    #         del values_dict[key]

    return values


def convert_pvalue_to_asterisks(pvalue):
    ns = "ns (p=" + str(pvalue)[1:4] + ")"
    if pvalue <= 0.0001:
        return "∗∗∗∗"
    elif pvalue <= 0.001:
        return "∗∗∗"
    elif pvalue <= 0.01:
        return "∗∗"
    elif pvalue <= 0.05:
        return "∗"
    return ns


# =============================================================================
# Plot functions
# =============================================================================

def timeseries(supervised_annotation, directory_output, column='huddle', color_contrast=red, ax=None):
    '''
    Parameters
    ----------
    supervised_annotation: data.TableDict from DeepOF
    directory_output: str
    column: str (huddle, lookaround, etc.)
    '''

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))

    sns.set_theme(style="whitegrid")
    ax.yaxis.grid(True)
    ax.xaxis.grid(False)

    label_offset = 0.2  # Offset for label positioning

    data1 = data_set_to_plot(supervised_annotation, directory_output, [
                             'protocol', 'group'], ['light', '3order'], column)
    data1['bin'] = data1['bin'] / 6
    sns.lineplot(x=data1['bin'], y=data1[column],
                 label='', legend=None, color=red)
    ax.text(data1['bin'].iloc[-1] + label_offset, data1[data1.bin == data1.bin.iloc[-1]]
            [column].mean(), 'light', fontsize=12, color=red, weight='bold')

    data2 = data_set_to_plot(supervised_annotation, directory_output, [
                             'protocol', 'group'], ['tone', '3order'], column)
    data2['bin'] = data2['bin'] / 6
    sns.lineplot(x=data2['bin'], y=data2[column],
                 label='', legend=None, color=blue)
    ax.text(data2['bin'].iloc[-1] + label_offset, data2[data2.bin == data2.bin.iloc[-1]]
            [column].mean(), 'tone', fontsize=12, color=blue, weight='bold')

    upper_limit = 100
    plt.ylim(0, upper_limit)
    # Add % symbol to the Y axis
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.set_xlabel('Minutes', loc='left')
    ax.set_ylabel('Percentage of time doing ' + column, loc='top')
    # ax.set_ylabel('Speed (mm/frame)', loc='top')

    plt.title(column.capitalize() + " response in young-adult males",
              loc='left', color=grey)

    # Grey color
    ax.xaxis.label.set_color(grey)
    ax.yaxis.label.set_color(grey)
    ax.tick_params(axis='x', colors=grey)
    ax.tick_params(axis='y', colors=grey)
    # for spine in plt.gca().spines.values():
    #     spine.set_edgecolor(grey)

    # Shaded area
    probetest_list = []
    probetest_coords = plt.Rectangle((3, 0), 1, upper_limit)
    probetest_list.append(probetest_coords)
    probetest_coords_2 = plt.Rectangle((5, 0), 1, upper_limit)
    probetest_list.append(probetest_coords_2)
    probetest_coll = PatchCollection(probetest_list, alpha=0.1, color=grey)
    ax.add_collection(probetest_coll)
    probetest_coll_border = PatchCollection(
        probetest_list, facecolor='none', edgecolor=grey, alpha=0.5)
    ax.add_collection(probetest_coll_border)
    ax.annotate('Cue', (3.5, upper_limit*0.9), ha='center',
                fontsize=12, color=grey, weight='bold', alpha=0.5)
    ax.annotate('Cue', (5.5, upper_limit*0.9), ha='center',
                fontsize=12, color=grey, weight='bold', alpha=0.5)

    plt.tight_layout()
    return ax


def boxplot(supervised_annotation, directory_output, column, learning, group, color_contrast, filter_outlier=True, ax=None):

    if ax is None:
        fig, ax = plt.subplots(figsize=(3.5, 4))
        
    sns.set_theme(style="whitegrid")
    ax.yaxis.grid(True)
    ax.xaxis.grid(False)
    
    jitter = 0.15  # Dots dispersion
    my_bins = [3,4] # The bin starts with 1 (i.e., 1, 2, 3, 4, etc.)
    
    dfs = []
    outlier_ids = []
    for my_bin in my_bins:
        data = data_set_to_plot(supervised_annotation, directory_output, [
                                'learning', 'group'], [learning, group], column, bin_size=60)
        
        data = data[data.bin == my_bin][column].tolist() 
        
        df = pd.DataFrame()
        df['values'] = data
        df['group'] = group
        df['id'] = df.index
        df['bin'] = my_bin
        mean = df['values'].mean()
        std_dev = df['values'].std()
        df['z_scores'] = np.abs((df['values'] - mean) / std_dev)
        df['outlier'] = df['z_scores'] > 2
        
        for index, row in df.iterrows():
            if row['outlier']:
                outlier_ids.append(row['id'])

        dfs.append(df)
    
    df = pd.concat(dfs, axis=0)
       
    # Remove rows whose IDs are listed in outlier_ids
    if filter_outlier is True:
        outlier_ids = list(set(outlier_ids))
        df = df[~df['id'].isin(outlier_ids)]
    
    data_1 = list(df[df['bin'] == 3]['values'])
    data_2 = list(df[df['bin'] == 4]['values'])
    
    # Perform statistics
    normality = pg.normality(df, 'values', 'bin')
    if normality['normal'].all():
        bartlett = pg.homoscedasticity(df, 'values', 'bin', method='bartlett')
        levene = pg.homoscedasticity(df, 'values', 'bin', method='levene')
        if bartlett['equal_var'].all() or levene['equal_var'].all():
            ttest = pg.ttest(data_1,
                             data_2,
                             correction=False,
                             paired=True)
            pval = ttest['p-val'][0]
            my_test = 'paired ttest'
        else:
            wilcoxon = pg.wilcoxon(data_1,
                                   data_2)
            pval = wilcoxon['p-val'][0]
            my_test = 'wilcoxon'
    else:
        wilcoxon = pg.wilcoxon(data_1, data_2)
        pval = wilcoxon['p-val'][0]
        my_test = 'wilcoxon'
    
    positions = []
    dispersion_values_list = []
    for my_bin in my_bins:
        position = my_bins.index(my_bin)
        positions.append(position)
        new_data = df[df['bin'] == my_bin]['values'].tolist()
        
        data_mean = np.mean(new_data)
        data_error = np.std(new_data, ddof=1)

        ax.hlines(data_mean, xmin=position-0.25,
                  xmax=position+0.25, color=grey, linewidth=1.5)
        ax.errorbar(position, data_mean, yerr=data_error,
                    lolims=False, capsize=3, ls='None', color=grey, zorder=-1)

        dispersion_values = np.random.normal(loc=position, scale=jitter, size=len(new_data)).tolist()
        dispersion_values_list.append(dispersion_values)
        ax.plot(dispersion_values, new_data,
                'o',
                markerfacecolor=color_contrast,
                markeredgecolor=color_contrast,
                markeredgewidth=1,
                markersize=5,
                label=my_bin)

    ax.set_xticks(positions)
    ax.set_xticklabels(['Before the cue', 'During the cue'])
    
    
    if len(data_1) == len(data_2):
        for x in range(len(df[df['bin'] == 3])):
            ax.plot([dispersion_values_list[0][x], dispersion_values_list[1][x]],
                    [data_1[x], data_2[x]], 
                    color=grey, linestyle='--', linewidth=0.5)

    plt.ylim(0, 100)
    ax.set_xlabel('')
    ax.set_ylabel(column.capitalize() + ' (%)', loc='top')
    # ax.set_ylabel('Speed (mm/frame)' + column, loc='top')
    ax.yaxis.set_major_formatter(mtick.PercentFormatter()) # Add % symbol to the Y axis

    plt.title(f'Young-adult males ({my_test})', loc='left', color=grey)

    # Grey color
    ax.xaxis.label.set_color(grey)
    ax.yaxis.label.set_color(grey)
    ax.tick_params(axis='x', colors=grey)
    ax.tick_params(axis='y', colors=grey)

    y, h, col = max(max(data_1), max(data_2)) + 5, 2, grey

    ax.plot([positions[0], positions[0], positions[1], positions[1]], [y, y+h, y+h, y], lw=1.5, c=col)
    
    ax.text((positions[0] + positions[1])*.5, y+2*h, convert_pvalue_to_asterisks(pval),
            ha='center', va='bottom', color=col, size=10, family='Arial Unicode MS')

    plt.tight_layout()
    return ax


def discrimination_index(supervised_annotation, directory_output, column, learning, group, color_contrast, ax=None):

    if ax is None:
        fig, ax = plt.subplots(figsize=(2.5, 4))

    sns.set_theme(style="whitegrid")
    ax.yaxis.grid(True)
    ax.xaxis.grid(False)

    data_position = 0

    data1 = data_set_to_plot(supervised_annotation, directory_output, [
                             'learning', 'group'], [learning, group], column, bin_size=60)
    # The bin starts with 1 (i.e., 1, 2, 3, 4, etc.)
    data1 = data1[data1.bin == 3]
    data1 = data1[column].tolist()

    data2 = data_set_to_plot(supervised_annotation, directory_output, [
                             'learning', 'group'], [learning, group], column, bin_size=60)
    # The bin starts with 1 (i.e., 1, 2, 3, 4, etc.)
    data2 = data2[data2.bin == 4]
    data2 = data2[column].tolist()

    # data3 = data_set_to_plot(supervised_annotation, directory_output, ['protocol','group'], ['s2','shock'], column, bin_size=60)
    # data3 = data3[data3.bin == 5] # The bin starts with 1 (i.e., 1, 2, 3, 4, etc.)
    # data3 = data3[column].tolist()

    # data4 = data_set_to_plot(supervised_annotation, directory_output, ['protocol','group'], ['s2','shock'], column, bin_size=60)
    # data4 = data4[data4.bin == 6] # The bin starts with 1 (i.e., 1, 2, 3, 4, etc.)
    # data4 = data4[column].tolist()

    # To calculate the discrimination index (2-3 vs 3-4)
    discrimination = calculate_discrimination(data1, data2)
    # discrimination = calculate_discrimination(data2, data1)

    # To calculate the global discrimination index (2-3 vs 3-4 vs 4-5 vs 5-6)
    # discrimination = calculate_global_discrimination(data1, data2, data3, data4)

    data_mean = np.mean(discrimination)
    data_error = np.std(discrimination, ddof=1)

    ax.hlines(data_mean, xmin=data_position-0.1,
              xmax=data_position+0.1, color=grey, linewidth=1.5)

    ax.errorbar(data_position, data_mean, yerr=data_error,
                lolims=False, capsize=3, ls='None', color=grey, zorder=-1)

    ax.set_xticks([data_position])
    ax.set_xticklabels([])

    jitter = 0.05  # Dots dispersion

    dispersion_values_data = np.random.normal(
        loc=data_position, scale=jitter, size=len(discrimination)).tolist()
    ax.plot(dispersion_values_data, discrimination,
            'o',
            markerfacecolor=color_contrast,
            markeredgecolor=color_contrast,
            markeredgewidth=1,
            markersize=5,
            label='Data1')

    plt.ylim(-1, 1)
    ax.axhline(y=0, color=grey, linestyle='--')
    ax.set_xlabel('')
    ax.set_ylabel('Discrimination Index ' + column.capitalize(), loc='top')

    # plt.title(column.capitalize() + " DI in TRAP2", loc = 'left', color=grey)

    # Grey color
    ax.xaxis.label.set_color(grey)
    ax.yaxis.label.set_color(grey)
    ax.tick_params(axis='x', colors=grey)
    ax.tick_params(axis='y', colors=grey)

    plt.tight_layout()
    return ax


def discrimination_index_summary(supervised_annotation, directory_output, column, learning, group, color_contrast, ax=None):

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 2))

    sns.set_theme(style="whitegrid")
    ax.yaxis.grid(True)
    ax.xaxis.grid(False)

    data1 = data_set_to_plot(supervised_annotation, directory_output, [
                             'learning', 'group'], [learning, group], column, bin_size=60)
    # The bin starts with 1 (i.e., 1, 2, 3, 4, etc.)
    data1 = data1[data1.bin == 3]
    data1 = data1[column].tolist()

    data2 = data_set_to_plot(supervised_annotation, directory_output, [
                             'learning', 'group'], [learning, group], column, bin_size=60)
    # The bin starts with 1 (i.e., 1, 2, 3, 4, etc.)
    data2 = data2[data2.bin == 4]
    data2 = data2[column].tolist()

    # data3 = data_set_to_plot(supervised_annotation, directory_output, ['protocol','group'], ['s2','shock'], column, bin_size=60)
    # data3 = data3[data3.bin == 5] # The bin starts with 1 (i.e., 1, 2, 3, 4, etc.)
    # data3 = data3[column].tolist()

    # data4 = data_set_to_plot(supervised_annotation, directory_output, ['protocol','group'], ['s2','shock'], column, bin_size=60)
    # data4 = data4[data4.bin == 6] # The bin starts with 1 (i.e., 1, 2, 3, 4, etc.)
    # data4 = data4[column].tolist()

    # To calculate the discrimination index (2-3 vs 3-4)
    discrimination = calculate_discrimination(data1, data2)
    # discrimination = calculate_discrimination(data2, data1)

    # To calculate the global discrimination index (2-3 vs 3-4 vs 4-5 vs 5-6)
    # discrimination = calculate_global_discrimination(data1, data2, data3, data4)

    values = count_function(discrimination)

    bar_height = 0.3
    categories = [0]

    bar1 = ax.barh(categories, np.array(values)[
                   0], bar_height, left=0, align='center', color=color_contrast, label='>0.4')
    bar2 = ax.barh(categories, np.array(values)[1], bar_height, left=np.array(
        values)[0], align='center', color=color_contrast, label='0.3-0.4')
    bar3 = ax.barh(categories, np.array(values)[2], bar_height, left=np.array(values)[
                   0] + np.array(values)[1], align='center', color=color_contrast, label='0.2-0.3')
    bar4 = ax.barh(categories, np.array(values)[3], bar_height, left=np.array(values)[
                   0] + np.array(values)[1] + np.array(values)[2], align='center', color=grey, label='0.1-0.2')
    bar5 = ax.barh(categories, np.array(values)[4], bar_height, left=np.array(values)[
                   0] + np.array(values)[1] + np.array(values)[2] + np.array(values)[3], align='center', color=grey, label='<0.1')

    ax.set_yticks([])  # Hide y-axis ticks

    ax.set_xlabel('% of animals', loc='left')
    plt.xlim(0, 100)
    # Add % symbol to the X axis
    ax.xaxis.set_major_formatter(mtick.PercentFormatter())

    # Move x-axis ticks and label to the top
    # ax.xaxis.tick_top()
    # ax.xaxis.set_label_position('top')

    # Remove the frame around the plot
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

    # Grey color
    ax.xaxis.label.set_color(grey)
    ax.yaxis.label.set_color(grey)
    ax.tick_params(axis='x', which='both', bottom=True, colors=grey)
    ax.tick_params(axis='y', colors=grey)

    plt.title('Distribution of Discrimination Index (' + column.capitalize() +
              ") among young-adult males", loc='left', color=grey)

    # Add labels inside the bars
    for bars, label in zip([bar1, bar2, bar3, bar4, bar5], ['DI\nMore than 0.4', 'DI\n0.3 to 0.4', 'DI\n0.2 to 0.3', 'DI\n0.1 to 0.2', 'DI\nLess than 0.1']):
        for bar in bars:
            if bar.get_width() != 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_y() + bar.get_height() / 2,
                        label, color='white', ha='center', va='center', fontsize=9)

    plt.tight_layout()
    return ax

# =============================================================================

def boxplot_anova(supervised_annotation, directory_output, column='huddle', ax=None):

    learning = 'direct'
    color_contrast = red
    groups = ['paired', 'unpaired', 'no-shock']

    if ax is None:
        fig, ax = plt.subplots(figsize=(3.5, 4))

    sns.set_theme(style="whitegrid")
    ax.yaxis.grid(True)
    ax.xaxis.grid(False)
    jitter = 0.15  # Dots dispersion

    dfs = []
    for group in groups:
        data = data_set_to_plot(supervised_annotation, directory_output, [
                                'learning', 'group'], [learning, group], column, bin_size=60)
        
        # To calculate just the bin_4
        # data = data[data.bin == 2][column].tolist() # The bin starts with 1 (i.e., 1, 2, 3, 4, etc.)
        
        # To calculate the discrimination index
        data_3 = data[data.bin == 3][column].tolist()
        data_4 = data[data.bin == 4][column].tolist()
        data = [(y-x)/(y+x) for x, y in zip(data_3, data_4)]
        
        df = pd.DataFrame()
        df['values'] = data
        df['group'] = group
        mean = df['values'].mean()
        std_dev = df['values'].std()
        df['z_scores'] = np.abs((df['values'] - mean) / std_dev)
        df['outlier'] = df['z_scores'] > 2
        
        dfs.append(df)
    
    df = pd.concat(dfs, axis=0)
    df = df[df['outlier'] != True] 
    
    # Perform statistics
    normality = pg.normality(df, 'values', 'group')
    if normality['normal'].all():
        bartlett = pg.homoscedasticity(df, 'values', 'group', method='bartlett')
        levene = pg.homoscedasticity(df, 'values', 'group', method='levene')
        if bartlett['equal_var'].all() or levene['equal_var'].all():
            anova = pg.anova(df, 'values', 'group')
            my_test = 'anova'
            pval = anova['p-unc'][0]
            if pval <= 0.05:
                print(pg.pairwise_tukey(df, 'values', 'group'))
        else:
            welch_anova = pg.welch_anova(df, 'values', 'group')
            my_test = 'welch-anova'
            pval = welch_anova['p-unc'][0]
            if pval <= 0.05:
                print(pg.pairwise_gameshowell(df, 'values', 'group'))
    else:
        kruskal = pg.kruskal(df, 'values', 'group')
        pval = kruskal['p-unc'][0]
        my_test = 'kruskal'
        if pval <= 0.05:
            print(pg.pairwise_tests(df, 'values', 'group', parametric=False))
    
    positions = []
    for group in groups:
        position = groups.index(group)
        positions.append(position)
        new_data = df[df['group'] == group]['values'].tolist()
        
        data_mean = np.mean(new_data)
        data_error = np.std(new_data, ddof=1)
        # y = max(new_data) + 5
        # ax.text(position, y, stats_dict[group], ha='center', va='bottom', color=grey, size=10)

        ax.hlines(data_mean, xmin=position-0.25,
                  xmax=position+0.25, color=grey, linewidth=1.5)
        ax.errorbar(position, data_mean, yerr=data_error,
                    lolims=False, capsize=3, ls='None', color=grey, zorder=-1)

        dispersion_values = np.random.normal(loc=position, scale=jitter, size=len(new_data)).tolist()
        ax.plot(dispersion_values, new_data,
                'o',
                markerfacecolor=color_contrast,
                markeredgecolor=color_contrast,
                markeredgewidth=1,
                markersize=5,
                label=group)
    
    ax.set_xticks(positions)
    ax.set_xticklabels(groups)

    plt.ylim(-1,1)
    ax.set_xlabel('')
    ax.set_ylabel(column.capitalize() + ' (D.I.)', loc='top')
    # ax.set_ylabel('Speed (mm/frame)' + column, loc='top')
    # Add % symbol to the Y axis
    # ax.yaxis.set_major_formatter(mtick.PercentFormatter())

    plt.title(f'Young-adult males ({my_test})', loc='left', color=grey)


    # Grey color
    ax.xaxis.label.set_color(grey)
    ax.yaxis.label.set_color(grey)
    ax.tick_params(axis='x', colors=grey)
    ax.tick_params(axis='y', colors=grey)

    plt.tight_layout()
    return ax

# =============================================================================

def iterate_plot_function(supervised_annotation, directory_output, column='climbing'):

    directory = '//folder/becell/Lab Projects/ERCstG_HighMemory/Data/Marc/1) SOC/2023-07a08 SOC Controls_males/DeepOF analysis/'
    learnings = ['direct', 'mediated']
    groups = ['paired', 'unpaired', 'no-shock']

    for learning in learnings:
        if learning == 'direct':
            color = '#801946'
        elif learning == 'mediated':
            color = '#194680'

        for group in groups:

            ax = boxplot(supervised_annotation, directory_output, column, learning, group, color)
            boxplot_tag = 'boxplot_' + column + '_' + group + '_' + learning
            boxplot_path = os.path.join(directory, f'{boxplot_tag}.png')
            plt.savefig(boxplot_path)
            plt.close()

            # ax = discrimination_index(supervised_annotation, directory_output, column, learning, group, color)
            # di_tag = 'di_' + column + '_' + group + '_' + learning
            # di_path = os.path.join(directory, f'{di_tag}.png')
            # plt.savefig(di_path)
            # plt.close()

            # ax = discrimination_index_summary(
            #     supervised_annotation, directory_output, column, learning, group, color)
            # barplot_tag = 'barplot_' + column + '_' + group + '_' + learning
            # barplot_path = os.path.join(directory, f'{barplot_tag}.png')
            # plt.savefig(barplot_path)
            # plt.close()

# =============================================================================
# Statistics functions
# =============================================================================


def generate_statistics_df(supervised_annotation, directory_output, column):

    learnings = ['direct', 'mediated']
    groups = ['paired', 'unpaired', 'no-shock']
    datas = []

    for learning in learnings:
        for group in groups:
            for x in [3, 4]:
                tag = learning + '_' + group + '_' + str(x)
                data1 = data_set_to_plot(supervised_annotation, directory_output, [
                                         'learning', 'group'], [learning, group], column, bin_size=60)
                # The bin starts with 1 (i.e., 1, 2, 3, 4, etc.)
                data1 = data1[data1.bin == x]
                data1['group'] = tag
                datas.append(data1)

    result_df = pd.concat(datas, axis=0, ignore_index=True)
    result_df = result_df.reset_index(drop=True)

    result_df[['learning', 'group', 'timepoint']
              ] = result_df['group'].str.split('_', expand=True)

    return result_df

# import pingouin as pg
# df = generate_statistics_df(supervised_annotation, directory_output, column='speed')
# pg.normality(df, dv='speed', group='group', method="shapiro")


def compare_timepoints(supervised_annotation, directory_output, factor1_col='learning', factor2_col='group', timepoint_col='timepoint', value_col='huddle'):

    df = generate_statistics_df(
        supervised_annotation, directory_output, value_col)

    # Get unique levels of Factor1 and Factor2
    factor1_levels = df[factor1_col].unique()
    factor2_levels = df[factor2_col].unique()

    # Initialize a dictionary to store results
    results_dict = {'Factor1': [], 'Factor2': [],
                    'T-statistic': [], 'P-value': []}
    decimal_places = 4

    # Iterate through unique combinations of Factor1 and Factor2
    for factor1_level in factor1_levels:
        for factor2_level in factor2_levels:
            # Select data for the current combination of Factor1 and Factor2
            subset = df[(df[factor1_col] == factor1_level) &
                        (df[factor2_col] == factor2_level)]

            # Extract values for before and after timepoints
            before_values = subset[subset[timepoint_col] == '3'][value_col]
            after_values = subset[subset[timepoint_col] == '4'][value_col]

            # Initialize t_stat and p_value variables
            t_stat, p_value = None, None

            # Perform statistical test (e.g., paired t-test or Wilcoxon signed-rank test)
            # Use t-test for normally distributed data, and Wilcoxon test otherwise
            if before_values.size >= 2 and after_values.size >= 2:
                # Assuming normal distribution for simplicity (you may need to check this)
                # t_stat, p_value = ttest_rel(before_values, after_values)
                # Wilcoxon signed-rank test for non-normally distributed data
                _, p_value = wilcoxon(before_values, after_values)

            # Append results to the dictionary, rounding p-value to specified decimal places
            results_dict['Factor1'].append(str(factor1_level))
            results_dict['Factor2'].append(str(factor2_level))
            results_dict['T-statistic'].append(t_stat)
            results_dict['P-value'].append(round(p_value, decimal_places))

    # Create a DataFrame from the results dictionary
    results_df = pd.DataFrame(results_dict)

    return results_df
