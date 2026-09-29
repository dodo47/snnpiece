import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns

def plot_mosaic(output, layer_id, neuron_id, NaN_color = 'silver', datapoint_color = 'black', path = None):
    extracted_IDs = []
    for dp in output['neuron_IDs']:
        extracted_IDs.append(dp[layer_id][neuron_id])
    IDs = np.array(extracted_IDs).flatten()
    # Generate a random permutation for the values from 0 to 1000
    unique_labels = np.arange(-1, np.max(IDs)+1)  # Array of original labels (0 to 1000)
    shuffled_labels = np.random.permutation(unique_labels)  # Shuffled labels

    # Create a mapping dictionary from original labels to new labels
    label_mapping = dict(zip(unique_labels, shuffled_labels))

    # Remap the z values using the mapping dictionary
    z_remapped = np.array(np.vectorize(label_mapping.get)(IDs), dtype=int)
    filters = z_remapped != -1

    # Creating the plot
    plt.figure(figsize=(10, 10))

    # Using scatter plot for visualization (with colormap)
    scatter = plt.scatter(np.array(output['alpha_values']).flatten()[filters], np.array(output['beta_values']).flatten()[filters], c=z_remapped[filters], cmap='Spectral', s=2, alpha=0.75)
    scatter = plt.scatter(np.array(output['alpha_values']).flatten()[~filters], np.array(output['beta_values']).flatten()[~filters], c=NaN_color, s=2, alpha=0.75)

    # plt.scatter([0,0,1], [0,1,0], c=datapoint_color, s=40)

    # remove x and y ticks
    plt.xticks([])
    plt.yticks([])

    sns.despine(bottom=True, left=True, right=True, top=True)
    # save as png and svg with tight borders 
    if path is not None:
        plt.savefig('{}.png'.format(path), bbox_inches='tight', dpi=600)
    else:
        plt.show()

def plot_mosaic_points(ax, output, layer_id, msize, seed = 412345, NaN_color = 'silver', datapoint_color = 'black', path = None):
    np.random.seed(seed)
    IDs = np.array([dp[layer_id] for dp in output['layer_IDs']]).flatten()
    unique_labels = np.arange(-1, np.max(IDs)+1)
    shuffled_labels = np.random.permutation(unique_labels)  # Shuffled labels

    # Create a mapping dictionary from original labels to new labels
    label_mapping = dict(zip(unique_labels, shuffled_labels))

    # Remap the z values using the mapping dictionary
    z_remapped = np.array(np.vectorize(label_mapping.get)(IDs), dtype=int)
    filters = z_remapped != label_mapping[-1]

    scatter = ax.scatter(np.array(output['alpha_values']).flatten()[filters], np.array(output['beta_values']).flatten()[filters], c=z_remapped[filters], cmap='Spectral', s=msize, edgecolor='k', alpha=1)
    scatter = ax.scatter(np.array(output['alpha_values']).flatten()[~filters], np.array(output['beta_values']).flatten()[~filters], c=NaN_color, s=msize, alpha=1, edgecolor='k')

    ax.set_xticks([])
    ax.set_yticks([])

    sns.despine(ax=ax, bottom=True, left=True, right=True, top=True)

def plot_spike_time(output, layer_id, neuron_id, path = None, xlim = (0,1), ylim = None):
    spike_times = []
    neuron_state = []
    for i in range(len(output['spike_times'])):
        spike_times.append(output['spike_times'][i][layer_id][0][neuron_id])
        neuron_state.append(output['neuron_IDs'][i][layer_id][neuron_id])

    filter = np.array(neuron_state != -1)
    neuron_state = np.array(neuron_state)[filter]
    spike_times = np.array(spike_times)[filter]

    unique_labels = np.arange(np.max(neuron_state)+1)  # Array of original labels (0 to 1000)
    shuffled_labels = np.random.permutation(unique_labels)  # Shuffled labels

    # Create a mapping dictionary from original labels to new labels
    label_mapping = dict(zip(unique_labels, shuffled_labels))

    # Remap the z values using the mapping dictionary
    z_remapped = np.array(np.vectorize(label_mapping.get)(neuron_state), dtype=int)

    # Creating the plot
    plt.rc('ytick', labelsize=14)
    plt.rc('xtick', labelsize=14)
    plt.figure(figsize=(10, 5))

    # Using scatter plot for visualization (with colormap)
    scatter = plt.scatter(np.array(output['alpha_values'])[filter], np.array(spike_times), c=z_remapped, cmap='Spectral', s=2, alpha=0.75)
    plt.xlabel(r'$\alpha$', fontsize=16)
    plt.ylabel('Spike time', fontsize=16)

    plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)

    sns.despine(right=True, top=True)
    # save as png and svg with tight borders 
    if path is not None:
        plt.savefig('{}.png'.format(path), bbox_inches='tight')
    else:
        plt.show()