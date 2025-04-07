import matplotlib.pyplot as plt
import numpy as np
import os
import torch

def plot_spike_raster(spike_data, epoch, batch_idx, prefix="train"):
    """
    Generate professional spike raster plots for SNN layers
    with enhanced visualization features.
    
    Parameters:
        spike_data (dict): Dictionary containing layer-wise spike tensors
        epoch (int): Current training epoch
        batch_idx (int): Batch index in epoch
        prefix (str): Filename prefix for saving
        
    Returns:
        str: Path to saved visualization file
    """
    # Create visualization directory if needed
    os.makedirs("visualizations", exist_ok=True)
    
    # Setup figure with constrained layout
    fig = plt.figure(figsize=(12, 8), dpi=100)
    fig.suptitle(f"Spike Activity - Epoch {epoch} Batch {batch_idx}", 
                y=0.97, fontsize=10, fontweight='bold')
    
    # Create subplots for each layer
    axes = fig.subplots(3, 1, sharex=True, gridspec_kw={'hspace': 0.4})
    
    # Layer order and colors
    layers = ['fc1', 'fc2', 'fc3']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Matplotlib default color cycle
    
    for idx, (layer_name, ax) in enumerate(zip(layers, axes)):
        if layer_name not in spike_data:
            continue
            
        # Convert spikes to numpy array
        spikes = spike_data[layer_name]
        if isinstance(spikes, torch.Tensor):
            spikes = spikes.cpu().numpy()
            
        # Plot raster
        neuron_ids, timesteps = np.where(spikes.T > 0)  # Transpose for correct dims
        ax.scatter(timesteps, neuron_ids, 
                  color=colors[idx], 
                  marker='|', 
                  s=15,
                  alpha=0.7,
                  linewidth=0.8)
        
        # Formatting
        ax.set_title(f"{layer_name.upper()} Layer", fontsize=9, pad=4)
        ax.set_ylabel("Neuron ID", fontsize=7, labelpad=2)
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.tick_params(axis='both', which='major', labelsize=6)
        
        # Set limits
        ax.set_xlim(0, spikes.shape[0])
        ax.set_ylim(0, spikes.shape[1])
        
    # X-axis label only on bottom plot
    axes[-1].set_xlabel("Time Step", fontsize=8, labelpad=3)
    
    # Save and close
    filename = f"visualizations/{prefix}_spikes_e{epoch}_b{batch_idx}.png"
    plt.savefig(filename, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    
    return filename