import matplotlib.pyplot as plt
import numpy as np

def create_threshold_plot(data_dict, estimated_threshold):
    fig, ax = plt.subplots(figsize=(10, 7))
    distances = sorted(data_dict.keys())
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(distances)))
    
    for d, color in zip(distances, colors):
        p_vals, lep_vals = data_dict[d]
        ax.plot(p_vals, lep_vals, marker='o', linestyle='-', label=f'd = {d}', color=color)
        
    ax.set_xlabel("Physical Error Rate (p)", fontsize=12)
    ax.set_ylabel("Logical Error Probability (LEP)", fontsize=12)
    ax.set_title("Surface Code Threshold Crossing", fontsize=14, pad=15)
    ax.set_yscale('log') 
    ax.grid(True, which="both", ls="--", alpha=0.4)
    ax.legend(loc="upper left", fontsize=11)
    
    axins = ax.inset_axes([0.55, 0.15, 0.4, 0.4])
    
    for d, color in zip(distances, colors):
        p_vals, lep_vals = data_dict[d]
        axins.plot(p_vals, lep_vals, marker='o', linestyle='-', color=color)
        
    zoom_margin_x = estimated_threshold * 0.15 
    axins.set_xlim(estimated_threshold - zoom_margin_x, estimated_threshold + zoom_margin_x)
    axins.axvline(x=estimated_threshold, color='red', linestyle='--', 
                  label=f'Threshold: {estimated_threshold:.5f}')
    
    axins.set_yscale('log')
    axins.legend(fontsize=9, loc="upper right")
    ax.indicate_inset_zoom(axins, edgecolor="black")
    
    plt.tight_layout()
    plt.show()
