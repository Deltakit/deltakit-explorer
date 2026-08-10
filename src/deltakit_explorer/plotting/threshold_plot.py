import matplotlib.pyplot as plt
import numpy as np

def create_threshold_plot(data_dict, estimated_threshold, title="Surface Code Threshold Crossing"):
    """Generates a complex log-log plot with an inset of the threshold crossover."""
    # Create the figure and main axes
    fig, ax = plt.subplots(figsize=(12, 8))

    # Define standard colors for distances
    colors = {3: '#440154', 5: '#7ad151'} # Dark purple, light green

    # --- Main Plot ---
    for distance, (p_vals, lep_vals) in sorted(data_dict.items()):
        color = colors.get(distance, 'black')
        ax.plot(
            p_vals, 
            lep_vals, 
            marker='o', 
            linestyle='-', 
            color=color, 
            markersize=6, 
            label=f'd = {distance}'
        )
        
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Physical Error Rate (p)', fontsize=12)
    ax.set_ylabel('Logical Error Probability (LEP)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, which="both", ls="--", color='grey', alpha=0.3)
    ax.legend(loc='upper left', fontsize=11)

    # This locks it strictly to the bottom right quadrant.
    ax_ins = ax.inset_axes([0.52, 0.08, 0.45, 0.35]) 
    
    for distance, (p_vals, lep_vals) in sorted(data_dict.items()):
        color = colors.get(distance, 'black')
        ax_ins.plot(
            p_vals, 
            lep_vals, 
            marker='o', 
            linestyle='-', 
            color=color, 
            markersize=4
        )

    # Define the zoom area
    zoom_xlim = [estimated_threshold * 0.9, estimated_threshold * 1.1]
    all_lep_near_threshold = []
    for distance, (p_vals, lep_vals) in data_dict.items():
        for p, lep in zip(p_vals, lep_vals):
            if zoom_xlim[0] <= p <= zoom_xlim[1]:
                all_lep_near_threshold.append(lep)
    
    if all_lep_near_threshold:
        zoom_ylim = [min(all_lep_near_threshold) * 0.8, max(all_lep_near_threshold) * 1.2]
    else:
        zoom_ylim = [1e-3, 1e-1]
    
    ax_ins.set_xscale('log')
    ax_ins.set_yscale('log')
    ax_ins.set_xlim(zoom_xlim)
    ax_ins.set_ylim(zoom_ylim)
    ax_ins.grid(True, which="both", ls="--", color='grey', alpha=0.3)

    ax_ins.axvline(
        x=estimated_threshold, 
        color='#b2182b', 
        linestyle='--', 
        label=f'Threshold: {estimated_threshold:.5f}'
    )
    
    ax_ins.legend(loc='upper right', frameon=True, facecolor='white', fontsize=9) 

    ax.indicate_inset_zoom(ax_ins, edgecolor="gray")

    plt.show()
    
    return plt

def generate_realistic_threshold_data(threshold_p, distances=[3, 5], points_per_distance=20):
    """Generates more realistic surface code threshold data for testing."""
    data = {}
    
    p_log = np.logspace(np.log10(threshold_p * 0.1), np.log10(threshold_p * 10), points_per_distance * 2)
    p_linear_near_threshold = np.linspace(threshold_p * 0.8, threshold_p * 1.2, points_per_distance)
    p_vals = np.sort(np.concatenate([p_log, p_linear_near_threshold]))

    for distance in distances:
        exponent_nu = 0.6 * np.log(distance + 1)
        lep_vals_clean = 1.0 / (1.0 + (p_vals / threshold_p)**exponent_nu)
        
        suppression_factor = np.exp(-0.05 * distance * (p_vals < threshold_p) * (threshold_p - p_vals) / threshold_p)
        lep_vals_suppressed = lep_vals_clean * suppression_factor
        
        above_suppression_factor = np.exp(0.01 * distance * (p_vals > threshold_p) * (p_vals - threshold_p) / threshold_p)
        lep_vals_modeled = lep_vals_suppressed * above_suppression_factor
        
        noise = 1e-4 * np.random.normal(0, 1.0, len(p_vals))
        lep_vals = lep_vals_modeled + noise
        
        lep_vals = np.clip(lep_vals, 1e-6, 0.9999)
        data[distance] = (p_vals, lep_vals)
        
    return data

if __name__ == "__main__":
    target_threshold = 0.00679 
    realistic_data = generate_realistic_threshold_data(
        target_threshold, 
        distances=[3, 5], 
        points_per_distance=20
    )
    
    create_threshold_plot(
        realistic_data, 
        target_threshold, 
        title="Surface Code Threshold Crossing"
    )
