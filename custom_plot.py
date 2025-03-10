import numpy as np
import matplotlib.pyplot as plt
from geometry import rot

# Plotting positions and velocities of data samples
def plot_positions_velocities_with_fixed_bounds(Y, link_lengths):
    num_samples = Y.shape[0]
    num_links = len(link_lengths)
    # Splitting joint angles and velocities
    joint_angles = Y[:, :num_links]
    joint_velocities = Y[:, num_links:]
    # Dictionary to store end positions
    end_positions = {i: np.zeros((num_samples, 2)) for i in range(num_links)}  
    for i in range(num_samples):
        # Extract joint angles for the sample
        q = joint_angles[i, :]  
        # Starting at the origin
        p = np.zeros((2, 1))  
        # Starting with no rotation
        R = np.eye(2)
        for j in range(num_links):
            # Update rotation matrix based on joint angle
            R = np.dot(R, rot(q[j]))  
            l = np.zeros((2, 1))
            # Length of current link
            l[0, 0] = link_lengths[j]  
            # Calculate end position of the current link
            p_next = p + np.dot(R, l)  
            # Store end position
            end_positions[j][i, :] = p_next.T  
            # Update current position for next link
            p = p_next
    # Plotting positions with green dots and fixed -3 to 3 range, and correcting outer bound circles
    fig, axs = plt.subplots(2, num_links, figsize=(18, 12))
    for i in range(num_links):
        # Calculate maximum reach for the current link and all preceding links
        current_max_reach = np.sum(link_lengths[:i+1])
        # Plot positions with within [-3,3]
        axs[0, i].scatter(end_positions[i][:, 0], end_positions[i][:, 1], alpha=0.5, s=10, color='green', label=f'Link {i+1} End Positions')
        axs[0, i].set_xlim(-3, 3)
        axs[0, i].set_ylim(-3, 3)
        axs[0, i].add_patch(plt.Circle((0, 0), current_max_reach, color='r', fill=False, linestyle='--'))
        axs[0, i].set_aspect('equal', 'box')
        axs[0, i].set_title(f'Link {i+1} End Position Distribution')
        axs[0, i].legend(loc='upper right')
        # Plot velocity distributions
        axs[1, i].hist(joint_velocities[:, i], bins=20, alpha=0.7, color='blue', label=f'Joint {i+1} Velocities')
        axs[1, i].set_title(f'Joint {i+1} Velocity Distribution')
        axs[1, i].legend()
        axs[1, i].set_xlabel('Velocity')
        axs[1, i].set_ylabel('Frequency')
    plt.tight_layout()
    plt.show()
    # plt.savefig('training_data_visualization.png', dpi=300)