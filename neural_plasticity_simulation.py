import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
import time
import random

class NeuralPlasticitySimulation:
    def __init__(self, grid_size=50, initial_active_percent=0.8):
        """Initialize the neural grid simulation."""
        self.grid_size = grid_size
        
        # Create grid of neurons (1 = active, 0 = inactive)
        self.neurons = np.random.choice(
            [1, 0], 
            size=(grid_size, grid_size), 
            p=[initial_active_percent, 1-initial_active_percent]
        )
        
        # Functionality/influence map (represents how much function each neuron has)
        self.function_map = np.copy(self.neurons).astype(float)
        
        # Neuron types: 0=regular, 1=fast-adapting, 2=slow-adapting
        # Fast adapting neurons compensate more quickly but are more vulnerable
        # Slow adapting neurons are more resilient but compensate more slowly
        self.neuron_types = np.random.choice(
            [0, 1, 2], 
            size=(grid_size, grid_size), 
            p=[0.7, 0.15, 0.15]
        )
        
        # Health/resistance to damage (1.0 = fully healthy)
        self.health_map = np.ones((grid_size, grid_size)) * 0.8 + np.random.rand(grid_size, grid_size) * 0.2
        
        # Connection strength map (represents synaptic connections between neurons)
        self.connections = np.zeros((grid_size, grid_size))
        
        # Age of neurons (younger neurons adapt faster)
        self.age_map = np.random.rand(grid_size, grid_size) * 100
        
        # Stimulation map (areas receiving external stimulation)
        self.stimulation = np.zeros((grid_size, grid_size))
        
        # Time step counter
        self.time_step = 0
        
        # Recovery probability per time step
        self.recovery_rate = 0.001
        
        # Random damage event probability
        self.random_damage_prob = 0.005
        
        # Track stress in the system
        self.stress_level = 0
        
        # Custom colormap: black (inactive) -> blue (active) -> red (compensating)
        self.cmap = LinearSegmentedColormap.from_list(
            'neural_cmap', 
            [(0, 0, 0),      # Black for inactive neurons
             (0, 0, 1),      # Blue for normal active neurons
             (1, 0, 0)]      # Red for compensating neurons (function > 1)
        )
        
        # History tracking
        self.history = {
            'active': [],
            'inactive': [],
            'compensating': [],
            'recovered': []
        }
        self.total_recovered = 0
    
    def damage_region(self, center_x, center_y, radius, intensity=1.0):
        """Simulate damage to a region of neurons."""
        x, y = np.ogrid[:self.grid_size, :self.grid_size]
        # Create a circular mask
        mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
        
        # Apply damage based on health, neuron type and intensity
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if mask[i, j]:
                    # Slow adapting neurons (type 2) are more resistant to damage
                    resistance = 1.0
                    if self.neuron_types[i, j] == 2:
                        resistance = 1.5
                    
                    # Apply damage based on health and resistance
                    damage_probability = intensity / (self.health_map[i, j] * resistance)
                    if np.random.random() < damage_probability:
                        self.neurons[i, j] = 0
                        self.function_map[i, j] = 0
        
        damaged_count = np.sum(mask & (self.neurons == 0))
        print(f"Damaged {damaged_count} neurons in a region centered at ({center_x}, {center_y})")
        
        # Increase stress level based on damage
        self.stress_level += damaged_count / 100
    
    def apply_stimulation(self, center_x, center_y, radius, strength=0.5):
        """Apply external stimulation to a region, which promotes recovery."""
        x, y = np.ogrid[:self.grid_size, :self.grid_size]
        # Create a circular mask
        mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
        
        # Apply stimulation
        self.stimulation[mask] += strength
        
        print(f"Applied stimulation to region centered at ({center_x}, {center_y})")
    
    def update(self):
        """Update the neural grid for one time step to simulate plasticity."""
        self.time_step += 1
        recovered_count = 0
        
        # Random damage events based on probability
        if np.random.random() < self.random_damage_prob and self.time_step > 50:
            x = np.random.randint(0, self.grid_size)
            y = np.random.randint(0, self.grid_size)
            radius = np.random.randint(2, 5)
            self.damage_region(x, y, radius, intensity=0.7)
        
        # Random stimulation events (representing rehabilitation)
        if self.time_step % 20 == 0 and self.time_step > 30:
            # Find an area with damage to stimulate
            inactive_coords = np.where(self.neurons == 0)
            if len(inactive_coords[0]) > 0:
                # Randomly select one inactive neuron
                idx = np.random.randint(0, len(inactive_coords[0]))
                x, y = inactive_coords[0][idx], inactive_coords[1][idx]
                # Stimulate around it
                self.apply_stimulation(x, y, radius=3, strength=0.3)
        
        # Create a copy of the current state
        new_function_map = np.copy(self.function_map)
        new_health_map = np.copy(self.health_map)
        
        # Gradually decay stimulation
        self.stimulation *= 0.95
        
        # Gradually decrease stress
        self.stress_level *= 0.98
        
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if self.neurons[i, j] == 0:
                    # Degeneration: inactive neurons lose function over time
                    decay_rate = 0.95  # Base decay rate
                    new_function_map[i, j] *= decay_rate
                    
                    # Recovery chance based on stimulation and neighboring activity
                    neighbors = self._get_neighbors(i, j)
                    active_neighbors = sum(1 for x, y in neighbors if self.neurons[x, y] == 1)
                    
                    # Calculate recovery probability
                    recovery_boost = self.stimulation[i, j] * 0.1  # Stimulation increases recovery
                    neighbor_boost = active_neighbors * 0.001  # Active neighbors help recovery
                    age_factor = (100 - self.age_map[i, j]) / 100  # Younger neurons recover better
                    
                    recovery_prob = self.recovery_rate + recovery_boost + neighbor_boost + age_factor
                    
                    # Check for recovery
                    if np.random.random() < recovery_prob:
                        self.neurons[i, j] = 1
                        new_function_map[i, j] = 0.5  # Start with partial function
                        recovered_count += 1
                        
                else:  # Active neuron
                    # Get the neighboring cells
                    neighbors = self._get_neighbors(i, j)
                    
                    # If any neighbors are damaged/inactive, this neuron starts compensating
                    inactive_neighbors = [(x, y) for x, y in neighbors if self.neurons[x, y] == 0]
                    
                    if inactive_neighbors:
                        # Base compensation factor
                        base_factor = 0.005
                        
                        # Adjust based on neuron type
                        if self.neuron_types[i, j] == 1:  # Fast adapting
                            type_multiplier = 2.0
                        elif self.neuron_types[i, j] == 2:  # Slow adapting
                            type_multiplier = 0.7
                        else:  # Regular
                            type_multiplier = 1.0
                            
                        # Age affects adaptation speed
                        age_factor = (100 - self.age_map[i, j]) / 100 + 0.5
                        
                        # Stimulation enhances compensation
                        stim_factor = 1.0 + self.stimulation[i, j]
                        
                        # Final compensation calculation
                        compensation_factor = (
                            base_factor * 
                            len(inactive_neighbors) * 
                            type_multiplier * 
                            age_factor * 
                            stim_factor
                        )
                        
                        new_function_map[i, j] += compensation_factor
                        
                        # Over-compensation stress - neurons that compensate too much become stressed
                        if new_function_map[i, j] > 1.5:
                            stress = (new_function_map[i, j] - 1.5) * 0.01
                            new_health_map[i, j] -= stress
                    
                    # Health regeneration for active neurons
                    new_health_map[i, j] = min(1.0, new_health_map[i, j] + 0.0005)
                
                # Aging process
                self.age_map[i, j] += 0.01
        
        # Update function map and health
        self.function_map = new_function_map
        self.health_map = new_health_map
        
        # Update connections based on function
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if self.neurons[i, j] == 1:
                    neighbors = self._get_neighbors(i, j)
                    active_neighbors = [(x, y) for x, y in neighbors if self.neurons[x, y] == 1]
                    
                    # Strengthen connections between active neurons
                    for x, y in active_neighbors:
                        # Higher function neurons create stronger connections
                        conn_strength = min(self.function_map[i, j], self.function_map[x, y]) * 0.01
                        self.connections[i, j] += conn_strength
        
        # Track history
        stats = self.get_stats()
        self.history['active'].append(stats['active'])
        self.history['inactive'].append(stats['inactive'])
        self.history['compensating'].append(stats['compensating'])
        self.history['recovered'].append(recovered_count)
        self.total_recovered += recovered_count
        
        return recovered_count
    
    def _get_neighbors(self, i, j):
        """Get valid neighboring cells (handles edge cases)."""
        neighbors = []
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                if di == 0 and dj == 0:
                    continue  # Skip the cell itself
                
                ni, nj = i + di, j + dj
                if 0 <= ni < self.grid_size and 0 <= nj < self.grid_size:
                    neighbors.append((ni, nj))
        
        return neighbors
    
    def get_display_map(self, display_type='function'):
        """Return a map for display based on the selected type."""
        if display_type == 'function':
            # Normalize values for display (0 to 2, where >1 indicates compensation)
            display_map = np.clip(self.function_map, 0, 2)
        elif display_type == 'health':
            display_map = self.health_map
        elif display_type == 'age':
            display_map = self.age_map / 100  # Normalize to 0-1 range
        elif display_type == 'stimulation':
            display_map = self.stimulation
        elif display_type == 'connections':
            display_map = self.connections
        elif display_type == 'types':
            # Create a colorful display of neuron types
            display_map = self.neuron_types.astype(float) / 2  # Normalize to 0-1 range
        else:
            display_map = np.clip(self.function_map, 0, 2)
            
        return display_map
    
    def get_stats(self):
        """Return statistics about the current state."""
        active_count = np.sum(self.neurons == 1)
        inactive_count = np.sum(self.neurons == 0)
        compensating_count = np.sum((self.neurons == 1) & (self.function_map > 1.1))
        
        return {
            "time_step": self.time_step,
            "active": active_count,
            "inactive": inactive_count,
            "compensating": compensating_count,
            "total_recovered": self.total_recovered,
            "stress_level": self.stress_level
        }


def run_simulation():
    """Run the neural plasticity simulation with visualization."""
    # Create the simulation
    sim = NeuralPlasticitySimulation(grid_size=50)
    
    # Create the figure and subplots
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 3)
    
    # Main display
    ax_main = fig.add_subplot(gs[0, :2])
    ax_main.set_title("Neural Function")
    
    # Stats plot
    ax_stats = fig.add_subplot(gs[1, :2])
    ax_stats.set_title("Statistics Over Time")
    ax_stats.set_xlabel("Time Step")
    ax_stats.set_ylabel("Neuron Count")
    
    # Secondary display
    ax_secondary = fig.add_subplot(gs[0, 2])
    ax_secondary.set_title("Neuron Types")
    
    # Tertiary display
    ax_tertiary = fig.add_subplot(gs[1, 2])
    ax_tertiary.set_title("Stimulation Level")
    
    # Initial damage to simulate injury
    sim.damage_region(center_x=25, center_y=25, radius=8)
    
    # Apply initial stimulation (like immediate medical intervention)
    sim.apply_stimulation(center_x=27, center_y=27, radius=4, strength=0.2)
    
    # Display the initial states
    im_main = ax_main.imshow(sim.get_display_map('function'), cmap=sim.cmap, vmin=0, vmax=2)
    cbar_main = fig.colorbar(im_main, ax=ax_main)
    cbar_main.set_label('Neural Function (>1 indicates compensation)')
    
    # Set up the type display with a different colormap
    im_secondary = ax_secondary.imshow(
        sim.get_display_map('types'), 
        cmap='viridis', 
        vmin=0, 
        vmax=1
    )
    cbar_secondary = fig.colorbar(im_secondary, ax=ax_secondary)
    cbar_secondary.set_label('Neuron Type (0=Regular, 0.5=Fast, 1.0=Slow)')
    
    # Set up the stimulation display
    im_tertiary = ax_tertiary.imshow(
        sim.get_display_map('stimulation'), 
        cmap='plasma', 
        vmin=0, 
        vmax=1
    )
    cbar_tertiary = fig.colorbar(im_tertiary, ax=ax_tertiary)
    cbar_tertiary.set_label('Stimulation Level')
    
    # Initial empty stats plot
    stats_lines = {}
    stats_lines['active'], = ax_stats.plot([], [], 'b-', label='Active')
    stats_lines['inactive'], = ax_stats.plot([], [], 'k-', label='Inactive')
    stats_lines['compensating'], = ax_stats.plot([], [], 'r-', label='Compensating')
    stats_lines['recovered'], = ax_stats.plot([], [], 'g-', label='Recovered')
    ax_stats.legend()
    
    # Text annotation for statistics
    stats = sim.get_stats()
    stats_text = ax_main.text(
        0.02, 0.95, 
        f"Time: {stats['time_step']}\nActive: {stats['active']}\n"
        f"Inactive: {stats['inactive']}\nCompensating: {stats['compensating']}\n"
        f"Total Recovered: {stats['total_recovered']}\nStress: {stats['stress_level']:.2f}",
        transform=ax_main.transAxes,
        fontsize=9,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7)
    )
    
    # Cycle between different view modes
    view_modes = ['function', 'types', 'stimulation', 'health', 'connections', 'age']
    current_mode_idx = 0
    current_secondary_idx = 1
    current_tertiary_idx = 2
    
    # Animation update function
    def update_frame(frame):
        nonlocal current_mode_idx, current_secondary_idx, current_tertiary_idx
        
        # Update the simulation
        sim.update()
        
        # Every 40 frames, switch the main display mode
        if frame % 50 == 0 and frame > 0:
            current_mode_idx = (current_mode_idx + 1) % len(view_modes)
            current_secondary_idx = (current_secondary_idx + 1) % len(view_modes)
            current_tertiary_idx = (current_tertiary_idx + 1) % len(view_modes)
            
            # Update titles and colorbars
            ax_main.set_title(f"Neural {view_modes[current_mode_idx].capitalize()}")
            ax_secondary.set_title(f"Neural {view_modes[current_secondary_idx].capitalize()}")
            ax_tertiary.set_title(f"Neural {view_modes[current_tertiary_idx].capitalize()}")
            
            # Update colorbar labels
            cbar_main.set_label(f'Neural {view_modes[current_mode_idx].capitalize()}')
            cbar_secondary.set_label(f'Neural {view_modes[current_secondary_idx].capitalize()}')
            cbar_tertiary.set_label(f'Neural {view_modes[current_tertiary_idx].capitalize()}')
        
        # Every 10 frames, apply some random stimulation
        if frame % 15 == 0 and frame > 20:
            # Apply stimulation to a random area
            x = np.random.randint(0, sim.grid_size)
            y = np.random.randint(0, sim.grid_size)
            sim.apply_stimulation(x, y, radius=3, strength=0.2)
            
        # Every 70 frames, cause new damage
        if frame % 70 == 0 and frame > 30:
            x = np.random.randint(10, sim.grid_size-10)
            y = np.random.randint(10, sim.grid_size-10)
            sim.damage_region(x, y, radius=3, intensity=0.8)
        
        # Update the displays
        im_main.set_array(sim.get_display_map(view_modes[current_mode_idx]))
        im_secondary.set_array(sim.get_display_map(view_modes[current_secondary_idx]))
        im_tertiary.set_array(sim.get_display_map(view_modes[current_tertiary_idx]))
        
        # Update statistics
        stats = sim.get_stats()
        stats_text.set_text(
            f"Time: {stats['time_step']}\nActive: {stats['active']}\n"
            f"Inactive: {stats['inactive']}\nCompensating: {stats['compensating']}\n"
            f"Total Recovered: {stats['total_recovered']}\nStress: {stats['stress_level']:.2f}"
        )
        
        # Update plot data
        for key in stats_lines:
            x_data = list(range(len(sim.history[key])))
            y_data = sim.history[key]
            stats_lines[key].set_data(x_data, y_data)
            
        # Adjust y-axis limits
        if sim.history['active']:
            max_val = max([
                max(sim.history['active'] or [0]),
                max(sim.history['inactive'] or [0]),
                max(sim.history['compensating'] or [0]) * 5,  # Scale up compensating for visibility
                max(sim.history['recovered'] or [0]) * 5      # Scale up recovered for visibility
            ])
            ax_stats.set_xlim(0, len(sim.history['active']))
            ax_stats.set_ylim(0, max_val * 1.1)
        
        return [im_main, im_secondary, im_tertiary, stats_text] + list(stats_lines.values())
    
    # Create the animation
    anim = animation.FuncAnimation(
        fig, update_frame, frames=300, interval=100, blit=True
    )
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_simulation() 