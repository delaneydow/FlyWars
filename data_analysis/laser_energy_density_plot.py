import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# Set publication-quality parameters with larger fonts
rcParams['font.size'] = 18
rcParams['axes.labelsize'] = 22
rcParams['axes.titlesize'] = 22
rcParams['xtick.labelsize'] = 18
rcParams['ytick.labelsize'] = 18
rcParams['legend.fontsize'] = 16
rcParams['lines.linewidth'] = 3.0
rcParams['axes.linewidth'] = 2.0
rcParams['xtick.major.width'] = 2.0
rcParams['ytick.major.width'] = 2.0
rcParams['xtick.major.size'] = 6
rcParams['ytick.major.size'] = 6

# Colorblind-friendly palette (from Wong 2011 / Okabe-Ito palette)
CB_BLUE = '#0173B2'
CB_ORANGE = '#DE8F05'
CB_GREEN = '#029E73'
CB_RED = '#CC3311'
CB_PURPLE = '#946AB5'
CB_BROWN = '#6D4C3D'

# Laser parameters
P = 60  # Laser power in Watts

# Distance array (meters)
distance = np.linspace(0.01, 10, 1000)

# Beam converges to minimum spot at d = 2.0 m with radius = 1.7 cm
# Then remains approximately collimated (or slowly diverges) beyond that
# Before 2m, beam is larger (converging)

d_focus = 2.0  # focal plane distance in meters
r_focus = 1.7  # spot radius at focal plane in cm

# Model: Gaussian beam focusing
# r(d) = r_focus * sqrt(1 + ((d - d_focus) / z_R)^2)
# where z_R is the Rayleigh range (depth of focus)

# Choose z_R to match the curve shape from your original plot
# Larger z_R = more gradual convergence/divergence
z_R = 0.5  # Rayleigh range in meters (adjust to match your system)

# Calculate beam radius as function of distance
beam_radius = r_focus * np.sqrt(1 + ((distance - d_focus) / z_R)**2)  # cm
beam_area = np.pi * beam_radius**2  # cm²
energy_density = P / beam_area  # W/cm²

# Verify at key points
idx_2m = np.argmin(np.abs(distance - 2.0))
idx_0m = 0
print(f"At d = 0.0 m: radius = {beam_radius[idx_0m]:.2f} cm, energy density = {energy_density[idx_0m]:.1f} W/cm²")
print(f"At d = 2.0 m: radius = {beam_radius[idx_2m]:.2f} cm, energy density = {energy_density[idx_2m]:.1f} W/cm²")

# Lethal thresholds for different exposure times and species
# Energy fluence (J/cm^2) / exposure time (s) = Power density (W/cm^2)
exposure_time = 0.25  # seconds

# LD50 for Drosophila hydei: 5.1 J/cm^2
# LD90 for Drosophila melanogaster: 26 J/cm^2
ld50_hydei = 5.1 / exposure_time  # W/cm^2
ld90_melanogaster = 26 / exposure_time  # W/cm^2

# Create horizontal lines for lethal thresholds
lethal_ld50 = np.ones_like(distance) * ld50_hydei
lethal_ld90 = np.ones_like(distance) * ld90_melanogaster

# Create the plot
fig, ax = plt.subplots(figsize=(14, 8))

# Plot the curves with colorblind-friendly colors
ax.plot(distance, energy_density, color=CB_BLUE, linewidth=3.5, 
        label='Laser Energy Density', zorder=3)

ax.plot(distance, lethal_ld50, color=CB_ORANGE, linewidth=3.0, 
        linestyle='--', label=f'LD50 D. hydei (5.1 J/cm², 0.25s)', zorder=2)

ax.plot(distance, lethal_ld90, color=CB_RED, linewidth=3.0, 
        linestyle='--', label=f'LD90 D. melanogaster (26 J/cm², 0.25s)', zorder=2)

# Set labels
ax.set_xlabel('Distance (m)', fontsize=24, fontweight='bold')
ax.set_ylabel('Energy Density (W·cm⁻²)', fontsize=24, fontweight='bold')

# Set axis limits
ax.set_xlim(0, 10)
ax.set_ylim(0, 3500)

# Grid with white background
ax.set_facecolor('white')
fig.patch.set_facecolor('white')
ax.grid(True, alpha=0.3, linestyle='-', linewidth=1.0, color='gray')

# Legend
ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=False, 
          fontsize=20, framealpha=0.98, edgecolor='black', facecolor='white')

# Tight layout
plt.tight_layout()

# Save as high-resolution figure
plt.savefig('laser_energy_density_publication.png', 
            dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.savefig('laser_energy_density_publication.pdf', 
            bbox_inches='tight', facecolor='white', edgecolor='none')

print("\n✓ Plot saved successfully!")
print(f"\nParameters:")
print(f"  Laser power: {P} W")
print(f"  Focal plane: {d_focus} m")
print(f"  Spot radius at focus: {r_focus} cm")
print(f"  Rayleigh range: {z_R} m")
print(f"  Exposure time: {exposure_time} s")
print(f"  Peak energy density (at focus): {energy_density[idx_2m]:.1f} W/cm²")
print(f"\nLethal thresholds:")
print(f"  LD50 (D. hydei, 5.1 J/cm²): {ld50_hydei:.1f} W/cm²")
print(f"  LD90 (D. melanogaster, 26 J/cm², 0.25s): {ld90_melanogaster:.1f} W/cm²")
print(f"\nFiles saved:")
print(f"  - laser_energy_density_publication.png (300 DPI)")
print(f"  - laser_energy_density_publication.pdf (vector)")

# Show the plot
plt.show()