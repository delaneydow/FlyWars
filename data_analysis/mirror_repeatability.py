import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# Set publication-quality parameters with larger fonts
rcParams['font.size'] = 18
rcParams['axes.labelsize'] = 22
rcParams['axes.titlesize'] = 22
rcParams['xtick.labelsize'] = 18
rcParams['ytick.labelsize'] = 18
rcParams['legend.fontsize'] = 14
rcParams['lines.linewidth'] = 2.5
rcParams['axes.linewidth'] = 2.0
rcParams['xtick.major.width'] = 2.0
rcParams['ytick.major.width'] = 2.0
rcParams['xtick.major.size'] = 6
rcParams['ytick.major.size'] = 6

# Colorblind-friendly palette (using colormap for multiple lines)
import matplotlib.cm as cm
colors = cm.viridis(np.linspace(0, 1, 11))

# Mirror repeatability data
# Optotune MRE-3: Repeatability 30-100 μrad RMS
# Different steering angles from 0 to 100 μrad

repeatability_angles = np.array([0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0])  # μrad
distance = np.linspace(0, 5, 100)  # meters

# Initial spot radius at d=0
r0 = 1.5  # cm

# Calculate spot radius for each angle
# r(d) = r0 + d * tan(θ), but for small angles: tan(θ) ≈ θ
# θ in radians = θ_μrad × 1e-6
# Convert to cm: multiply by 100 (m to cm)

fig, ax = plt.subplots(figsize=(14, 8))

for i, angle_urad in enumerate(repeatability_angles):
    angle_rad = angle_urad * 1e-6  # convert μrad to radians
    # Spot radius increases with distance due to beam divergence from mirror angle
    spot_radius = r0 + distance * np.tan(angle_rad) * 100  # cm
    
    ax.plot(distance, spot_radius, color=colors[i], linewidth=2.5,
            label=f'{angle_urad:.1f} μrad')

# Set labels
ax.set_xlabel('Distance (m)', fontsize=24, fontweight='bold')
ax.set_ylabel('Spot Radius (mm)', fontsize=24, fontweight='bold')

# Set axis limits to match original
ax.set_xlim(0, 5)
ax.set_ylim(1.5, 2.0)

# Grid with white background
ax.set_facecolor('white')
fig.patch.set_facecolor('white')
ax.grid(True, alpha=0.3, linestyle='-', linewidth=1.0, color='gray')

# Legend - two columns for better space usage
ax.legend(loc='upper left', frameon=True, fancybox=True, shadow=False,
          fontsize=13, framealpha=0.98, edgecolor='black', facecolor='white',
          ncol=2, title='Mirror Angle', title_fontsize=14)

# Tight layout
plt.tight_layout()

# Save as high-resolution figure
plt.savefig('mirror_repeatability_publication.png',
            dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.savefig('mirror_repeatability_publication.pdf',
            bbox_inches='tight', facecolor='white', edgecolor='none')

print("✓ Mirror repeatability plot saved successfully!")
print(f"\nOptotune MRE-3 Specifications:")
print(f"  Repeatability: 30-100 μrad RMS")
print(f"  Diameter: 15 mm")
print(f"  Initial spot radius: {r0} cm")
print(f"\nFiles saved:")
print(f"  - mirror_repeatability_publication.png (300 DPI)")
print(f"  - mirror_repeatability_publication.pdf (vector)")

plt.show()