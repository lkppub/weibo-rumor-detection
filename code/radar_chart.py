import os
import numpy as np
import matplotlib.pyplot as plt

# Set default font to sans-serif for clean English display
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# Data: Normalized feature means for rumors and non-rumors
# Dimensions: Text Length, Exclamation Count, Sentiment Score, Total Interactions, Time Span, Followers
categories = ['Text\nLength', 'Exclamation\nCount', 'Sentiment\nScore', 'Total\nInteractions', 'Time\nSpan', 'Followers']

# Normalized values (range 0-1 for radar chart comparability)
non_rumor_norm = [0.51, 0.43, 0.54, 0.86, 0.88, 0.92]
rumor_norm = [0.58, 0.86, 0.22, 0.63, 0.37, 0.06]

# Close the data for radar plotting
angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
angles_closed = angles + [angles[0]]

non_rumor_closed = non_rumor_norm + [non_rumor_norm[0]]
rumor_closed = rumor_norm + [rumor_norm[0]]

# Create radar chart
fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))

# Plot two curves
ax.plot(angles_closed, non_rumor_closed, 'o-', linewidth=2, 
        color='#2ca02c', markersize=8, label='Non-rumor')
ax.fill(angles_closed, non_rumor_closed, alpha=0.15, color='#2ca02c')

ax.plot(angles_closed, rumor_closed, 'o-', linewidth=2, 
        color='#d62728', markersize=8, label='Rumor')
ax.fill(angles_closed, rumor_closed, alpha=0.15, color='#d62728')

# Set tick labels
ax.set_xticks(angles)
ax.set_xticklabels(categories, fontsize=11, fontweight='bold')

# Set radial ticks (normalized 0~1)
ax.set_ylim(0, 1)
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=9)

ax.set_title('Feature Profile Comparison: Rumor vs. Non-rumor\n(Normalized Values)', fontsize=14, pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1), fontsize=12)

plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'figures', 'radar_rumor_vs_nonrumor.png'), dpi=300, bbox_inches='tight')
plt.show()