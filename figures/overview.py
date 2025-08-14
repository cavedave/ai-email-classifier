# Recreate a simple version of the provided illustration using matplotlib.
# This script draws two colored ellipses ("Solution space" and "Opportunity space"),
# labeled boxes (DATA, INTELLIGENCE, USER EXPERIENCE, OPPORTUNITY, VALUE),
# a GOVERNANCE bar, and connecting arrows.

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle, FancyArrow
import os

# Set up canvas
fig, ax = plt.subplots(figsize=(11, 7))
ax.set_xlim(0, 100)
ax.set_ylim(0, 70)
ax.axis('off')

# --- Background ellipses ---
solution = Ellipse((45, 35), width=80, height=55, facecolor='#F8DDBE', edgecolor='none')
opportunity = Ellipse((88, 40), width=25, height=45, facecolor='#CFE38A', edgecolor='none')  # Made bigger
ax.add_patch(solution)
ax.add_patch(opportunity)

# --- Governance bar ---
gov_bar = Rectangle((15, 18), width=50, height=9, facecolor='white', edgecolor='#2B3A67', linewidth=1.5)  # Moved further left (from 20 to 15)
ax.add_patch(gov_bar)
ax.text(40, 22.5, "GOVERNANCE", ha='center', va='center', fontsize=10, color='#2B3A67')  # Centered text moved left to match box

# --- Blue box helper ---
def blue_box(x, y, w, h, label, fontsize=11):
    rect = Rectangle((x, y), w, h, facecolor='#233B7B', edgecolor='#233B7B')
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, label, color='white', ha='center', va='center', fontsize=fontsize, fontweight='bold')
    return rect

# --- Boxes in solution space ---
w, h = 14, 8
b_data = blue_box(16, 33, w, h, "DATA")
b_intel = blue_box(36, 33, w, h, "INTELLIGENCE", fontsize=10.5)
b_ux = blue_box(56, 33, w, h, "USER\nEXPERIENCE", fontsize=10.5)

# --- Boxes in opportunity space ---
b_opp = blue_box(82, 47, 14, 9, "OPPORTUNITY", fontsize=9.0)  # Adjusted position and size to fit better in ellipse
b_val = blue_box(82, 31, 12, 8, "VALUE", fontsize=9.0)

# --- Arrows inside solution space ---
def arrow(x1, y1, x2, y2):
    ax.add_patch(FancyArrow(x1, y1, x2-x1, y2-y1, width=0.2, head_width=1.8, head_length=2.2, length_includes_head=True, color='#233B7B'))

arrow(30, 37, 36, 37)   # DATA -> INTELLIGENCE
arrow(50, 37, 56, 37)   # INTELLIGENCE -> UX
arrow(70, 37, 83, 37)   # UX -> VALUE

# --- Arrows from governance up to boxes ---
# Calculate the center of each box and align arrows from governance bar
data_center = 16 + w/2  # Center of DATA box
intel_center = 36 + w/2  # Center of INTELLIGENCE box  
ux_center = 56 + w/2     # Center of USER EXPERIENCE box

arrow(data_center, 27, data_center, 33)   # Governance -> DATA
arrow(intel_center, 27, intel_center, 33) # Governance -> INTELLIGENCE
arrow(ux_center, 27, ux_center, 33)      # Governance -> USER EXPERIENCE

# --- Arrow between OPPORTUNITY and VALUE (two-way) ---
# Up arrow
arrow(89, 39, 89, 47)
# Down arrow
arrow(89, 47, 89, 39)  # overlapping to create two-headed effect

# --- Labels over regions ---
ax.text(45, 60, "Solution space", ha='center', va='center', fontsize=12, fontweight='bold')
ax.text(88, 62, "Opportunity space", ha='center', va='center', fontsize=12, fontweight='bold')

# --- Figure caption ---
ax.text(5, 4, "Mental model of an AI system", fontsize=11)

# Save to current directory
out_path = "ai_system_mental_model.png"
plt.savefig(out_path, dpi=200, bbox_inches='tight')
print(f"✅ Figure saved as: {os.path.abspath(out_path)}")

# Display the plot
plt.show()
