import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# Create figure and subplots
fig = plt.figure(figsize=(12, 8))
fig.suptitle('Visualization of f(x) = cos(x) + cos²(2x) + cos³(3x) + ... + cosⁿ(nx)', fontsize=16)

# Main function plot
ax1 = fig.add_subplot(2, 1, 1)
ax1.set_title('Full Function')
ax1.set_xlabel('x')
ax1.set_ylabel('f(x)')
ax1.grid(True)

# Individual terms plot
ax2 = fig.add_subplot(2, 1, 2)
ax2.set_title('Individual Terms')
ax2.set_xlabel('x')
ax2.set_ylabel('Value')
ax2.grid(True)

# Adjust layout to make room for sliders
plt.subplots_adjust(bottom=0.25)

# Create x range for plotting - adjusted to range of 10 with steps of 0.001
x = np.arange(0, 10.001, 0.001)  # From 0 to 10 with step 0.001

# Calculate the function values for different n
def recursive_cos(x, n):
    """Calculate cos(x) + cos²(2x) + cos³(3x) + ... + cosⁿ(nx)"""
    result = np.zeros_like(x, dtype=float)
    terms = []
    
    for i in range(1, n+1):
        # Correct calculation: cos(i*x)^i
        term = np.cos(i*x)**i
        result += term
        if i <= 5:  # Store first 5 terms for individual plotting
            terms.append(term)
    
    return result, terms

# Initial parameters
initial_n = 5

# Calculate initial function values
y, terms = recursive_cos(x, initial_n)

# Create the main function plot
line1, = ax1.plot(x, y, 'b-', lw=2, label=f'n={initial_n}')
ax1.set_ylim(-1, 5)
ax1.legend()

# Create individual term plots
term_lines = []
colors = ['r', 'g', 'b', 'm', 'c']
for i, term in enumerate(terms):
    if i < len(colors):
        line, = ax2.plot(x, term, color=colors[i], lw=1.5, label=f'cos^{i+1}({i+1}x)')
        term_lines.append(line)

ax2.legend()
ax2.set_ylim(-1, 1.2)

# Add slider for n
ax_n = plt.axes([0.2, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')
slider_n = Slider(ax_n, 'n', 1, 15, valinit=initial_n, valstep=1)

# Update function for when slider changes
def update(val):
    n = int(slider_n.val)
    y, terms = recursive_cos(x, n)
    
    # Update main function plot
    line1.set_ydata(y)
    line1.set_label(f'n={n}')
    ax1.legend()
    
    # Update individual term plots
    for i, line in enumerate(term_lines):
        if i < len(terms):
            line.set_ydata(terms[i])
    
    # Adjust y-axis limits if needed
    max_y = max(5, np.max(y) + 0.5)
    ax1.set_ylim(-1, max_y)
    
    fig.canvas.draw_idle()

slider_n.on_changed(update)

# Add text explanation
plt.figtext(0.1, 0.02, 'Move the slider to change the number of terms (n) in the summation', fontsize=10)

# Add vertical lines for notable points (adjusted for new x range)
notable_points = [0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi]
labels = ['0', 'π/2', 'π', '3π/2', '2π']

for point, label in zip(notable_points, labels):
    if 0 <= point <= 10:
        ax1.axvline(x=point, color='gray', linestyle='--', alpha=0.3)
        ax2.axvline(x=point, color='gray', linestyle='--', alpha=0.3)
        ax1.text(point, -0.5, label, ha='center')

# Show the plot
plt.tight_layout(rect=[0, 0.25, 1, 0.95])
plt.show()