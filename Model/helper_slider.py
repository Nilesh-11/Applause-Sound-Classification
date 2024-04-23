import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# Define the function that will be called when the slider changes
def calculate_value(x):
    # Example function: returns the square of the input
    return x ** 2

# Create a figure and axis
fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.25)

# Set initial value for the slider
initial_value = 0

# Create the slider
ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')
slider = Slider(ax_slider, 'Input', -10, 10, valinit=initial_value)

# Function to update plot when slider value changes
def update(val):
    y = calculate_value(slider.val)
    ax.clear()
    ax.plot(slider.val, y, 'ro')
    ax.set_xlabel('Input')
    ax.set_ylabel('Output')
    ax.set_title('Function Output')
    fig.canvas.draw_idle()

# Attach the update function to the slider
slider.on_changed(update)

# Show the plot
plt.show()
