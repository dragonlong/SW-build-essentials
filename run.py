# Import our modules that we are using
import matplotlib.pyplot as plt
import numpy as np

# Create the vectors X and Y
x = np.array(range(-2,2))
y = x ** (4) + 2 * (x**(3)) - x**(2)-x+2

fig = plt.figure()
# Create the plot
plt.plot(x,y)

# Show the plot
plt.show()
