#ANOOP KUMAR MISHRA
#2023PCW5321
#ASSIGNMENT -4

# x and y coordinates of the grid points

import numpy as np

# Define the grid size (N)
N = 61  # or 81 or 101

# Calculate the grid spacing
delta = 1 / (N - 1)

# Create the x and y arrays (grid points)
x = np.linspace(0, 1, N)
y = np.linspace(0, 1, N)

# Create a 2D grid using meshgrid function
X, Y = np.meshgrid(x, y)

"""Initialize the flow variables (stream function and vorticity)"""

#NUMPY FOR ARRAY
import numpy as np

# Define grid size
N = 81  # Adjust N as needed: 61, 81, or 101

# Calculate grid spacing
delta = 1 / (N - 1)

# Create the x and y arrays (grid points)
x = np.linspace(0, 1, N)
y = np.linspace(0, 1, N)

# Create a 2D grid using meshgrid function
X, Y = np.meshgrid(x, y)

# Initialize the stream function (psi) and vorticity (omega) arrays
psi = np.zeros((N, N))
omega = np.zeros((N, N))

# Set boundary conditions for vorticity and stream function
# Top boundary (moving lid): Set vorticity based on lid velocity
# The lid moves from left to right with velocity U (e.g., U = 1)
U = 1  # Lid velocity
omega[0, :] = -2 * U / delta  # Top boundary condition for vorticity

# Bottom and side walls: Vorticity boundary conditions (assuming no-slip conditions)
omega[-1, :] = 0  # Bottom wall
omega[:, 0] = 0   # Left wall
omega[:, -1] = 0  # Right wall

# The stream function is initialized to zero everywhere
# Additional boundary conditions may be needed for stream function, depending on the method
print("Stream function and vorticity initialized.")

"""Boundary conditions for the lid-driven cavity"""

import numpy as np

def apply_boundary_conditions(psi, omega, N, delta, U):
    # Top boundary (moving lid)
    # Vorticity at the top boundary based on the known velocity (central difference for u)
    omega[0, 1:-1] = -2 * (U - (psi[0, 2:] - psi[0, :-2]) / (2 * delta)) / delta

    # Bottom wall (stationary)
    # Vorticity at the bottom boundary is set to zero (no-slip condition)
    omega[-1, :] = 0

    # Left and right walls (stationary)
    # Vorticity at the side boundaries is set to zero (no-slip condition)
    omega[:, 0] = 0  # Left wall
    omega[:, -1] = 0  # Right wall

    # Stream function boundary conditions (psi = 0) at all boundaries
    psi[0, :] = 0  # Top boundary
    psi[-1, :] = 0  # Bottom boundary
    psi[:, 0] = 0  # Left boundary
    psi[:, -1] = 0  # Right boundary

    print("Boundary conditions applied.")

# Sample usage:
N = 81  # Number of grid points in each direction (can be 61, 81, or 101)
delta = 1 / (N - 1)  # Grid spacing
U = 1  # Lid velocity

# Initialize flow variables
psi = np.zeros((N, N))
omega = np.zeros((N, N))

# Apply boundary conditions
apply_boundary_conditions(psi, omega, N, delta, U)

""" libraries such as matplotlib for plotting and numpy for numerical operations"""

import numpy as np
import matplotlib.pyplot as plt


N = 81  # Number of grid points
x = np.linspace(0, 1, N)
y = np.linspace(0, 1, N)
X, Y = np.meshgrid(x, y)

psi = np.zeros((N, N))  
omega = np.zeros((N, N))  

# Plotting streamlines
plt.figure()
streamlines = plt.contour(X, Y, psi, levels=20, cmap='cool')
plt.title('Streamlines')
plt.xlabel('x')
plt.ylabel('y')
plt.colorbar(streamlines, label='Stream Function')
plt.grid(True)

# Plotting vorticity contours
plt.figure()
vorticity_contours = plt.contourf(X, Y, omega, levels=20, cmap='jet')
plt.title('Vorticity Contours')
plt.xlabel('x')
plt.ylabel('y')
plt.colorbar(vorticity_contours, label='Vorticity')
plt.grid(True)
plt.show()