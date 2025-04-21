from math import dist
import numpy as np
import random
import json
from typing import List
from scipy.optimize import NonlinearConstraint, LinearConstraint
import matplotlib.pyplot as plt


from Point import Point
from Functions import *


def save_to_json(points, filename="nonDominated.json"):
    data = []
    for p in points:
        data.append({
            "vector": p.vector.tolist(),
            "eval_f": p.eval_f,
            "eval_d": [d.tolist() for d in p.eval_d]
        })
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)

# def generatePoints(npoints:int, dim:int, objFuncs:List[Function],eqConstraints:List[Constraint], ineqConstraints:List[Constraint],bounds:List[float]) -> List[Point]:
#     """
#     This function generates a set of random points with dimentions as inputed and evaluates them on the objective functions

#     INPUTS
#     npoints: Number of points to generate
#     dim: Dimension of the points
#     objFuncs: List of objective functions

#     OUTPUTS
#     Xorg: List of points evaluated on the objective functions
#     """

#     Xorg = []

#     constraints = []
#     #Add constraints in form needed for scipy and convert from
#     print("Im doing something, anything..")
#     i = 1
#     for f in ineqConstraints:
#         #if f.nonlinear == True:
#         constraints.append(NonlinearConstraint(lambda x: -f(x), float(0), np.inf))
#         # else:
#         #     constraints.append(LinearConstraint(lambda x: -f(x), 0.0, np.inf))

#     for f in eqConstraints:
#         constraints.append(NonlinearConstraint(lambda x: f(x), 0, 0))


#     from scipy.optimize import differential_evolution

#     def objective(x):
#         return 0  # We're only interested in finding feasible points


#     for i in range(npoints):
#         print(f"Point {i} generate")
#         result = differential_evolution(objective, bounds, constraints=constraints)
#         x = Point(result.x)
#         x.evaluate(objFuncs)

#         Xorg.append(x)

#     return Xorg

###############################################

#DEFINE PARAMETERS
dimension = 36
grid_size = 6
npoints = 108

#Setting the variables
mu = 0.25
beta = 0.5

###############################################
#DEFINE HELPER FUNCTIONS
###############################################

def power_calc(temp,irr):
    """
    Power in Watts
    Calculate the power output of a PV panel based on temperature and irradiation.
    """
    k1 = -0.01724
    k2 = -0.04047
    k3 = -0.0047
    k4 = 1.48 * (10**-4)
    k5 = 1.48 * (110**-4)
    k6 = 5.0 * (110**-6)
    kT = 0.035

    T_module = temp + kT*irr
    irr = irr/1000
    G_calc = np.log(irr)
    T_calc = T_module - 25



    P = irr*(200 + k1*G_calc + k2*(G_calc**2) + k3*(T_calc) + k4*(T_calc*G_calc) + \
        k5*((G_calc**2)*T_calc) + k6*(T_calc**2))

    return P

def embodiedCarbonCalc(power, distance, Area, LifeSpan =25, weight = 26, LifeTimeCarbon = (24/1000)):
    """
    Calculate the embodied carbon based on distance and area.
    """
    #Lifetime energy calc
    LifetimePower = power  * LifeSpan * (Area * 1000)
    CarbonPerKm = weight * 0.10650 * (Area * 1000)


    # Calculate embodied carbon
    embodied_carbon = (distance * CarbonPerKm) + (LifeTimeCarbon * LifetimePower)

    return embodied_carbon

# Random Source Points
point1 = (-40, -10)
point2 = (32, 37)

# Create the grid
square_size = 1
x = np.arange(0, grid_size , square_size)
y = np.arange(0,  grid_size, square_size)
xx, yy = np.meshgrid(x, y)

# Calculate distances from the points
distances_point1 = np.sqrt((xx - point1[0])**2 + (yy - point1[1])**2)
distances_point2 = np.sqrt((xx - point2[0])**2 + (yy - point2[1])**2)

# Combine distances (e.g., take the minimum distance to either point)
combined_distances = np.minimum(distances_point1, distances_point2)


###############################################
#Generate values as an example
###############################################

#Random temp and irradiation values
temperature = np.array([random.uniform(16.5, 1.5) for _ in range(dimension)])
irradiation = np.array([random.uniform(108, 123) for _ in range(dimension)])

instantaneous_power = np.array([power_calc(temp, irr) for temp, irr in zip(temperature, irradiation)])

#Random cost of land values
LandCost = np.array([random.uniform(0.1,0.5) for _ in range(dimension)])

#Random ALC values
ALC = np.array([random.randint(1, 14) for _ in range(dimension)])
ALC_Inclusion = np.array([True if x > 4 else False for x in ALC])


###############################################################
#Graph Generation
################################################################

# Create a color-coded grid based on proximity to point1 or point2
color_grid = np.where(distances_point1 < distances_point2, 'red', 'blue')



# Create a 2x2 layout for the plots
fig, axs = plt.subplots(2, 2, figsize=(12, 12))

# Plot 1: Color-coded grid
for i in range(grid_size ):
    for j in range(grid_size ):
        axs[0, 0].add_patch(plt.Rectangle((i, j), square_size, square_size, color=color_grid[j, i], alpha=0.5, edgecolor='black'))
axs[0, 0].plot(point1[0], point1[1], 'ro', label='Point 1')
axs[0, 0].plot(point2[0], point2[1], 'bo', label='Point 2')
axs[0, 0].set_title('Color-coded Grid')
axs[0, 0].set_xlim(0, grid_size )
axs[0, 0].set_ylim(0, grid_size )
axs[0, 0].set_aspect('equal', adjustable='box')
axs[0, 0].legend()


# Plot 2: Instantaneous power color-coded overlay
power_grid = np.zeros((grid_size, grid_size))
for i in range(grid_size):
    for j in range(grid_size):
        index = i * grid_size + j
        if index < dimension:
            power_grid[i, j] = instantaneous_power[index]
im = axs[0, 1].imshow(power_grid, cmap='viridis', origin='lower', extent=[0, grid_size, 0, grid_size])
fig.colorbar(im, ax=axs[0, 1], orientation='vertical', label='Instantaneous Power')

# Plot 3: ALC overlay with color-coding based on ALC_Inclusion
for i in range(grid_size):
    for j in range(grid_size):
        index = i * grid_size + j
        if index < dimension:
            color = 'green' if ALC_Inclusion[index] else 'red'
            axs[1, 0].add_patch(plt.Rectangle((j, i), square_size, square_size, color=color, alpha=0.5, edgecolor='black'))
axs[1, 0].set_title('ALC Overlay (Color-coded)')
axs[1, 0].set_xlim(0, grid_size )
axs[1, 0].set_ylim(0, grid_size )
axs[1, 0].set_aspect('equal', adjustable='box')

# Plot 4: Land cost overlay with color-coding based on land cost
land_cost_grid = np.zeros((grid_size, grid_size))
for i in range(grid_size):
    for j in range(grid_size):
        index = i * grid_size + j
        if index < dimension:
            land_cost_grid[i, j] = LandCost[index]
im = axs[1, 1].imshow(land_cost_grid, cmap='coolwarm', origin='lower', extent=[0, grid_size, 0, grid_size])
fig.colorbar(im, ax=axs[1, 1], orientation='vertical', label='Land Cost')
axs[1, 1].set_title('Land Cost Overlay (Color-coded)')
axs[1, 1].set_xlim(0, grid_size)
axs[1, 1].set_ylim(0, grid_size)
axs[1, 1].set_aspect('equal', adjustable='box')

# Adjust layout and show the combined plot
plt.tight_layout()
plt.show()

# Save the plot to a file
output_path = "ExampleRun/PlotStartValues.png"
plt.savefig(output_path)
print(f"Plot saved to {output_path}")

################################################################
#Define Constraints and Objective Functions
################################################################

###############################################
#Minimise Cost
def f1(x):
    return np.dot(cost,x)

def f1_d(x):
    return cost

#Setting the functions
func_f1 = Function(f1,f1_d,"min_cost")

#Maximise Power
def f2(x):
    return - np.dot(power_val,x)/1000

def f2_d(x):
    return -power_val/1000

#Setting the functions
func_f2 = Function(f2,f2_d,"max_energy")


#Minimise Embodied Carbon
def f3(x):
    return np.dot(embodied_carbon,x)


def f3_d(x):
    return embodied_carbon

#Setting the functions
func_f3 = Function(f3,f3_d,"min_embodied carbon")

functions = [func_f1,func_f2,func_f3]

#################################################################
#Inequality constraints

g = []
for i in range(size):
    if ALC_Inclusion[i] == True:

        #Constraints, >=0, <= 1
        def constraint_g1(x,i=i):
            return -x[i]

        def constraint_g1_d(x,i=i):
            d = np.zeros(size)
            d[i] = -1
            return d


        def constraint_g2(x,i=i):
            return x[i] - 1

        def constraint_g2_d(x,i=i):
            d = np.zeros(size)
            d[i] = 1
            return d

        g.append(Constraint(constraint_g1,constraint_g1_d,(str(i) + "_g1")))
        g.append(Constraint(constraint_g2,constraint_g2_d,(str(i) + "_g2")))


max_power = 0.3 * 356 * size
#POWER CONSTRAINT
def constraint_g3(x):
    v = max_power - np.dot(power_val,x)
    v = v/100
    return v

def constraint_g3_d(x):
    v = (1/100) * (-power_val)
    return (-power_val /100)

g.append(Constraint(constraint_g3,constraint_g3_d,"PowerConstraint"))


#AREA CONSTRAINT

a_max = (size)*(3/4)

def constraint_g4(x):
    return np.sum(x) - a_max

def constraint_g4_d(x):
    return np.ones(size)

g.append(Constraint(constraint_g4,constraint_g4_d,"AreaConstraint"))

#########################################
#Equality Constraints
h = []

#Where location is excluded, area = 0
for i in range(size):
    if ALC_Inclusion[i] == False:
        def constraint_h1(x,i=i):
            return x[i]

        def constraint_h1_d(x,i=i):
            d = np.zeros(size)
            d[i] = 1
            return d

        h.append(Constraint(constraint_h1,constraint_h1_d,(str(i) + "_h1"),True))


#########################################

print(ALC_Inclusion)

np.random.seed(0)
#Setting the variables
mu = 0.25
beta = 0.5


numPoints = 20
iterations = 5

bounds = [(0, 1) if ALC_Inclusion[i] else (0, 0) for i in range(size)]  # Define search space

originalPoints, nonDominated = main(functions, g, h, size,bounds, numPoints, iterations, mu, beta,testing = False)








###############################################




print("Done")
print(len(nonDominated))


# Save nonDominated points to a JSON file
def save_to_json(points, filename="nonDominated.json"):
    data = []
    for p in points:
        data.append({
            "vector": p.vector.tolist(),
            "eval_f": p.eval_f,
            "eval_d": [d.tolist() for d in p.eval_d]
        })
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)

save_to_json(nonDominated)


arrangements = [[0,1],[0,2],[1,2]]

# Create a single figure for all arrangements
plt.figure(figsize=(15, 10))

for i, functions in enumerate(arrangements, 1):
    # Create a subplot for each arrangement
    plt.subplot(2, 2, i)

    # Plot the points
    for p in nonDominated:
        plt.scatter(p.eval_f[functions[0]], p.eval_f[functions[1]], color='blue')

    # Set labels and title
    plt.xlabel(f"Objective Function {functions[0]+1}")
    plt.ylabel(f"Objective Function {functions[1]+1}")
    plt.title(f"Pareto Front: Objective {functions[0]+1} vs Objective {functions[1]+1}")
    plt.grid(True)

# Adjust layout and show the plot
plt.tight_layout()
plt.show()

