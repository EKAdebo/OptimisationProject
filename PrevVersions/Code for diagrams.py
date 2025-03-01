import numpy as np
import matplotlib.pyplot as plt

# # plot the feasible region
# d = np.linspace(-2, 4, 300)
# x, y = np.meshgrid(d, d)
# plt.imshow(((y >= 0) & (y <= 1) & (x >= 0) & (x <= 1)).astype(int),
#            extent=(x.min(), x.max(), y.min(), y.max()), origin="lower", cmap="Greys", alpha=0.3)

# # plot the lines defining the constraints
# x = np.linspace(-1, 2, 2000)
# # y >= 0
# y1 = (x * 0)
# # y <= 1
# y2 = (x * 0) + 1
# # x >= 0
# x3 = (x * 0)
# # x <= 1
# x4 = (x * 0) + 1

# ex2 = np.array([[0,1],[1,1],[1,0],[0,0]])

# # Make plot
# plt.scatter(ex2[:,0], ex2[:,1], color='blue', zorder=3)

# plt.plot(x, y1, label=r'$x_2 \geq 0$')
# plt.plot(x, y2, label=r'$x_2 \leq 1$')
# plt.plot(x3, x, label=r'$x_1 \geq 0$')
# plt.plot(x4, x, label=r'$x_1 \leq 1$')



# plt.xlim(-0.2, 1.75)
# plt.ylim(-0.2, 1.75)
# plt.legend(loc='upper right')

# plt.xlabel(r'$x_1$')
# plt.ylabel(r'$x_2$')
# plt.show()

# plt.close()


import KungsMethod as KM
ex2 = np.array([[0,1],[1,1],[1,0],[0,0]])

def f1(x):
    return -x[0]
    
def f2(x):
    return x[0] + x[1]

functions = [f1,f2]

results = []
#TODO (Fix this so that is works for NP array instead of converting at the end)
for i in ex2:
    r = [func(i) for func in functions]
    results.append(r)

result = np.array(results)


VM2 = KM.KungMethod(result)

plt.scatter(result[:,0],result[:,1],color = 'blue', label = "Dominated Set")
plt.scatter(VM2[:,0],VM2[:,1],color = 'red', label = "Non-Dominated Set")
plt.xlabel(r'$z_1$')
plt.ylabel(r'$z_2$')
plt.legend(loc='upper right')
plt.show()