import numpy as np

route = 'fallback_points.txt'
data = np.loadtxt(route)
north = []
east = []
for i in range(0, (int(np.size(data) / 2))):
    north.append(data[i][0])
    east.append(data[i][1])