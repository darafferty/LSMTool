import grouper
import numpy as np

a = np.array([[1.0, 2.0],[3.0,4.0],[5.0,6.0]])
f = np.array([1.0,2.0,3.0])

g = grouper.Grouper()

g.readCoordinates(a,f)

g.setLookDistance(5.0)
g.setGroupingDistance(5.0)

g.run()
b = []
g.group(b)

print(b)
