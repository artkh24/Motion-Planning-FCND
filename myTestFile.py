import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point, LineString
from queue import PriorityQueue
from sampling import Sampler
import numpy.linalg as LA
from sklearn.neighbors import KDTree

print(nx.__version__)


filename = 'colliders.csv'
data = np.loadtxt(filename, delimiter=',', dtype='Float64', skiprows=2)
sampler = Sampler(data)

def can_connect(n1, n2):

	polygons = sampler._polygons
	l = LineString([n1, n2])
	for p in polygons:
		minofn1n2 = min(n1[2], n2[2])

		if p.crosses(1) and p.height >= minofn1n2:
			return False
	return True

def create_graph():

	# print(polygons)

	nodes = sampler.sample(300)

	k = 10

	g = nx.Graph()
	tree = KDTree(nodes)
	for n1 in nodes:
		idxs = tree.query([n1], k, return_distance=False)[0]

		for idx in idxs:
			n2 = nodes[idx]
			if n2 == n1:
				continue

			if can_connect(n1, n2):
				g.add_edge(n1, n2, weight=1)
	return g

g = create_graph()
print(f"NUMBER OF EDGES: {g.edges}")