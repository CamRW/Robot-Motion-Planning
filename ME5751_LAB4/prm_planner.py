import cv2
import numpy as np
import math
import random
from PIL import Image, ImageTk

from Path import *
from sklearn.neighbors import KDTree
import time
# from Queue import Queue

class prm_node:
	def __init__(self,map_i=int(0),map_j=int(0)):
		self.map_i = map_i
		self.map_j = map_j
		self.edges = [] #edges of child nodes
		self.parent = None #parent node


class prm_edge:
	def __init__(self,node1=None,node2=None):
		self.node1 = node1 #parent node
		self.node2 = node2 #child node

#You may modify it to increase efficiency as list.append is slow
class prm_tree:
	def __init__(self):
		self.nodes = []
		self.edges = []

	def add_nodes(self,node):
		self.nodes.append(node)

	#add an edge to our PRM tree, node1 is parent, node2 is the kid
	def add_edges(self,node1,node2): 
		edge = prm_edge(node1,node2)
		self.edges.append(edge)
		node1.edges.append(edge)
		node2.parent=edge.node1



class Edge:
    def __init__(self, to_node, length):
        self.to_node = to_node
        self.length = length


class Graph:
    def __init__(self):
        self.nodes = set()
        self.edges = dict()

    def add_node(self, node):
        self.nodes.add(node)

    def add_edge(self, from_node, to_node, length):
        edge = Edge(to_node, length)
        if from_node in self.edges:
            from_node_edges = self.edges[from_node]
        else:
            self.edges[from_node] = dict()
            from_node_edges = self.edges[from_node]
        from_node_edges[to_node] = edge


class path_planner:
	def __init__(self,graphics):
		self.graphics = graphics
		# self.graphics.scale = 400 #half pixel number on canvas, the map should be 800 x 800
		# self.graphics.environment.robots[0].set_bot_size(body_cm = 2*self.inflation_radius)
		#self.graphics.environment.width/height = 2

		self.costmap = self.graphics.map
		self.map_width = self.costmap.map_width
		self.map_height = self.costmap.map_height

		self.pTree=prm_tree()

		self._init_path_img()
		self.path = Path()
		
		self.set_start(world_x = .0, world_y = .0)
		self.set_goal(world_x = -100.0, world_y = 200.0, world_theta = .0)


		self.valid_start = True
		self.valid_goal = True

		self.plan_path()
		self._show_path()

	def set_start(self, world_x = 0, world_y = 0, world_theta = 0):
		self.start_state_map = Pose()
		map_i, map_j = self.world2map(world_x,world_y)
		print("Start with %d, %d on map"%(map_i,map_j))
		self.start_state_map.set_pose(map_i,map_j,world_theta)
		self.start_node = prm_node(map_i,map_j)
		self.pTree.add_nodes(self.start_node)

	def set_goal(self, world_x, world_y, world_theta = 0):
		self.goal_state_map = Pose()
		goal_i, goal_j = self.world2map(world_x, world_y)
		print ("goal is %d, %d on map"%(goal_i, goal_j))
		self.goal_state_map.set_pose(goal_i, goal_j, world_theta)
		self.goal_node = prm_node(goal_i,goal_j)
		self.pTree.add_nodes(self.goal_node)

	#convert a point a map to the actual world position
	def map2world(self,map_i,map_j):
		world_x = -self.graphics.environment.width/2*self.graphics.scale + map_j
		world_y = self.graphics.environment.height/2*self.graphics.scale - map_i
		return world_x, world_y

	#convert a point in world coordinate to map pixel
	def world2map(self,world_x,world_y):
		map_i = int(self.graphics.environment.width/2*self.graphics.scale - world_y)
		map_j = int(self.graphics.environment.height/2*self.graphics.scale + world_x)
		if(map_i<0 or map_i>=self.map_width or map_j<0 or map_j>=self.map_height):
			warnings.warn("Pose %f, %f outside the current map limit"%(world_x,world_y))

		if(map_i<0):
			map_i=int(0)
		elif(map_i>=self.map_width):
			map_i=self.map_width-int(1)

		if(map_j<0):
			map_j=int(0)
		elif(map_j>=self.map_height):
			map_j=self.map_height-int(1)

		return map_i, map_j

	def _init_path_img(self):
		self.map_img_np = 255*np.ones((int(self.map_width),int(self.map_height),4),dtype = np.int16)
		# self.map_img_np[0:-1][0:-1][3] = 0
		self.map_img_np[:,:,3] = 0

	def _show_path(self):
		for pose in self.path.poses:
			map_i = pose.map_i
			map_j = pose.map_j 
			self.map_img_np[map_i][map_j][1] =0
			self.map_img_np[map_i][map_j][2] =0
			self.map_img_np[map_i][map_j][3] =255

		np.savetxt("file.txt", self.map_img_np[1])

		self.path_img=Image.frombytes('RGBA', (self.map_img_np.shape[1],self.map_img_np.shape[0]), self.map_img_np.astype('b').tostring())
		# self.path_img = toimage(self.map_img_np)
		#self.path_img.show()
		self.graphics.draw_path(self.path_img)

	def check_vicinity(self,x1,y1,x2,y2,threshold = 1.0):
		if(math.sqrt((x1-x2)**2+(y1-y2)**2)<threshold):
			return True
		else:
			return False

	def generate_points(self, current, destination, numOfCoords = 100, mapsize=500):
		coordsList = np.random.randint(mapsize-1, size=(numOfCoords, 2))
        # Adding begin and end points
		coordsList = np.concatenate((coordsList, current, destination), axis=0)

		return coordsList

	def generate_path(self, numOfCoords=100):
		self.valid_start = True
		self.valid_goal = True
		graph = Graph()
		current=np.array([[self.start_node.map_i,self.start_node.map_j]])
		destination=np.array([[self.goal_node.map_i,self.goal_node.map_j]])
		graph.add_node(tuple([self.start_node.map_i,self.start_node.map_j]))
		graph.add_node(tuple([self.goal_node.map_i,self.goal_node.map_j]))

		if self.costmap.costmap[self.start_node.map_i][self.start_node.map_j] == 255:
			self.valid_start = False
			return 0, 0, 0
		if self.costmap.costmap[self.goal_node.map_i][self.goal_node.map_j] == 255:
			self.valid_goal = False
			return 0, 0, 0
		
		points = self.generate_points(current,destination, numOfCoords=numOfCoords)
		for p in points:
			if(self.costmap.costmap[p[0]][p[1]]) == 255:
				points = np.delete(points, np.where(p),0)

		tree = KDTree(points)
					
		dist, ind = tree.query(points[:], k=len(points))

		for index in ind:
			anchor = points[index[0]]
			for i in range(1,len(index)):
				line = bresenham(anchor[0],anchor[1],points[i][0],points[i][1])
				obstacle = False
				for l in line:
					if self.costmap.costmap[l[0]][l[1]] == 255:
						obstacle = True
						break

				if obstacle == False:
					graph.add_node(tuple([anchor[0],anchor[1]]))
					graph.add_node(tuple([points[i][0],points[i][1]]))
					graph.add_edge(tuple([anchor[0],anchor[1]]), tuple([points[i][0],points[i][1]]), dist[index[0]][i])

		previous_nodes, shortest_path = dijkstra(graph,tuple([self.start_node.map_i,self.start_node.map_j]))

		return previous_nodes, shortest_path, points

		
	def check_valid_path(self,previous_nodes, points):
		pNode_list = []
		pNode = tuple([self.goal_node.map_i,self.goal_node.map_j])
		pNode_list.append(pNode)
		

		while pNode != tuple([self.start_node.map_i,self.start_node.map_j]):
			try:
				pNodeNext = previous_nodes[pNode]
				points = bresenham(pNode[0],pNode[1], pNodeNext[0], pNodeNext[1])
				for p in points:
					self.path.add_pose(Pose(map_i=p[0],map_j=p[1],theta=0))
				pNode = pNodeNext
				pNode_list.append(pNode)
			except:
				print("Random Generation with", len(points), "valid points did not yield a result, trying again")
				return False
			
		return True



	def plan_path(self):
		self.valid_start = True
		self.valid_goal = True
		numOfPoints = 100 # Max number of random points sampled
		iterations = 0
		start_time = time.time()
		isValid = False
		while (isValid == False and self.valid_start and self.valid_goal):
			previous_nodes, shortest_path, points = self.generate_path(numOfCoords=numOfPoints)
			if (self.valid_start and self.valid_goal):
				isValid = self.check_valid_path(previous_nodes, points)
				iterations += 1
			if iterations >= 20:
				print("Max iterations reached: ", iterations)
				return
		if not (self.valid_start & self.valid_goal):
			print("Start or Goal not valid","Start: ", self.valid_start,"Goal: ", self.valid_goal)
			print("Pick a different start or goal")
		else:
			end_time = time.time()
			print("Goal Reached!")
			print("Computation took ", end_time - start_time, "seconds.")
			print("With ",iterations, "iterations.")
			print("Using", len(points), "random points.")

def dijkstra(graph, source):


	# Graph is a set of nodes as (x,y) values
	# edges is a dictionary of points as keys with to_nodes and their lengths as values

	# Source is the start node

	unvisited_nodes = list(graph.nodes)	

	shortest_path = {}
	previous_nodes = {}

	INFINITY = float('Infinity')

	for node in unvisited_nodes:
		shortest_path[node] = INFINITY

	shortest_path[source] = 0	

	while unvisited_nodes:
		current_min_node = None
		for node in unvisited_nodes:
			if current_min_node == None:
				current_min_node = node
			elif shortest_path[node] < shortest_path[current_min_node]:
				current_min_node = node

		neighbors = []

		for _, v in graph.edges[current_min_node].items():
			neighbors.append(tuple([v.to_node,v.length]))

		for neighbor in neighbors:
			tentative_value = shortest_path[current_min_node] + neighbor[1]
			if tentative_value < shortest_path[neighbor[0]]:
				shortest_path[neighbor[0]] = tentative_value
				previous_nodes[neighbor[0]] = current_min_node

		unvisited_nodes.remove(current_min_node)

	return previous_nodes, shortest_path


# bresenham algorithm for line generation on grid map
# from http://www.roguebasin.com/index.php?title=Bresenham%27s_Line_Algorithm
def bresenham(x1, y1, x2, y2):
    dx = x2 - x1
    dy = y2 - y1
 
    # Determine how steep the line is
    is_steep = abs(dy) > abs(dx)
 
    # Rotate line
    if is_steep:
        x1, y1 = y1, x1
        x2, y2 = y2, x2
 
    # Swap start and end points if necessary and store swap state
    swapped = False
    if x1 > x2:
        x1, x2 = x2, x1
        y1, y2 = y2, y1
        swapped = True
 
    # Recalculate differentials
    dx = x2 - x1
    dy = y2 - y1
 
    # Calculate error
    error = int(dx / 2.0)
    ystep = 1 if y1 < y2 else -1
 
    # Iterate over bounding box generating points between start and end
    y = y1
    points = []
    for x in range(x1, x2 + 1):
        coord = (y, x) if is_steep else (x, y)
        points.append(coord)
        error -= abs(dy)
        if error < 0:
            y += ystep
            error += dx
 
    # Reverse the list if the coordinates were swapped
    if swapped:
        points.reverse()

    # print points
    return points
