from re import S
import cv2
import numpy as np
import math
import warnings
from PIL import Image
import datetime

from bsTree import *
from Path import *

class path_planner:
	def __init__(self,graphics):
		self.graphics = graphics
		# self.graphics.scale = 400 #half pixel number on canvas, the map should be 800 x 800
		# self.graphics.environment.robots[0].set_bot_size(body_cm = 2*self.inflation_radius)
		#self.graphics.environment.width/height = 2

		self.costmap = self.graphics.map
		self.map_width = self.costmap.map_width
		self.map_height = self.costmap.map_height

		self._init_path_img()
		self.path = Path()

	
		self.set_start(world_x = 0, world_y = 0)
		self.set_goal(world_x = -56, world_y = 210, world_theta = 0)

		self.plan_path()
		self._show_path()


	def set_start(self, world_x = 0, world_y = 0, world_theta = 0):
		self.start_state_map = Pose()
		map_i, map_j = self.world2map(world_x,world_y)
		print("Start with %d, %d on map"%(map_i,map_j))
		self.start_state_map.set_pose(map_i,map_j,world_theta)


	def set_goal(self, world_x, world_y, world_theta = 0):
		self.goal_state_map = Pose()
		map_i, map_j = self.world2map(world_x, world_y)
		print ("goal is %d, %d on map"%(map_i,map_j))
		self.goal_state_map.set_pose(map_i, map_j, world_theta)


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
		
		# If you want to output an image of map and path, un-comment the following two lines

		# self.path_img = toimage(self.map_img_np)
		# self.path_img.show()
		
		self.graphics.draw_path(self.path_img)

	def heuristicMan(self, pose_a, pose_b):
		# Manhattan Distance
		return abs(pose_a.map_i - pose_b.map_i) + abs(pose_a.map_j - pose_b.map_j)

	def heuristicEuc(self, pose_a, pose_b):
		# Euclidean Distance
		return math.sqrt((pose_a.map_i - pose_b.map_i)**2 + (pose_a.map_j - pose_b.map_j)**2)

	def heuristic3(self, pose_a, pose_b, cellcost):
		return 1.4*(((pose_a.map_i - pose_b.map_i)**2 + (pose_a.map_j - pose_b.map_j)**2)**(1/3.5))*(1-cellcost/255)

	def plan_path(self):

		self.path.clear_path()
		m,n = self.map_width, self.map_height


		# Start and Goal
		start = self.start_state_map
		goal = self.goal_state_map

		# Mu for varying our cost mix
		mu = 2

		# Initialize node objects and cost values
		start_node = Node(None, start)
		start_node.g = start_node.h = start_node.f = 0
		goal_node = Node(None, goal)

		# Yet to visit it the list we use to see which nodes to lookat
		yet_to_visit = []

		# Visited is the list of visited nodes
		visited = []
		yet_to_visit.append(start_node)

		# This is used to set a max number of iterations for testing difficult paths
		outer_iterations = 0
		max_iterations = 15000


		# 8-connection movement
		move  =  [[-1, 0], # go up
				[-1,-1], # up left
				[0,-1],
				[1,-1],
				[1,0],
				[1,1],
				[0,1],
				[-1,1]]


		# These arrays are used for checking certain conditions
		# We use arrays because array element access is faster
		# than list searching and memory is cheap
		d = np.zeros((m,n)) # Used for seeing how many times we hit a cell
		j = np.zeros((m,n)) # Used for seeing if we have visited a cell
		k = np.zeros((m,n)) # Used for storing cost at a cell location
		l = np.zeros((m,n)) # Used for seeing if this cell is in the yet_to_visit list


		# While there's still elements to check
		while len(yet_to_visit) > 0:
			outer_iterations += 1
			current_node = yet_to_visit[0]
			current_index = 0

			
			# # See if we need to change our branch
			for index, item in enumerate(yet_to_visit):
				if item.f < current_node.f:

					# Update the current branch
					current_node = item
					current_index = index

			# Iteration break
			if outer_iterations > max_iterations:
				print("Too many iterations")
				break

			# Remove our current node from yet to visit
			yet_to_visit.pop(current_index)
			# Add to visited
			visited.append(current_node)
			# Update grid contents
			j[current_node.pose.map_i,current_node.pose.map_j] = 1
			l[current_node.pose.map_i,current_node.pose.map_j] = 0

			# Found goal break
			if current_node.pose.map_i == goal_node.pose.map_i and current_node.pose.map_j == goal_node.pose.map_j:
				print("Found Goal")
				break

			# Empty list for getting 8-connection "children"
			children = []

			# Generate children
			for new_position in move:
				node_position = [current_node.pose.map_i + new_position[0], current_node.pose.map_j + new_position[1]]

				# If children positions are invalid
				if (node_position[0] > (m - 1) or 
					node_position[0] < 0 or 
					node_position[1] > (n -1) or 
					node_position[1] < 0):
					continue

				# If the child is a wall
				if self.costmap.costmap[node_position[0],node_position[1]] == 255:
					continue

				# Create a node object from the child and add it to the children list
				new_node = Node(current_node, Pose(node_position[0],node_position[1],0))
				children.append(new_node)

			# Loop through children
			for child in children:

				# Create the f, g, and h values

				# Cost of child is the cost to get here plus a scaling value based on how many times we've visited this cell
				# This was done in hopes to move around walls to combat the euclidean distance dominating the total cost
				child.g = child.parent.g + self.costmap.costmap[child.pose.map_i,child.pose.map_j]*d[child.pose.map_i,child.pose.map_j]*10

				# Euclidean distance because we are using an 8-connection style
				child.h = self.heuristicEuc(child.pose, goal_node.pose)
				#child.h = self.heuristic3(goal_node.pose, child.pose, self.costmap.costmap[child.pose.map_i,child.pose.map_j])

				# Apply a mu value to the cost
				child.f = child.g + child.h*mu

				# We have now looked at this grid cell one more time
				d[child.pose.map_i,child.pose.map_j] += 1
				

				# If child is visited and the cost we just calculated is higher than the cost previously calculated
				if j[child.pose.map_i,child.pose.map_j] == 1 and child.f > k[child.pose.map_i,child.pose.map_j]:
					continue

				# If child is in the yet_to_be_visited list and the cost we just calculated is higher
				if l[child.pose.map_i,child.pose.map_j] == 1 and child.f > k[child.pose.map_i,child.pose.map_j]:
					continue

				# Else, append the child, update the l and k grids
				yet_to_visit.append(child)
				l[current_node.pose.map_i,current_node.pose.map_j] = 1
				k[child.pose.map_i,child.pose.map_j] = child.f

		# Helper function to rebuild path
		self.return_path(current_node)
		# Save path to file, currently appends to the csv
		timestamp = '{:%Y-%m-%d-%H:%M:%S}'.format(datetime.datetime.now())
		self.path.save_path(file_name="Log/path-"+timestamp+".csv")
			
	#This function return the path of the search
	def return_path(self,current_node):
		current = current_node
		counter = 0
		while current is not None:
			self.path.insert_pose(current.pose)
			current = current.parent
			counter +=1
		print("Total Path Length: ", counter)

# Simple node class to build parent/child tree, store poses and costs
class Node:
	def __init__(self, parent=None, pose=None):
		self.parent = parent
		self.pose = pose
		
		self.g = 0
		self.h = 0
		self.f = 0

		def __eq__(self, other):
			return self.pose == other.pose



# bresenham algorithm for line generation on grid map
# from http://www.roguebasin.com/index.php?title=Bresenham%27s_Line_Algorithm
def bresenham(x1, y1, x2, y2):
    """Bresenham's Line Algorithm
    Produces a list of tuples from start and end
 
    >>> points1 = get_line((0, 0), (3, 4))
    >>> points2 = get_line((3, 4), (0, 0))
    >>> assert(set(points1) == set(points2))
    >>> print points1
    [(0, 0), (1, 1), (1, 2), (2, 3), (3, 4)]
    >>> print points2
    [(3, 4), (2, 3), (1, 2), (1, 1), (0, 0)]
    """
    # Setup initial conditions

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
