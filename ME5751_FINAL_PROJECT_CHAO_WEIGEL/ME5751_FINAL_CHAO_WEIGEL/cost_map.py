from itertools import count
import cv2
import numpy as np
import math
from PIL import Image, ImageTk
from queue import Queue
import sys

class cost_map:
	def __init__(self,graphics):
		self.graphics = graphics
		# self.graphics.scale = 400 # This value should be same as the pixel value of the image
		self.inflation_radius = 18 # radius of our robot is 18 pixel or cm
		self.graphics.environment.robots[0].set_bot_size(body_cm = 2*self.inflation_radius)
		#self.graphics.environment.width/height = 2
		self.map_width = int(self.graphics.environment.width*self.graphics.scale)
		self.map_height = int(self.graphics.environment.height*self.graphics.scale)
		try:
			self.load_map(map = "maps/testmap10.png") #load map
		except:
			self.graphics.show_map_button.configure(state="disabled")
			print ("no map loaded") #if fail to find the map png
			return
		self.show_map()
		self.compute_costmap()
		self.save_costmap_asbitmap()


		# self.show_costmap()
		self.save_vis_map(map = "maps/testcostmap10.png")

	#load occupancy grid into self.map
	#self.map is a numpy 2d array
	#initialize self.costmap, a numpy 2d array, same as self.map
	def load_map(self,map="maps/testmap10.png"):
		self.map_img = Image.open(map).convert('L')
		self.map_img = self.map_img.resize((int(self.map_width),int(self.map_height)),Image.ANTIALIAS)
		# self.graphics.draw_map(map_img=self.map_img)
		self.map = cv2.imread(map,cv2.IMREAD_GRAYSCALE)
		print (self.map.dtype)
		print ("Loaded map dimension: %d x %d pixel"%(self.map.shape[0],self.map.shape[1]))
		self.map = cv2.resize(self.map, dsize=(int(self.map_width),int(self.map_height)), interpolation=cv2.INTER_CUBIC)
		self.vis_map=np.copy(self.map) #map for visualization
		self.distmap=np.copy(self.map).astype(np.float)
		self.costmap=np.copy(self.map).astype(np.float)

	#save your costmap into a grayscale image
	def save_vis_map(self,map="maps/costmap10.png"):
		save_img = Image.fromarray(self.vis_map)
		save_img.save(map)

	def show_vis_map(self):
		self.get_vis_map()
		self.vis_map_img=Image.frombytes('L', (self.vis_map.shape[1],self.vis_map.shape[0]), self.vis_map.astype('b').tostring())
		self.graphics.draw_map(map_img=self.vis_map_img)

	#display costmap on the dialogue window
	def show_costmap(self):
		self.costmap_img=Image.frombytes('L', (self.costmap.shape[1],self.costmap.shape[0]), self.costmap.astype('b').tostring())
		self.graphics.draw_map(map_img=self.costmap_img)

	#display binary occupancy grid on the dialogue window 
	def show_map(self):
		self.graphics.draw_map(map_img=self.map_img)


	#This is the function you really should update!

	def inflate_map(self, inflation):
		self.costmap[:,:] = 0
		m,n = np.shape(self.map)
		for x in range(m):
			for y in range(n):
				if self.map[x,y] <= 254 and x-inflation >=0 and x+inflation+1 <m and y-inflation >=0 and y+inflation+1<n:
					self.costmap[x-inflation:x+inflation+1:1, y-inflation:y+inflation+1:1] = 255
				elif self.map[x,y]==0:
					for i in range(0,inflation):
						if x-i >=0 and x+i+1 <m and y-i >=0 and y+i+1<n:
							self.costmap[x-i:x+i+1:1, y-i:y+i+1:1] = 255

	def inflate_edges(self, inflation):
		m,n = np.shape(self.map)

		## LEFT EDGE
		for x in range(m):
			if self.map[x,0] <= 254:
				self.costmap[x,0:inflation] = 255
			if self.map[x,n-1] <= 254:
				self.costmap[x,n-inflation:n] = 255


		##TOP EDGE
		for y in range(n):
			if self.map[0,y] <= 254:
				self.costmap[0:inflation, y] = 255
			if self.map[m-1,y] <= 254:
				self.costmap[m-inflation:m,y] = 255

		##RIGHT EDGE

		##BOTTOM EDGE

		#print(self.map)
		#print(self.costmap)
		#self.costmap = self.costmap + self.map

	def brushfire(self):
		cost_const = 3
		q = []
		cost = 1
		m,n = np.shape(self.map)
		d = np.zeros((m,n))
		
		for x in range(m):
			for y in range(n):
				if self.costmap[x,y] == 255:
					q.append([x,y])
					d[x][y] = 1
		q.append([m,n])
		#print(q)
		#print(d)
		while len(q) > 0:
			[i,j] = q[0]
			q.pop(0)

			#LEFT
			if j-1 >= 0 and self.map[i][j-1] == 255 and d[i][j-1] != 1:
				self.costmap[i][j-1] = 255 - cost*cost_const
				d[i][j-1] = 1
				q.append([i,j-1])
			#LEFT UP
			#if i-1 >= 0 and j-1 >= 0 and self.map[i-1][j-1] == 255 and d[i-1][j-1] != 1:
			#	self.costmap[]
			#UP
			if i-1 >= 0 and self.map[i-1][j] == 255 and d[i-1][j] != 1:
				self.costmap[i-1][j] = 255 - cost*cost_const
				d[i-1][j] = 1
				q.append([i-1,j])
			#RIGHT
			if j+1 < n and self.map[i][j+1] == 255 and d[i][j+1] != 1:
				self.costmap[i][j+1] = 255 - cost*cost_const
				d[i][j+1] = 1
				q.append([i,j+1])
			#DOWN
			if i+1 < m and self.map[i+1][j] == 255 and d[i+1][j] != 1:
				self.costmap[i+1][j] = 255 - cost*cost_const
				d[i+1][j] = 1
				q.append([i+1,j])
			#COST VALUE
			if q[0] == [m,n] and len(q) == 1:
				q.pop(0)
			elif q[0] == [m,n]:
				q.pop(0)
				cost = cost + 1
				q.append([m,n])
			
			if cost*cost_const >= 255:
				cost = 255
				cost_const = 1

			#print(q)
		print(self.costmap)
		
				
	def compute_costmap(self):
		inflation = 12
		self.inflate_map(inflation)
		self.inflate_edges(inflation)
		self.brushfire()

		np.savetxt("costmap6.csv", self.costmap, delimiter=",")

	def save_costmap_asbitmap(self):
		map = np.uint8(self.map)
		save_img = Image.fromarray(map)
		save_img = save_img.convert("L")
		save_img.save("maps/costmapforobstacleavoidance.png")

	#scale costmap to 0 - 255 for visualization
	def get_vis_map(self):
		self.vis_map = np.uint8(255-self.costmap)
		np.savetxt("Log/vismap.txt",self.vis_map)
