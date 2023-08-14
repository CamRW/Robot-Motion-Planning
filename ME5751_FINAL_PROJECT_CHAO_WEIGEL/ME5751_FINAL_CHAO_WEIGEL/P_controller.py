#!/usr/bin/python
# -*- coding: utf-8 -*-

from E160_state import *
from E160_robot import *
import math
import time
import numpy as np
import random
import Path
import stanley_controller
import cv2

class P_controller:

	def __init__(self, robot, logging = False):
		self.robot = robot  # do not delete this line
		self.kp = 0  # k_rho
		self.ka = 0  # k_alpha
		self.kb = 0  # k_beta

		# Made these global for use across many functions
		# Not sure if I prefer this though, could cause
		# unintended consequences
		self.rho = 0
		self.alpha = 0
		self.beta = 0

		self.logging = logging

		self.bodyL = 12
		self.wheelr = 3

		self.phi_l = 0
		self.phi_r = 0
		self.max_steer=np.radians(32.0)
		self.max_vel = .925*60

		if(logging == True):
			self.robot.make_headers(['pos_X','posY','posZ','vix','viy','wi','vr','wr'])

		self.set_goal_points()

		self.cx = self.robot.state_des.x
		
		self.cy = self.robot.state_des.y
		self.cyaw = self.robot.state_des.theta
		self.target_idx = 1

		self.last_idx = len(self.robot.state_des.x)-1

		self.costmap = cv2.imread("maps/costmapforobstacleavoidance.png",cv2.IMREAD_GRAYSCALE)

		self.costmap2 = self.costmap.copy()
		self.iter = 0
		self.v = 30




		#self.add_goal_points()

	#Edit goal point list below, if you click a point using mouse, the points programmed
	#will be washed out
	def set_goal_points(self):

		# Our goal point tests
		#self.robot.state_des.add_destination(0,0,0)
		#self.robot.state_des.add_destination(x=-250,y=-150,theta=0) #goal point 1
		#self.robot.state_des.add_destination(x=50,y=50,theta=math.pi/4) #goal point 2
		#self.robot.state_des.add_destination(x=20,y=30,theta=math.pi/2) #goal point 3
		pass

	def add_goal_points(self, path):
		for p in path.poses:
			#print(p.map_i, p.map_j)
			self.robot.state_des.add_destination(x=p.map_i,y=p.map_j,theta=p.theta)
				

	# Unused function for transforming 2D coordinate frame
	def transform_to_goal(self, x, y, theta):
		t = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
		arr =np.dot(t,np.array([x,y]))

		return arr[0], arr[1]

	# Function for updating wheel velocity based on max speed
	# This function is not optimized but it does satisfy all cases tried
	def check_wheel_spd_angle(self, c_v, c_w, steering_angle):
		WB = 20
		TW = 16
		R = 3
		right_wheel_angle = math.atan((WB*math.tan(steering_angle))/(WB - 0.5*TW*math.tan(steering_angle)))
		right_wheel_angle = right_wheel_angle*(180.0/math.pi)
		wheel_speed = (c_v - 0.5*TW*c_w)/R

		#print("Right Wheel angle",right_wheel_angle)
		#print("Right wheel rads/s", wheel_speed)

		return right_wheel_angle, wheel_speed
		

	## UNUSED FOR FINAL PROJECT
	def check_wheel_speed(self, c_v, c_w):

		# Get wheel velocities from inputs
		phi_l_tmp = (c_v + self.bodyL*c_w)/self.wheelr
		phi_r_tmp = (c_v - self.bodyL*c_w)/self.wheelr

		# Set local gains to global gains
		ka = self.ka
		kb = self.kb
		kp = self.kp

		# Iterative process to reduce gains if needed to maintain 16 rad/sec limit
		while ((phi_l_tmp < -16.0 or phi_r_tmp < -16.0) or (phi_l_tmp > 16.0 or phi_r_tmp > 16.0)):
			ka = ka/1.1
			kb = kb/1.1
			kp = kp/1.1
			#print(ka, kb, kp)

			# Reassign velocities based on new gains
			c_v = kp*self.rho
			c_w = ka*self.alpha + kb*self.beta

			# Reassign wheel velocities based on gains
			phi_l_tmp = (c_v + self.bodyL*c_w)/self.wheelr
			phi_r_tmp = (c_v - self.bodyL*c_w)/self.wheelr

		# Return new gains and wheel velocities
		return ka, kb, kp, phi_l_tmp, phi_r_tmp

	# UNUSED FOR FINAL PROJECT
	def track_point1(self):

		# All d_ means destination

		(d_posX, d_posY, d_theta) = self.robot.state_des.get_des_state()  # get next destination configuration

		# All c_ means current_

		(c_posX, c_posY, c_theta) = self.robot.state.get_pos_state()  # get current position configuration
		(c_vix, c_viy, c_wi) = self.robot.state.get_global_vel_state() #get current velocity configuration, in the global frame
		(c_v, c_w) = self.robot.state.get_local_vel_state() #get current local velocity configuration

		# Most of your program should be here, compute rho, alpha and beta using d_pos and c_pos
		# set new c_v = k_rho*rho, c_w = k_alpha*alpha + k_beta*beta
		# Random k values chosen using stable sytem

		
		self.kp = 3.5
		self.ka = 10.0
		self.kb = -1

		# Most of your program should be here, compute rho, alpha and beta using d_pos and c_pos
		# set new c_v = k_rho*rho, c_w = k_alpha*alpha + k_beta*beta

		# Deltas
		delta_x = d_posX - c_posX
		delta_y = d_posY - c_posY
		theta = c_theta - d_theta


		# Initial Rho and Alpha vals
		self.rho = math.sqrt(delta_x**2 + delta_y**2)
		self.alpha = -c_theta + math.atan2(delta_y, delta_x)

		# Used for keeping Alpha/Beta in [-Pi, Pi]
		if(self.alpha>math.pi):
			self.alpha -= math.pi*2
		elif(self.alpha<-math.pi):
			self.alpha += math.pi*2

		# Check to see if the waypoint is in front or behind the robot
		if (self.alpha <= 1.57 and self.alpha > -1.57):
			pass
		else:
			# Have robot drive backwards
			self.rho = -self.rho
			self.alpha = -c_theta + math.atan2(-delta_y, -delta_x)

		# Set Beta
		self.beta = -theta - self.alpha
		

		# Find our Linear and Angular Velocities
		c_w = self.ka*self.alpha + self.kb*self.beta
		c_v = self.kp*self.rho

		# Check if we're moving to fast, change gains if we need to
		# Also report left and right wheel speed with phi_l phi_r
		ka, kb, kp, phi_l, phi_r = self.check_wheel_speed(c_v, c_w)

		# If gains have changed, these gains will be different so we need
		# to update our velocities
		c_w = ka*self.alpha + kb*self.beta
		c_v = kp*self.rho

		#print(self.rho, self.alpha, self.beta)

		# self.robot.set_motor_control(linear velocity (cm), angular velocity (rad))
		self.robot.set_motor_control(c_v, c_w)  # use this command to set robot's speed in local frame
		
		# you need to write code to find the wheel speed for your c_v, and c_w, the program won't calculate it for you.
		self.robot.send_wheel_speed(phi_l = phi_l,phi_r=phi_r) #unit rad/s


		# use the following to log the variables, use [] to bracket all variables you want to store
		# stored values are in log folder
		if self.logging == True:
			self.robot.log_data([c_posX,c_posY,c_theta,c_vix,c_viy,c_wi,c_v,c_w])

		if abs(self.rho) < .1 and abs(self.alpha) < .02: #you need to modify the reach way point criteria
			if(self.robot.state_des.reach_destination()):
				print("final goal reached")
				#print(d_theta, c_theta)
				self.robot.set_motor_control(.0, .0)  # stop the motor
				return True
			else:
				print("one goal point reached, continute to next goal point")
		
		return False


	# CALCULATES ANGULAR VELOCITY
	def calc_ang_vel(self, c_v, steering_angle):
		l = 16
		c_w = c_v/l*math.tan(steering_angle)

		return c_w

	# STANLEY CONTROLLER
	def track_point(self):


		self.costmap = cv2.imread("maps/costmapforobstacleavoidance.png",cv2.IMREAD_GRAYSCALE)


		self.cx = self.robot.state_des.x
		
		self.cy = self.robot.state_des.y
		self.cyaw = self.robot.state_des.theta

		

		(c_posX, c_posY, c_theta) = self.robot.state.get_pos_state()  # get current position configuration


		# TWO OBSTACLE DETECTION LAYERS
		obstacle, turn = self.dwa(r=25,rx=round(c_posX), ry=round(c_posY),rtheta=c_theta)
		obstacle2, turn2 = self.dwa(r=17,rx=round(c_posX), ry=round(c_posY),rtheta=c_theta)


		state = stanley_controller.State(x=c_posX, y=c_posY, yaw=c_theta, v=self.v)

		di, self.target_idx = stanley_controller.stanley_control(state, self.cx, self.cy, self.cyaw, self.target_idx)


		di = np.clip(di, -self.max_steer, self.max_steer)
		self.v = self.max_vel*abs(math.cos(di)**2)

		c_w = self.calc_ang_vel(self.v, di)


		if(obstacle2):
			self.v = self.max_vel*abs(math.cos(turn)**2)
			angle, spd = self.check_wheel_spd_angle(-self.v, c_w, turn)
			self.robot.send_wheel_speed(phi_l = angle,phi_r=spd)
			self.robot.set_motor_control(-self.v, 0)
			#print("BACKING UP",c_posX,c_posY)

		if (obstacle or self.iter != 0) and (self.iter < 3):
			self.v = self.max_vel*abs(math.cos(turn)**2)
			c_w = self.calc_ang_vel(self.v,turn)
			self.iter = self.iter+1
			#print("OBSTACLE")
			angle, spd = self.check_wheel_spd_angle(self.v, c_w, turn)
			self.robot.send_wheel_speed(phi_l = angle,phi_r=spd)
			self.robot.set_motor_control(self.v, c_w)
		else:
			self.iter = 0
			angle, spd = self.check_wheel_spd_angle(self.v, c_w, di)
			self.robot.send_wheel_speed(phi_l = angle,phi_r=spd)
			self.robot.set_motor_control(self.v, c_w)

		
		goalx = self.robot.state_des.x[-1]
		
		goaly = self.robot.state_des.y[-1]

		if abs(goalx-c_posX) < 10.0 and abs(goaly-c_posY) < 10.0:
			self.robot.set_motor_control(.0, .0)
			print("Goal reached")

			return True
		


		if self.target_idx == self.last_idx:
			(d_posX, d_posY, d_theta) = self.robot.state_des.get_des_state()
			self.robot.set_motor_control(.0, .0)
			print("Goal reached")

			return True


		cv2.imshow('image',self.costmap)
		cv2.waitKey(1)
		return False



	def dwa(self, r, rx, ry,rtheta):

		# Creating equally spaced 349 data in range 0 to 2*pi
		theta = np.linspace(-np.pi/2, np.pi/2, 180)
		# theta = np.linspace(0, 2 * np.pi, 349)

		# Setting radius
		radius = r

		# Generating x and y data
		xidx = ((np.round(radius * np.cos(rtheta+theta)) + rx)+250).astype(int)
		yidx = (-(np.round(radius * np.sin(rtheta+theta)) + ry)+250).astype(int)

		xidx[xidx > 499] = 499
		yidx[yidx > 499] = 499

		xidx[xidx < 0] = 0
		yidx[yidx < 0] = 0

		#print(xidx)

		self.costmap[yidx,xidx] = 0

		# self.costmap[yidx[0],xidx[0]] = 0 ## FRONT
		self.costmap[yidx[round(len(yidx)/2)], xidx[round(len(xidx)/2)]] = 0 ## MIDDLE

		right_yidxs = np.array([yidx[round(len(yidx)/2):]])

		right_xidxs = np.array([xidx[round(len(xidx)/2):]])

		left_yidxs = np.array([yidx[:round(len(yidx)/2)-1]])

		left_xidxs = np.array([xidx[:round(len(xidx)/2-1)]])

		if 0 in self.costmap2[right_yidxs, right_xidxs]:
			# TURN LEFT
			pass
			return True, -self.max_steer
		elif 0 in self.costmap2[left_yidxs,left_xidxs]:
			# TURN RIGHT
			pass
			return True, self.max_steer # TURN 40 degrees

		else:
			return False, 0





