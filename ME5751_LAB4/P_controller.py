#!/usr/bin/python
# -*- coding: utf-8 -*-

from E160_state import *
from E160_robot import *
import math
import time
import numpy as np
import random


class P_controller:

	def __init__(self, robot, logging = True):
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

		if(logging == True):
			self.robot.make_headers(['pos_X','posY','posZ','vix','viy','wi','vr','wr'])

		self.set_goal_points()

	#Edit goal point list below, if you click a point using mouse, the points programmed
	#will be washed out
	def set_goal_points(self):

		# Our goal point tests
		self.robot.state_des.add_destination(x=10,y=10,theta=math.pi/8) #goal point 1
		self.robot.state_des.add_destination(x=20,y=20,theta=-math.pi/4) #goal point 2
		self.robot.state_des.add_destination(x=20,y=30,theta=math.pi/2) #goal point 3
				

	# Unused function for transforming 2D coordinate frame
	def transform_to_goal(self, x, y, theta):
		t = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
		arr =np.dot(t,np.array([x,y]))

		return arr[0], arr[1]

	# Function for updating wheel velocity based on max speed
	# This function is not optimized but it does satisfy all cases tried
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


	def track_point(self):

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
				print(d_theta, c_theta)
				self.robot.set_motor_control(.0, .0)  # stop the motor
				return True
			else:
				print("one goal point reached, continute to next goal point")
		
		return False