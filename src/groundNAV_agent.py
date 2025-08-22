import numpy as np
import cv2
import open3d as o3d 
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio
from scipy.spatial.transform import Rotation as R
from scipy.spatial import cKDTree
from matplotlib.path import Path
import os 

from colmapParsingUtils import *

class gNAV_agent:
	"""
	Project for enhancing autonomous ground navigation of aerial vehicles
	Initialization of agent 
	Inputs: Reference image (satellite), SfM solution (COLMAP), selected images
	"""
	def __init__(self, images_colm, cameras_colm, pts3d_colm, images, sat_ref):
		self.images_c_loc = images_colm		# Location of images file
		self.cameras_c_loc = cameras_colm	# Location of cameras file
		self.pts3d_c_loc = pts3d_colm 		# Location of ptd3d file
		self.imss = images 					# Location of specific image folder 
		self.sat_ref = cv2.imread(sat_ref) 	# Satellite reference image
		self.sat_ref = cv2.cvtColor(self.sat_ref, cv2.COLOR_BGR2GRAY)
		self.read_colmap_data()
		self.image_parsing()
		self.sat_im_init()
		self.im_pts_best_guess = {}
		self.ssds_curr = {}
		self.ssds1_curr = {}

	def read_colmap_data(self):
		self.images_c = read_images_text(self.images_c_loc)
		self.cameras_c = read_cameras_text(self.cameras_c_loc)
		self.pts3d_c = read_points3D_text(self.pts3d_c_loc)

	def image_parsing(self):
		""" 
		Gets the specific image IDs according to COLMAP file. Useful 
		for grabbing transformations later 
		Input: class
		Output: image IDs
		"""
		self.images_dict = {}
		self.im_pts_2d = {}
		self.im_mosaic = {}

		# Specify image ordering 
		self.im1 = self.imss + '/IMG_9475.JPEG'
		self.im2 = self.imss + '/IMG_9464.JPEG'
		self.im3 = self.imss + '/IMG_9467.JPEG'
		self.im4 = self.imss + '/IMG_9473.JPEG'
		self.im5 = self.imss + '/IMG_9476.JPEG'
		self.im6 = self.imss + '/IMG_9446.JPEG'
		self.im7 = self.imss + '/IMG_9520.JPEG'
		self.im8 = self.imss + '/IMG_9531.JPEG'
		self.im9 = self.imss + '/IMG_9542.JPEG'
		self.im10 = self.imss + '/IMG_9576.JPEG'

		self.images = [self.im1, self.im2, self.im3, self.im4, self.im5,
		self.im6, self.im7, self.im8, self.im9, self.im10]

		im_ids = np.zeros((len(self.images)), dtype=int)
		# print(self.images_c.items())

		for i, image_path in enumerate(self.images):
			# Read images and create new image variables 
			self.read_image_files(image_path,i)
			# Grab file name on end of path
			filename = image_path.split('/')[-1]
			# print(filename)
			
			# Look up corresponding ID
			for img_c_id, img_c in self.images_c.items():
				if img_c.name.startswith(filename):
					im_ids[i] = img_c_id
					break

		self.im_ids = im_ids

	def read_image_files(self, image_path, i):
		"""
		Reads in each image file to be parsed through later
		Inputs: filename, picture ID number
		Output: variable created according to image number
		"""
		image = cv2.imread(image_path)
		self.images_dict[i] = image

	def sat_im_init(self):
		"""
		Initializing the satellite reference image and creating a cloud and RGB array
		NOTE: The image is already in grayscale. Keeping in RGB format for open3d
		Input: reference image 
		Output: 3xn array of points (z=1), and 3xn array of colors (grayscale)
		"""
		cols, rows = self.sat_ref.shape
		x, y = np.meshgrid(np.arange(rows), np.arange(cols))
		ref_pts = np.stack([x.ravel(), y.ravel(), np.ones_like(x).ravel()], axis=1)

		gray_vals = self.sat_ref.ravel().astype(np.float32)
		ref_rgb = np.stack([gray_vals]*3, axis=1)
		ref_rgb /= 255

		# ADDING A SHIFT TO FULL SAT IMAGE - comment out for old version
		ref_pts -= np.array([700,600,0])

		self.ref_pts = ref_pts
		self.ref_rgb = ref_rgb