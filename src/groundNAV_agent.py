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
import open3d as o3d

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
		# Initial scene points and RGB data
		self.grab_pts(self.pts3d_c)
		# Initialize best guess and SSDs
		self.im_pts_best_guess = {}
		self.ssds_curr = {}
		# self.ssds1_curr = {}
		# Ground plane points - chosen MANUALLY
		self.pts_gnd_idx = np.array([25440, 25450, 25441, 25449, 25442, 25445, 103922, 103921, 103919, 103920])

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
		# image = cv2.imread(image_path)
		# self.images_dict[i] = image
		if os.path.exists(image_path):
			image = cv2.imread(image_path)
			if image is not None:
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

	def grab_pts(self, pts3d):
		"""
		Grabbing raw point cloud and RGB data from scene data
		"""
		# Loop through pts3d dictionary using keys
		raw_pts = [pts3d[key].xyz for key in pts3d.keys()]
		raw_rgb = [pts3d[key].rgb for key in pts3d.keys()]

		# Stack into numpy array 
		scene_pts =  np.vstack(raw_pts)
		rgb_data = np.vstack(raw_rgb)
		# Normalize rgb data 
		rgb_data = rgb_data/255 

		self.scene_pts = scene_pts
		self.scene_rgb = rgb_data

	def pose_scene_visualization(self, vis):
		"""
		Creating a visualization of pose estimations and sparse point cloud
		Input: vis (open3d)
		Output: vis (with pose estimations and point cloud)
		"""
		# Add origin axes
		axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1)
		vis.add_geometry(axes)

		# Add each pose estimate frame
		self.grab_poses(self.images_c)
		for p in self.poses:
			axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1).transform(p)
			vis.add_geometry(axes)

		# Add sparse point cloud
		scene_cloud = o3d.geometry.PointCloud()
		scene_cloud.points = o3d.utility.Vector3dVector(self.scene_pts)
		scene_cloud.colors = o3d.utility.Vector3dVector(self.scene_rgb)
		vis.add_geometry(scene_cloud)

		# Size options (jupyter gives issues when running this multiple times, but it looks better)
		render_option = vis.get_render_option()
		render_option.point_size = 2

		# Run and destroy visualization 
		vis.run()
		vis.destroy_window()

	def grab_poses(self, images_c):
		"""
		Grabs initial image poses for visualizations
		Input: Image data
		Output: Poses
		"""
		poses = []
		# Loop through each image
		for i in images_c:
			# Get quaternion and translation vector
			qvec = images_c[i].qvec
			tvec = images_c[i].tvec[:,None]
			# print(tvec)
			t = tvec.reshape([3,])
			# Create rotation matrix
			Rotmat = qvec2rotmat(qvec) # Positive or negative does not matter
			# print("\n Rotation matrix \n", Rotmat)
			# Create 4x4 transformation matrix with rotation and translation
			tform_mat = np.eye(4)
			tform_mat[:3, :3] = Rotmat
			tform_mat[:3, 3] = t
			w2c = tform_mat
			c2w = np.linalg.inv(w2c)
			poses.append(c2w)
		poses = np.stack(poses)
		self.poses = poses


	def grav_SVD(self, pts_gnd):
		"""
		Getting the gravity vector for a set of points on the ground plane
		Input: Indices for the ground plane pts
		Output: Gravity vector 
		Note: potentially automate ground point process in the future 
		"""

		# Subtract centroid for SVD
		centroid = np.mean(pts_gnd, axis=0)
		centered_points = pts_gnd - centroid

		# Singular value decomposition (SVD)
		U, S, Vt = np.linalg.svd(centered_points)

		grav_vec = Vt[-1,:]

		return grav_vec

	def height_avg(self, pts_gnd, origin):
		"""
		Get the initial height of the origin above the ground plane 
		Input: Indices for the ground plane pts
		Output: Average h_0
		"""

		# Multiple h0's
		h0s = np.zeros((len(pts_gnd)))
		for i in range(len(pts_gnd)):
			h0i = np.dot(self.grav_vec, pts_gnd[i]-origin)
			h0s[i] = h0i

		# Average 
		h_0 = np.mean(h0s)

		return h_0

	def set_ref_frame(self, pts_gnd_idx):
		"""
		Defines a reference coordinate frame for the matching process
		Input: ground plane points 
		Output: reference frame transformation matrix
		"""
		self.origin_w = np.array([0,0,0])
		self.pts_gnd = self.scene_pts[self.pts_gnd_idx]

		# Find gravity and height
		self.grav_vec = self.grav_SVD(self.pts_gnd)
		# print('Gravity vector \n', self.grav_vec)
		self.h_0 = self.height_avg(self.pts_gnd, self.origin_w)
		# print('\nHeight h_0 = ', self.h_0)

		# Get focal length 
		cam_id = list(self.cameras_c.keys())[0]
		self.focal = self.cameras_c[cam_id].params[0]
		# print("Focal length \n", self.focal)


		# Define coordinate frame 
		z_bar = self.grav_vec
		P1, P2 = self.scene_pts[pts_gnd_idx[0],:], self.scene_pts[pts_gnd_idx[5],:]
		v = P2-P1

		# X Direction as ZcrossV
		x_dir = np.cross(z_bar, v)
		x_bar = x_dir/np.linalg.norm(x_dir)
		# print("\nX unit vector \n", x_bar)
		# Y Direction as ZcrossX
		y_dir = np.cross(z_bar, x_bar)
		y_bar = y_dir/np.linalg.norm(y_dir)
		# print("\nY unit vector \n", y_bar)

		# Rotation matrix 
		rotmat = np.column_stack((x_bar, y_bar, z_bar))
		# print("\nRotation Matrix\n", rotmat)
		# Translation Vector
		trans = P1.reshape([3,1])

		# Form transformation matrix 
		bottom = np.array([0.0, 0.0, 0.0, 1.0]).reshape([1,4])
		tform = np.concatenate([np.concatenate([rotmat, trans],1),bottom],0)
		# print("\nTransformation matrix to ground \n", tform)

		# Translation from ground to desired height 
		x = 0
		y = -6
		z = -1
		yaw = np.deg2rad(0)
		# Translation 2
		trans2 = np.array([x, y, z]).reshape([3,1])
		# Rotation 2
		euler_angles = [0., 0., yaw]
		rotmat2 = R.from_euler('xyz', euler_angles).as_matrix()
		tform2 = np.concatenate([np.concatenate([rotmat2, trans2],1),bottom],0)
		# print("\nTransformation from ground to desired coord frame (added a 220 deg yaw)\n", tform2)

		# Combine 
		tform_ref_frame = tform @ tform2
		self.tform_ref_frame = tform_ref_frame

		return tform_ref_frame


	def inv_homog_transform(self, homog):
		""" 
		Inverting a homogeneous transformation matrix
		Inputs: homogeneous transformation matrix (4x4)
		Outputs: inverted 4x4 matrix
		"""
		# Grab rotation matrix
		R = homog[:3,:3]

		# Transpose rotation matrix 
		R_inv = R.T

		# Grab translation matrix 
		t = homog[:-1, -1]
		t = t.reshape((3, 1))
		t_inv = -R_inv @ t

		# Form new transformation matrix 
		bottom = np.array([0.0, 0.0, 0.0, 1.0]).reshape([1, 4])
		homog_inv = np.concatenate([np.concatenate([R_inv, t_inv], 1), bottom], 0)    
		# print("\n Homogeneous new = \n", homog_inv)

		return homog_inv

	def unit_vec_tform(self, pts_vec, origin, homog_t):
		"""
		Takes a set of unit vectors and transforms them according to a homogeneous transform
		Input: Unit vectors, transform 
		Output: Origin of new unit vectors, end points of new unit vectors, new unit vectors
		"""
		# Get new origin
		origin_o = np.append(origin,1).reshape(-1,1)
		origin_n = (homog_t @ origin_o)[:-1].flatten()

		# Unit vectors to homogeneous coords 
		pts_homog = np.hstack((pts_vec, np.ones((pts_vec.shape[0], 1)))).T

		# Apply transformation
		pts_trans = (homog_t @ pts_homog)[:-1].T

		# New vectors 
		pts_vec_n = pts_trans - origin_n

		return origin_n, pts_trans, pts_vec_n


	def pose_scene_visualization_ref(self, vis, scene_pts_ref):
		"""
		Creating a visualization of pose estimations and sparse point cloud
		Input: vis (open3d)
		Output: vis (with pose estimations and point cloud)
		"""
		# Add origin axes
		axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1)
		vis.add_geometry(axes)

		# Add sparse point cloud
		scene_cloud = o3d.geometry.PointCloud()
		scene_cloud.points = o3d.utility.Vector3dVector(scene_pts_ref)
		scene_cloud.colors = o3d.utility.Vector3dVector(self.scene_rgb)
		vis.add_geometry(scene_cloud)

		# Size options (jupyter gives issues when running this multiple times, but it looks better)
		render_option = vis.get_render_option()
		render_option.point_size = 2

		# Run and destroy visualization 
		vis.run()
		vis.destroy_window()


	def plot_rect_im(self, x, y, width, height, imnum):
		"""
		Plotting a rectangle over the desired area of choice for ground plane point 
		Input: x and y starting point, width and height crop size, image number
		Output: plot with cropped section in red
		Specified by user with x, y, width, height 
		"""
		# Make figure 
		fig, ax = plt.subplots(figsize=(15,8))
		# Draw rectangle 
		rect = plt.Rectangle((x,y), width, height, linewidth=1, edgecolor='r', facecolor='none')
		# Grab correct image based on number indicator 
		im_gnd_plt = self.images_dict[imnum]
		im_gnd_plt = cv2.cvtColor(im_gnd_plt, cv2.COLOR_BGR2RGB)

		# print(im_gnd_plt)

		# Plot 
		ax.imshow(im_gnd_plt)
		ax.add_patch(rect)
		ax.axis("off")

		# Show plot 
		plt.show()

	def grab_image_pts(self, x, y, width, height, imnum):
		"""
		Grab points of an image (that we know are on ground plane)
		Based on specified starting x and y location, width, height
		Looking to automate process in future - currently manually chosen pts
		"""

		# Create grid of coords
		x_coords = np.arange(x, x+width)
		y_coords = np.arange(y, y+height)
		Px, Py = np.meshgrid(x_coords, y_coords, indexing='xy')
		# print(Px, Py)

		# Stack points for array 
		pts_loc = np.stack((Px, Py), axis=-1) # -1 forms a new axis 
		# print(pts_loc)

		# Extract RGB
		im_gnd = self.images_dict[imnum]
		pts_rgb = im_gnd[y:y+height, x:x+width].astype(int)

		# Store in dict
		corners = np.array([x,y,width,height])
		self.im_pts_2d[imnum] = {'pts': pts_loc}
		self.im_pts_2d[imnum]['rgbc'] = pts_rgb
		self.im_pts_2d[imnum]['corners'] = corners

		# return pts_loc, pts_rgb

	def grab_image_pts_tot(self, mosaic_params):
		"""
		Grab points of an image (that we know are on ground plane)
		Based on specified starting x and y location, width, height
		Looking to automate process in future - currently manually chosen pts
		"""
		self.mosaic_params = mosaic_params
		# Loop through mosaic_params for each image
		for imnum in range(len(self.images_dict)):
			x, y, width, height = mosaic_params[imnum]

			# Create grid of coords
			x_coords = np.arange(x, x+width)
			y_coords = np.arange(y, y+height)
			Px, Py = np.meshgrid(x_coords, y_coords, indexing='xy')
			# print(Px, Py)

			# Stack points for array 
			pts_loc = np.stack((Px, Py), axis=-1) # -1 forms a new axis 
			# print(pts_loc)

			# Extract RGB
			im_gnd = self.images_dict[imnum]
			pts_rgb = im_gnd[y:y+height, x:x+width].astype(int)

			# Store in dict
			corners = np.array([x,y,width,height])
			self.im_pts_2d[imnum] = {'pts': pts_loc}
			self.im_pts_2d[imnum]['rgbc'] = pts_rgb
			self.im_pts_2d[imnum]['corners'] = corners

		# return pts_loc, pts_rgb

	def plot_gnd_pts(self):
		"""
		Plotting boxes on each local image to represent ground sections to be used in mosaic process
		Input: figure, axes
		Output: subplot with proper ground section identification 
		"""
		rows = int(len(self.images_dict)/5)
		# Loop through each image
		for imnum in range(len(self.images_dict)):
			plt.subplot(rows,5,imnum+1)
			# Grab parameters 
			x, y, width, height = self.mosaic_params[imnum]
			# Draw rectangle 
			rect = plt.Rectangle((x,y), width, height, linewidth=1, edgecolor='r', facecolor='none')
			# Grab correct image based on number indicator 
			im_gnd_plt = self.images_dict[imnum]
			im_gnd_plt = cv2.cvtColor(im_gnd_plt, cv2.COLOR_BGR2RGB)
			# print(im_gnd_plt)

			# Plot 
			plt.imshow(im_gnd_plt)
			plt.gca().add_patch(rect)
			plt.axis("off")

		# Show plot 
		plt.show()

	def unit_vec_c(self, imnum):
		"""
		Create unit vectors in camera frame coordinates for desired pixels 
		Using pixel location of points.
		"""
		# Get pixel locations and RGB values
		pts_loc = self.im_pts_2d[imnum]['pts']  # Shape (H, W, 2)
		# print(pts_loc)
		pts_rgb = self.im_pts_2d[imnum]['rgbc']  # Shape (H, W, 3)
		im_imnum = self.images_dict[imnum]

		shape_im_y, shape_im_x = im_imnum.shape[:2]

		# Compute shifted pixel coordinates
		Px = pts_loc[..., 0] - shape_im_x / 2  # Shape (H, W)
		Py = -pts_loc[..., 1] + shape_im_y / 2  # Shape (H, W)

		# Apply final coordinate transformations
		Px, Py = -Py, -Px  # Swap and negate as per coordinate system

		# Compute magnitude of vectors
		mag = np.sqrt(Px**2 + Py**2 + self.focal**2)  # Shape (H, W)
		self.im_pts_2d[imnum]['mag'] = mag

		# Compute unit vectors
		pts_vec_c = np.stack((Px / mag, Py / mag, np.full_like(Px, self.focal) / mag), axis=-1)  # Shape (H, W, 3)

		# Reshape into (N, 3) where N = H * W
		pts_vec_c = pts_vec_c.reshape(-1, 3)
		# pts_vec_c = pts_vec_c.reshape(-1, 3, order='F') # This would flatten by COLUMN first (top to bottom, then L to R)
		pts_rgb_gnd = pts_rgb.reshape(-1, 3) / 255  # Normalize and reshape

		return pts_vec_c, pts_rgb_gnd


	def get_pose_id(self, id,imnum):
		"""
		Get the pose transformation for a specific image id
		Input: Image ID
		Output: transform from camera to world coordinates
		"""
		# Get quaternion and translation vector
		qvec = self.images_c[id].qvec
		tvec = self.images_c[id].tvec[:,None]
		# print(tvec)

		t = tvec.reshape([3,1])

		# Create rotation matrix
		Rotmat = qvec2rotmat(qvec) # Positive or negative does not matter
		# print("\n Rotation matrix \n", Rotmat)

		# Create 4x4 transformation matrix with rotation and translation
		bottom = np.array([0.0, 0.0, 0.0, 1.0]).reshape([1, 4])
		w2c = np.concatenate([np.concatenate([Rotmat, t], 1), bottom], 0)
		c2w = np.linalg.inv(w2c)

		self.im_pts_2d[imnum]['w2c'] = w2c
		self.im_pts_2d[imnum]['c2w'] = c2w

		return w2c, c2w


	def pt_range(self, pts_vec, homog_t, origin, imnum):
		"""
		Finding the range of the point which intersects the ground plane 
		Input: Unit vectors, homogeneous transform 
		Output: Range for numbers, new 3D points 
		"""

		# Get translation vector 
		t_cw = homog_t[:-1,-1]
		a = np.dot(t_cw, self.grav_vec)

		# Numerator
		num = self.h_0 - a

		# Denominator
		denom = np.dot(pts_vec, self.grav_vec)

		# Compute range
		r = num/denom
		self.im_pts_2d[imnum]['r'] = r
		self.im_pts_2d[imnum]['origin'] = origin

		# New points
		new_pts = origin + pts_vec*r[:, np.newaxis]

		return r.reshape(-1,1), new_pts

	def conv_to_gray(self, pts_rgb, imnum):
		"""
		Takes RGB values and converts to grayscale
		Uses standard luminance-preserving transformation
		Inputs: RGB values (nx3), image number 
		Outputs: grayscale values (nx3) for open3d
		"""

		# Calculate intensity value
		intensity = 0.299 * pts_rgb[:, 0] + 0.587 * pts_rgb[:, 1] + 0.114 * pts_rgb[:, 2]
		# Create nx3
		gray_colors = np.tile(intensity[:, np.newaxis], (1, 3))  # Repeat intensity across R, G, B channels
		# print(gray_colors)

		return gray_colors


	def mosaic_visualization(self, vis):
		""" 
		Plotting the new scene mosaic 
		Input: vis (from open3d)
		Output: vis with mosaic (from open3d)
		"""

		# Create axes @ origin
		axis_origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1)
		vis.add_geometry(axis_origin)

		for i in range(len(self.images_dict)):
			cloud = o3d.geometry.PointCloud()
			cloud.points = o3d.utility.Vector3dVector(self.im_mosaic[i]['pts'])
			cloud.colors = o3d.utility.Vector3dVector(self.im_mosaic[i]['color_g'])
			vis.add_geometry(cloud)

		# # Create point cloud for reference cloud (satellite)
		# ref_cloud = o3d.geometry.PointCloud()
		# ref_cloud.points = o3d.utility.Vector3dVector(gnav.ref_pts)
		# ref_cloud.colors = o3d.utility.Vector3dVector(gnav.ref_rgb)

		# # Size options (jupyter gives issues when running this multiple times, but it looks better)
		# render_option = vis.get_render_option()
		# render_option.point_size = 2

		# # Set up initial viewpoint
		# view_control = vis.get_view_control()
		# # Direction which the camera is looking
		# view_control.set_front([0, 0, -1])  # Set the camera facing direction
		# # Point which the camera revolves about 
		# view_control.set_lookat([0, 0, 0])   # Set the focus point
		# # Defines which way is up in the camera perspective 
		# view_control.set_up([0, -1, 0])       # Set the up direction
		# view_control.set_zoom(.45)           # Adjust zoom if necessary

		# Run and destroy visualization 
		vis.run()
		vis.destroy_window()