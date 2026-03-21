import os
from typing import Tuple

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset

from multi_contrast_inr.dataset import _BaseDataset
from multi_contrast_inr.dataset_utils import norm_grid

def get_image_coordinate_grid_nib(image: nib.Nifti1Image):
	img_header = image.header
	img_data = image.get_fdata()
	img_affine = image.affine
	(x, y, z) = image.shape

	X = np.linspace(0, x - 1, x)
	Y = np.linspace(0, y - 1, y)
	Z = np.linspace(0, z - 1, z)
	points = np.meshgrid(X, Y, Z, indexing='ij')
	points = np.stack(points).transpose(1, 2, 3, 0).reshape(-1, 3)
	coordinates = list(nib.affines.apply_affine(img_affine, points))

	label = list(img_data.flatten())
	coordinates_arr = np.array(coordinates, dtype=np.float32)
	label_arr = np.array(label, dtype=np.float32)


	image_dict = {
		'affine': torch.tensor(img_affine),
		'origin': torch.tensor(np.array([0])),
		'spacing': torch.tensor(np.array(img_header["pixdim"][1:4])),
		'dim': torch.tensor(np.array([x, y, z])),
		'intensity': torch.tensor(label_arr, dtype=torch.float32).view(-1, 1),
		'coordinates': torch.tensor(coordinates_arr, dtype=torch.float32),
	}
	return image_dict

def get_image_coordinate_grid_nib_slicesampling(image: nib.Nifti1Image, slice_axis=2): # 0, 1, 2 = x, y, z
	img_header = image.header
	img_data = image.get_fdata()
	img_affine = image.affine
	(x, y, z) = image.shape

	X = np.linspace(0, x - 1, x)
	Y = np.linspace(0, y - 1, y)
	Z = np.linspace(0, z - 1, z)

	points = np.meshgrid(X, Y, Z, indexing='ij')
	points = np.stack(points).transpose(1, 2, 3, 0)

	# point stack slice-wise
	points_shape = np.array(points.shape)
	batchsize = np.prod(np.delete(points_shape, slice_axis)[:2])

	points_slicewise = np.empty((0, 3)).astype('float')
	label_slicewise = np.empty((0, 1)).astype('float')
	for i in range(points_shape[slice_axis]):
		points_slicewise = np.concatenate((points_slicewise,np.take(points, i, slice_axis).reshape(-1, 3)))
		label_slicewise = np.concatenate((label_slicewise, np.take(img_data, i, slice_axis).reshape(-1, 1)))

	points_sliceidx = points_slicewise[:, slice_axis, None]
	coordinates_slicewise = list(nib.affines.apply_affine(img_affine, points_slicewise))
	coordinates_arr_slicewise = np.array(coordinates_slicewise, dtype=np.float32)

	image_dict = {
		'affine': torch.tensor(img_affine),
		'origin': torch.tensor(np.array([0])),
		'spacing': torch.tensor(np.array(img_header["pixdim"][1:4])),
		'dim': torch.tensor(np.array([x, y, z])),
		'intensity': torch.tensor(label_slicewise, dtype=torch.float32).view(-1, 1),
		'coordinates': torch.tensor(coordinates_arr_slicewise, dtype=torch.float32),
		'coordinates_slice_index': torch.tensor(points_sliceidx, dtype=torch.int),
		'batchsize': batchsize
	}

	return image_dict

class MultiViewDataset(_BaseDataset):
	r"""Paired multi-view / multi-contrast dataset.

	Both images must already be registered to one another.
	Modified from https://github.com/jqmcginnis/multi_contrast_inr/blob/main/dataset.py
	"""

	def __init__(self, image_dir: str = "", name="BrainLesionDataset",
				 subject_id: str = "123456",
				 contrast1_LR_str: str = 'axial_LR',
				 contrast2_LR_str: str = 'coronal_LR',

				 transform=None, target_transform=None):
		super(MultiViewDataset, self).__init__(image_dir)
		self.dataset_name = name
		self.subject_id = subject_id
		self.contrast1_LR_str = contrast1_LR_str
		self.contrast2_LR_str = contrast2_LR_str
		if 'prostate' in name:
			self.contrast1_GT_str = contrast1_LR_str
			self.contrast2_GT_str = contrast2_LR_str
		else:
			self.contrast1_GT_str = ""
			self.contrast2_GT_str = ""

		self.dataset_name = (
			f'preprocessed_data/{self.dataset_name}_'
			f'{self.subject_id}_'
			f'{self.contrast1_LR_str}_{self.contrast1_GT_str}_'
			f'{self.contrast2_LR_str}_{self.contrast2_GT_str}_'
			f'.pt'
		)

		print(self.dataset_name)
		self.lr_contrast1 = os.path.join(self.image_dir, self.subject_id) + f'_{self.contrast1_LR_str}.nii.gz'
		self.lr_contrast2 = os.path.join(self.image_dir, self.subject_id) + f'_{self.contrast2_LR_str}.nii.gz'
		gt1_sep = '_' if self.contrast1_GT_str else ''
		gt2_sep = '_' if self.contrast2_GT_str else ''
		self.gt_contrast1 = os.path.join(self.image_dir, self.subject_id) + f'{gt1_sep}{self.contrast1_GT_str}.nii.gz'
		self.gt_contrast2 = os.path.join(self.image_dir, self.subject_id) + f'{gt2_sep}{self.contrast2_GT_str}.nii.gz'

		if os.path.isfile(self.dataset_name):
			print("Dataset available.")
			dataset = torch.load(self.dataset_name, weights_only=False)
			self.data1 = dataset["data1"]
			self.label1 = dataset["label1"]
			self.data2 = dataset["data2"]
			self.label2 = dataset["label2"]
			self.affine1 = dataset["affine1"]
			self.dim1 = dataset["dim1"]
			self.coordinates1 = dataset["coordinates1"]
			self.affine2 = dataset["affine2"]
			self.dim2 = dataset["dim2"]
			self.coordinates2 = dataset["coordinates2"]
			self.len = dataset["len"]
			self.gt_contrast1 = dataset["gt_contrast1"]
			self.gt_contrast2 = dataset["gt_contrast2"]
			self.coordinates_minmax = dataset["coordinates_minmax"]
			print("Skipped preprocessing.")


		else:
			self.len = 0
			self.data = []
			self.label = []
			self._process()

		self.transform = transform
		self.target_transform = target_transform

	def __len__(self):
		return self.len

	def __getitem__(self, idx) -> Tuple[dict, dict]:
		data1 = self.data1[idx]
		label1 = self.label1[idx]
		data2 = self.data2[idx]
		label2 = self.label2[idx]
		return (data1, label1), (data2, label2)

	def get_intensities(self):
		return self.label

	def get_mask(self):
		return self.mask

	def get_coordinates1(self):
		return self.coordinates1

	def get_affine1(self):
		return self.affine1

	def get_dim1(self):
		return self.dim1


	def get_coordinates2(self):
		return self.coordinates2

	def get_affine2(self):
		return self.affine2

	def get_dim2(self):
		return self.dim2


	def get_contrast1_gt(self):
		return self.gt_contrast1

	def get_contrast2_gt(self):
		return self.gt_contrast2

	def get_contrast2_gt_mask(self):
		return self.gt_contrast2_mask

	def get_contrast1_gt_mask(self):
		return self.gt_contrast1_mask

	def _process(self):
		print(f"Using {self.lr_contrast1} as contrast1.")
		print(f"Using {self.lr_contrast2} as contrast2.")
		print(f"Using {self.gt_contrast1} as gt contrast1.")
		print(f"Using {self.gt_contrast2} as gt contrast2.")

		contrast1_dict = get_image_coordinate_grid_nib(nib.load(str(self.lr_contrast1)))
		contrast2_dict = get_image_coordinate_grid_nib(nib.load(str(self.lr_contrast2)))
		data_contrast1 = contrast1_dict["coordinates"]
		data_contrast2 = contrast2_dict["coordinates"]

		min1, max1 = data_contrast1.min(), data_contrast1.max()
		min2, max2 = data_contrast2.min(), data_contrast2.max()

		min_c, max_c = np.min(np.array([min1, min2])), np.max(np.array([max1, max2]))
		self.coordinates_minmax = [min_c, max_c]

		print(f'Min/Max of Contrast 1 {min1, max1}')
		print(f'Min/Max of Contrast 2 {min2, max2}')
		print(f'Global Min/Max of Contrasts {min_c, max_c}')

		data_contrast1 = norm_grid(data_contrast1, xmin=min_c, xmax=max_c)
		data_contrast2 = norm_grid(data_contrast2, xmin=min_c, xmax=max_c)

		labels_contrast1 = contrast1_dict["intensity"]
		labels_contrast2 = contrast2_dict["intensity"]
		Imin1, Imax1 = labels_contrast1.min(), labels_contrast1.max()
		Imin2, Imax2 = labels_contrast2.min(), labels_contrast2.max()

		min_I, max_I = np.min(np.array([Imin1, Imin2])), np.max(np.array([Imax1, Imax2]))

		def min_max_scale(X, x_min, x_max, s_min=-1, s_max=1):
			return (X - x_min) / (x_max - x_min) * (s_max - s_min) + s_min

		labels_contrast1 = min_max_scale(labels_contrast1, min_I, max_I)
		labels_contrast2 = min_max_scale(labels_contrast2, min_I, max_I)

		data_length = np.max([len(data_contrast1),len(data_contrast2)])
		data_template = torch.ones((data_length, 3)) * -1
		label_template = torch.ones((data_length, 1)) * -1

		self.data1 = data_template.clone()
		self.label1 = label_template.clone()
		self.data2 = data_template.clone()
		self.label2 = label_template.clone()

		self.coordinates1 = data_contrast1
		self.coordinates2 = data_contrast2
		self.data1[:len(data_contrast1)] = data_contrast1
		self.data2[:len(data_contrast2)] = data_contrast2
		self.label1[:len(labels_contrast1)] = labels_contrast1.clone()
		self.label2[:len(labels_contrast2)] = labels_contrast2.clone()

		self.mask = None
		self.len = len(self.label2)

		# Keep the GT image for downstream metrics.
		gt_contrast1_dict = get_image_coordinate_grid_nib(nib.load(str(self.gt_contrast1)))
		self.gt_contrast1 = min_max_scale(gt_contrast1_dict["intensity"], min_I, max_I)
		self.gt_coordinates = norm_grid(gt_contrast1_dict['coordinates'], xmin=min_c, xmax=max_c) # isotropic ..
		self.gt_affine = gt_contrast1_dict['affine']
		self.gt_dim = gt_contrast1_dict['dim']

		self.affine1 = contrast1_dict["affine"]
		self.dim1 = contrast1_dict["dim"]

		self.affine2 = contrast2_dict["affine"]
		self.dim2 = contrast2_dict["dim"]

		# Cache the processed dataset to avoid repeating preprocessing.
		dataset = {
			'len': self.len,
			'data1': self.data1,
			'label1': self.label1,
			'data2': self.data2,
			'label2': self.label2,
			'affine1': self.affine1,
			'dim1': self.dim1,
			'coordinates1': self.coordinates1,
			'coordinates_minmax': self.coordinates_minmax,
			'affine2': self.affine2,
			'dim2': self.dim2,
			'coordinates2': self.coordinates2,
			'gt_contrast1': self.gt_contrast1,
			'gt_contrast2': self.gt_contrast2,
			'gt_coordinates': self.gt_coordinates,
			'gt_affine': self.affine1,
			'gt_dim': self.dim1,
		}
		if not os.path.exists(os.path.join(os.getcwd(), os.path.split(self.dataset_name)[0])):
			os.makedirs(os.path.join(os.getcwd(), os.path.split(self.dataset_name)[0]))

		torch.save(dataset, self.dataset_name)


class InferDataset(Dataset):
	def __init__(self, grid):
		super(InferDataset, self, ).__init__()
		self.grid = grid

	def __len__(self):
		return len(self.grid)

	def __getitem__(self, idx):
		data = self.grid[idx]
		return data

class MultiViewDataset_slicesampling(_BaseDataset):
	r"""Slice-sampling variant of the paired multi-view dataset.

	Both images must already be registered to one another.
	Modified from https://github.com/jqmcginnis/multi_contrast_inr/blob/main/dataset.py
	"""

	def __init__(self, image_dir: str = "", name="BrainLesionDataset",
				 subject_id: str = "123456",
				 contrast1_LR_str: str = 'axial_LR',
				 contrast2_LR_str: str = 'coronal_LR',
				 transform=None, target_transform=None, view_axis=2):
		super(MultiViewDataset_slicesampling, self).__init__(image_dir)
		self.dataset_name = name
		self.subject_id = subject_id
		self.contrast1_LR_str = contrast1_LR_str
		self.contrast2_LR_str = contrast2_LR_str
		self.view_axis = view_axis
		if 'prostate' in name:
			self.contrast1_GT_str = contrast1_LR_str
			self.contrast2_GT_str = contrast2_LR_str
		else:
			self.contrast1_GT_str = ""
			self.contrast2_GT_str = ""

		self.dataset_name = (
			f'preprocessed_data/{self.dataset_name}_'
			f'{self.subject_id}_'
			f'{self.contrast1_LR_str}_{self.contrast1_GT_str}_'
			f'{self.contrast2_LR_str}_{self.contrast2_GT_str}_'
			f'.pt'
		)

		print(self.dataset_name)
		self.lr_contrast1 = os.path.join(self.image_dir, self.subject_id) + f'_{self.contrast1_LR_str}.nii.gz'
		self.lr_contrast2 = os.path.join(self.image_dir, self.subject_id) + f'_{self.contrast2_LR_str}.nii.gz'
		gt1_sep = '_' if self.contrast1_GT_str else ''
		gt2_sep = '_' if self.contrast2_GT_str else ''
		self.gt_contrast1 = os.path.join(self.image_dir, self.subject_id) + f'{gt1_sep}{self.contrast1_GT_str}.nii.gz'
		self.gt_contrast2 = os.path.join(self.image_dir, self.subject_id) + f'{gt2_sep}{self.contrast2_GT_str}.nii.gz'

		if os.path.isfile(self.dataset_name):
			print("Dataset available.")
			dataset = torch.load(self.dataset_name, weights_only=False)
			self.slice_index1 = dataset["slice_index1"]
			self.slice_index2 = dataset["slice_index2"]
			self.data1 = dataset["data1"]
			self.label1 = dataset["label1"]
			self.data2 = dataset["data2"]
			self.label2 = dataset["label2"]
			self.affine1 = dataset["affine1"]
			self.dim1 = dataset["dim1"]
			self.coordinates1 = dataset["coordinates1"]
			self.affine2 = dataset["affine2"]
			self.dim2 = dataset["dim2"]
			self.coordinates2 = dataset["coordinates2"]
			self.len = dataset["len"]
			self.gt_contrast1 = dataset["gt_contrast1"]
			self.gt_contrast2 = dataset["gt_contrast2"]
			self.coordinates_minmax = dataset["coordinates_minmax"]

			print("Skipped preprocessing.")


		else:
			self.len = 0
			self.data = []
			self.label = []
			self._process()

		self.transform = transform
		self.target_transform = target_transform

	def __len__(self):
		return self.len

	def __getitem__(self, idx) -> Tuple[dict, dict]:
		if self.view_axis == 1: # coronal
			data2 = self.data2[idx]
			label2 = self.label2[idx]
			slice_index2 = self.slice_index2[idx]

			return data2, label2, slice_index2

		elif self.view_axis == 2: # axial
			data1 = self.data1[idx]
			label1 = self.label1[idx]
			slice_index1 = self.slice_index1[idx]

			return data1, label1, slice_index1

		else:
			raise NotImplementedError

	def get_intensities(self):
		return self.label

	def get_mask(self):
		return self.mask

	def get_coordinates1(self):
		return self.coordinates1

	def get_affine1(self):
		return self.affine1

	def get_dim1(self):
		return self.dim1


	def get_coordinates2(self):
		return self.coordinates2

	def get_affine2(self):
		return self.affine2

	def get_dim2(self):
		return self.dim2


	def get_contrast1_gt(self):
		return self.gt_contrast1

	def get_contrast2_gt(self):
		return self.gt_contrast2

	def get_contrast2_gt_mask(self):
		return self.gt_contrast2_mask

	def get_contrast1_gt_mask(self):
		return self.gt_contrast1_mask

	def _process(self):
		print(f"Using {self.lr_contrast1} as contrast1.")
		print(f"Using {self.lr_contrast2} as contrast2.")
		print(f"Using {self.gt_contrast1} as gt contrast1.")
		print(f"Using {self.gt_contrast2} as gt contrast2.")
		contrast1_dict = get_image_coordinate_grid_nib_slicesampling(nib.load(str(self.lr_contrast1)), slice_axis=2) # axial
		contrast2_dict = get_image_coordinate_grid_nib_slicesampling(nib.load(str(self.lr_contrast2)), slice_axis=1) # coronal

		data_contrast1 = contrast1_dict["coordinates"]
		data_contrast2 = contrast2_dict["coordinates"]

		min1, max1 = data_contrast1.min(), data_contrast1.max()
		min2, max2 = data_contrast2.min(), data_contrast2.max()

		min_c, max_c = np.min(np.array([min1, min2])), np.max(np.array([max1, max2]))
		self.coordinates_minmax = [min_c, max_c]

		print(f'Min/Max of Contrast 1 {min1, max1}')
		print(f'Min/Max of Contrast 2 {min2, max2}')
		print(f'Global Min/Max of Contrasts {min_c, max_c}')

		data_contrast1 = norm_grid(data_contrast1, xmin=min_c, xmax=max_c)
		data_contrast2 = norm_grid(data_contrast2, xmin=min_c, xmax=max_c)

		labels_contrast1 = contrast1_dict["intensity"]
		labels_contrast2 = contrast2_dict["intensity"]
		Imin1, Imax1 = labels_contrast1.min(), labels_contrast1.max()
		Imin2, Imax2 = labels_contrast2.min(), labels_contrast2.max()

		min_I, max_I = np.min(np.array([Imin1, Imin2])), np.max(np.array([Imax1, Imax2]))

		def min_max_scale(X, x_min, x_max, s_min=-1, s_max=1):
			return (X - x_min) / (x_max - x_min) * (s_max - s_min) + s_min

		labels_contrast1 = min_max_scale(labels_contrast1, min_I, max_I)
		labels_contrast2 = min_max_scale(labels_contrast2, min_I, max_I)

		self.data1 = data_contrast1.clone()
		self.label1 = labels_contrast1.clone()
		self.data2 = data_contrast2.clone()
		self.label2 = labels_contrast2.clone()
		self.coordinates1 = data_contrast1
		self.coordinates2 = data_contrast2


		self.mask = None
		self.len = len(self.label2)

		# Keep the GT image for downstream metrics.
		gt_contrast1_dict = get_image_coordinate_grid_nib(nib.load(str(self.gt_contrast1)))
		self.gt_contrast1 = min_max_scale(gt_contrast1_dict["intensity"], min_I, max_I)
		self.gt_coordinates = norm_grid(gt_contrast1_dict['coordinates'], xmin=min_c, xmax=max_c) # isotropic ..
		self.gt_affine = gt_contrast1_dict['affine']
		self.gt_dim = gt_contrast1_dict['dim']

		self.affine1 = contrast1_dict["affine"]
		self.dim1 = contrast1_dict["dim"]

		self.affine2 = contrast2_dict["affine"]
		self.dim2 = contrast2_dict["dim"]

		self.slice_index1 = contrast1_dict['coordinates_slice_index']
		self.slice_index2 = contrast2_dict['coordinates_slice_index']

		# Cache the processed dataset to avoid repeating preprocessing.
		dataset = {
			'len': self.len,
			'data1': self.data1,
			'label1': self.label1,
			'data2': self.data2,
			'label2': self.label2,

			'slice_index1': self.slice_index1,
			'slice_index2': self.slice_index2,

			'affine1': self.affine1,
			'dim1': self.dim1,
			'coordinates1': self.coordinates1,
			'coordinates_minmax': self.coordinates_minmax,
			'affine2': self.affine2,
			'dim2': self.dim2,
			'coordinates2': self.coordinates2,
			'gt_contrast1': self.gt_contrast1,
			'gt_contrast2': self.gt_contrast2,
			'gt_coordinates': self.gt_coordinates,
			'gt_affine': self.affine1,
			'gt_dim': self.dim1,
		}

		if not os.path.exists(os.path.join(os.getcwd(), os.path.split(self.dataset_name)[0])):
			os.makedirs(os.path.join(os.getcwd(), os.path.split(self.dataset_name)[0]))

		torch.save(dataset, self.dataset_name)


class FinetuningTwoSlicePoints(Dataset):
	def __init__(self, image_dir: str = "", name="BrainLesionDataset",
				 subject_id: str = "123456",
				 contrast1_LR_str: str = 'axial_LR',
				 contrast2_LR_str: str = 'coronal_LR',
				 view_axis=2):
		super(FinetuningTwoSlicePoints, self).__init__()

		self.image_dir = image_dir
		self.dataset_name = name
		self.subject_id = subject_id
		self.contrast1_LR_str = contrast1_LR_str
		self.contrast2_LR_str = contrast2_LR_str
		self.view_axis = view_axis
		if 'prostate' in name:
			self.contrast1_GT_str = contrast1_LR_str
			self.contrast2_GT_str = contrast2_LR_str
		else:
			self.contrast1_GT_str = ""
			self.contrast2_GT_str = ""

		self.dataset_name = (
			f'preprocessed_data/{self.dataset_name}_'
			f'{self.subject_id}_'
			f'{self.contrast1_LR_str}_{self.contrast1_GT_str}_'
			f'{self.contrast2_LR_str}_{self.contrast2_GT_str}_'
			f'.pt'
		)

		print(self.dataset_name)
		self.lr_contrast1 = os.path.join(self.image_dir, self.subject_id) + f'_{self.contrast1_LR_str}.nii.gz'
		self.lr_contrast2 = os.path.join(self.image_dir, self.subject_id) + f'_{self.contrast2_LR_str}.nii.gz'
		gt1_sep = '_' if self.contrast1_GT_str else ''
		gt2_sep = '_' if self.contrast2_GT_str else ''
		self.gt_contrast1 = os.path.join(self.image_dir, self.subject_id) + f'{gt1_sep}{self.contrast1_GT_str}.nii.gz'
		self.gt_contrast2 = os.path.join(self.image_dir, self.subject_id) + f'{gt2_sep}{self.contrast2_GT_str}.nii.gz'

		if os.path.isfile(self.dataset_name):
			print("Dataset available.")
			dataset = torch.load(self.dataset_name, weights_only=False)
			self.slice_index1 = dataset["slice_index1"]
			self.slice_index2 = dataset["slice_index2"]
			self.data1 = dataset["data1"]
			self.label1 = dataset["label1"]
			self.data2 = dataset["data2"]
			self.label2 = dataset["label2"]
			self.affine1 = dataset["affine1"]
			self.dim1 = dataset["dim1"]
			self.coordinates1 = dataset["coordinates1"]
			self.affine2 = dataset["affine2"]
			self.dim2 = dataset["dim2"]
			self.coordinates2 = dataset["coordinates2"]
			self.len = dataset["len"]
			self.gt_contrast1 = dataset["gt_contrast1"]
			self.gt_contrast2 = dataset["gt_contrast2"]
			self.coordinates_minmax = dataset["coordinates_minmax"]

			print("Skipped preprocessing.")


	def __len__(self):
		return self.slice_index1.max().item()

	def __getitem__(self, index):
		next_index = index+1

		if self.view_axis == 1: # coronal

			point_idx = np.where(self.slice_index2 == index)[0]
			point_idx_next = np.where(self.slice_index2 == next_index)[0]

			data2 = self.data2[point_idx]
			label2 = self.label2[point_idx]
			slice_index2 = self.slice_index2[point_idx]

			data2next = self.data2[point_idx_next]
			label2next = self.label2[point_idx_next]
			slice_index2next = self.slice_index2[point_idx_next]

			return torch.cat((data2, data2next), 0), \
				   torch.cat((label2, label2next), 0), \
				   torch.cat((slice_index2, slice_index2next), 0)

		elif self.view_axis == 2: # axial

			point_idx = np.where(self.slice_index1 == index)[0]
			point_idx_next = np.where(self.slice_index1 == next_index)[0]

			data1 = self.data1[point_idx]
			label1 = self.label1[point_idx]
			slice_index1 = self.slice_index1[point_idx]

			data1next = self.data1[point_idx_next]
			label1next = self.label1[point_idx_next]
			slice_index1next = self.slice_index1[point_idx_next]

			return torch.cat((data1, data1next), 0), \
				   torch.cat((label1, label1next), 0), \
				   torch.cat((slice_index1, slice_index1next), 0)
		else:
			raise NotImplementedError
