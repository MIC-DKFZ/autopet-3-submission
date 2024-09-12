from copy import deepcopy
from typing import Tuple, List, Union, Dict

import math
import numpy as np
import pandas as pd
import torch
from scipy.ndimage import fourier_gaussian
from torch import Tensor
from torch.nn.functional import grid_sample

from batchgeneratorsv2.helpers.scalar_type import RandomScalar, sample_scalar
from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform
from batchgeneratorsv2.transforms.utils.cropping import crop_tensor


class Misalign(BasicTransform):
    def __init__(self,
                 im_channels_2_misalign: Tuple[int,] = [0, ],
                 label_channels_2_misalign: Tuple[int,] = [0, ],

                 angle_x: RandomScalar = (0, 2 * np.pi),
                 angle_y: RandomScalar = (0, 2 * np.pi),
                 angle_z: RandomScalar = (0, 2 * np.pi),
                 p_rot: float = 0,

                 shiftZYX: Tuple[int, ...] = (2, 32, 32),
                 p_shift: float = None,

                 bg_style_seg_sampling: bool = True,
                 mode_seg: str = 'bilinear'
                 ):
        super().__init__()
        self.im_channels_2_misalign = im_channels_2_misalign
        self.label_channels_2_misalign = label_channels_2_misalign

        self.angle_x = angle_x
        self.angle_y = angle_y
        self.angle_z = angle_z
        self.p_rot = p_rot

        self.shiftZYX = shiftZYX
        self.p_shift = p_shift

        self.bg_style_seg_sampling = bg_style_seg_sampling
        self.mode_seg = mode_seg

    def get_parameters(self, **data_dict) -> dict:
        dim = data_dict['image'].ndim - 1
        # grid center must be in [-1, 1] as required by grid_sample
        shape = data_dict['image'].shape[1:]

        do_shift = np.random.uniform() < self.p_shift

        if do_shift:
            center_location_in_pixels = []
            for d in range(dim):
                center_location_in_pixels.append(shape[d] / 2 + np.random.randint(-self.shiftZYX[d], self.shiftZYX[d] + 1))
        else:
            center_location_in_pixels = [i / 2 for i in shape]

        affine = None
        offsets = None
        return {
            'affine': affine,
            'elastic_offsets': offsets,
            'center_location_in_pixels': center_location_in_pixels
        }

    def _apply_to_image(self, img: torch.Tensor, **params) -> torch.Tensor:
        shape = img.shape[1:]
        if params['affine'] is None and params['elastic_offsets'] is None:
            # No spatial transformation is being done. Round grid_center and crop without having to interpolate.
            # This saves compute.
            # cropping requires the center to be given as integer coordinates
            for ch in self.im_channels_2_misalign:
                img[ch, ...] = crop_tensor(img[ch, ...].unsqueeze(0), [math.floor(i) for i in params['center_location_in_pixels']], shape, pad_mode='constant',
                                           pad_kwargs={'value': 0})
            return img
        else:
            grid = _create_identity_grid(shape)

            # the grid must be scaled. The grid is [-1, 1] in image coordinates, but we want it to represent the smaller patch
            grid_scale = torch.Tensor([i / j for i, j in zip(img.shape[1:], shape)])
            grid /= grid_scale

            # we deform first, then rotate
            if params['elastic_offsets'] is not None:
                grid += params['elastic_offsets']
            if params['affine'] is not None:
                grid = torch.matmul(grid, torch.from_numpy(params['affine']).float())

            # we center the grid around the center_location_in_pixels. We should center the mean of the grid, not the center position
            mn = grid.mean(dim=list(range(img.ndim - 1)))
            new_center = torch.Tensor(
                [(j / (i / 2) - 1) for i, j in zip(img.shape[1:], params['center_location_in_pixels'])])
            grid += - mn + new_center
            return grid_sample(img[None], grid[None], mode='bilinear', padding_mode="zeros", align_corners=False)[0]

    def _apply_to_segmentation(self, segmentation: torch.Tensor, **params) -> torch.Tensor:
        segmentation = segmentation.contiguous()
        shape = segmentation.shape[1:]

        if params['affine'] is None and params['elastic_offsets'] is None:
            # No spatial transformation is being done. Round grid_center and crop without having to interpolate.
            # This saves compute.
            # cropping requires the center to be given as integer coordinates
            for ch in self.label_channels_2_misalign:
                segmentation[ch, ...] = crop_tensor(segmentation[ch, ...].unsqueeze(0), [math.floor(i) for i in params['center_location_in_pixels']], shape,
                                                    pad_mode='constant', pad_kwargs={'value': 0})
            return segmentation
        else:
            grid = _create_identity_grid(shape)

            # the grid must be scaled. The grid is [-1, 1] in image coordinates, but we want it to represent the smaller patch
            grid_scale = torch.Tensor([i / j for i, j in zip(segmentation.shape[1:], shape)])
            grid /= grid_scale

            # we deform first, then rotate
            if params['elastic_offsets'] is not None:
                grid += params['elastic_offsets']
            if params['affine'] is not None:
                grid = torch.matmul(grid, torch.from_numpy(params['affine']).float())

            # we center the grid around the center_location_in_pixels. We should center the mean of the grid, not the center coordinate
            mn = grid.mean(dim=list(range(segmentation.ndim - 1)))
            new_center = torch.Tensor(
                [(j / (i / 2) - 1) for i, j in zip(segmentation.shape[1:], params['center_location_in_pixels'])])
            grid += - mn + new_center

            if self.mode_seg == 'nearest':
                result_seg = grid_sample(
                    segmentation[None].float(),
                    grid[None],
                    mode=self.mode_seg,
                    padding_mode="zeros",
                    align_corners=False
                )[0].to(segmentation.dtype)
            else:
                result_seg = torch.zeros((segmentation.shape[0], *shape), dtype=segmentation.dtype)
                if self.bg_style_seg_sampling:
                    for c in range(segmentation.shape[0]):
                        labels = torch.from_numpy(np.sort(pd.unique(segmentation[c].numpy().ravel())))
                        # if we only have 2 labels then we can save compute time
                        if len(labels) == 2:
                            out = grid_sample(
                                ((segmentation[c] == labels[1]).float())[None, None],
                                grid[None],
                                mode=self.mode_seg,
                                padding_mode="zeros",
                                align_corners=False
                            )[0][0] >= 0.5
                            result_seg[c][out] = labels[1]
                            result_seg[c][~out] = labels[0]
                        else:
                            for i, u in enumerate(labels):
                                result_seg[c][
                                    grid_sample(
                                        ((segmentation[c] == u).float())[None, None],
                                        grid[None],
                                        mode=self.mode_seg,
                                        padding_mode="zeros",
                                        align_corners=False
                                    )[0][0] >= 0.5] = u
                else:
                    for c in range(segmentation.shape[0]):
                        labels = torch.from_numpy(np.sort(pd.unique(segmentation[c].numpy().ravel())))
                        # torch.where(torch.bincount(segmentation.ravel()) > 0)[0].to(segmentation.dtype)
                        tmp = torch.zeros((len(labels), *shape), dtype=torch.float16)
                        scale_factor = 1000
                        done_mask = torch.zeros(*shape, dtype=torch.bool)
                        for i, u in enumerate(labels):
                            tmp[i] = grid_sample(((segmentation[c] == u).float() * scale_factor)[None, None], grid[None],
                                                 mode=self.mode_seg, padding_mode="zeros", align_corners=False)[0][0]
                            mask = tmp[i] > (0.7 * scale_factor)
                            result_seg[c][mask] = u
                            done_mask = done_mask | mask
                        if not torch.all(done_mask):
                            result_seg[c][~done_mask] = labels[tmp[:, ~done_mask].argmax(0)]
                        del tmp
            del grid
            return result_seg.contiguous()

    def _apply_to_regr_target(self, regression_target, **params) -> torch.Tensor:
        return self._apply_to_image(regression_target, **params)

    def _apply_to_keypoints(self, keypoints, **params):
        raise NotImplementedError

    def _apply_to_bbox(self, bbox, **params):
        raise NotImplementedError

class Misalign2(BasicTransform):
    def __init__(self,
                 im_channels_2_misalign: Tuple[int,] = [0, ],
                 asynchron: bool = False,

                 squeezing_xyz: Tuple[float, ...] = (0.1, 0, 0),
                 p_squeeze: float = 0.0,

                 rotation_sag_cor_ax: Tuple[float, ...] = (np.pi, np.pi, np.pi),
                 rad_or_deg = None,
                 p_rotation: float = 0.0,

                 shift_zyx: Tuple[int, ...] = (2, 32, 32),
                 p_shift: float = 0.0,

                 ):
        super().__init__()
        self.im_channels_2_misalign = im_channels_2_misalign
        self.asynchron = asynchron

        self.squeezingXYZ = squeezing_xyz
        self.p_squeeze = p_squeeze

        if rad_or_deg == "rad":
            if any(rot > np.pi for rot in rotation_sag_cor_ax):
                raise ValueError("The rotation is probably in deg")
            self.rotation_sag_cor_ax = rotation_sag_cor_ax
        elif rad_or_deg == "deg":
            self.rotation_sag_cor_ax = [rot/360*(2*np.pi) for rot in rotation_sag_cor_ax]
        else:
            raise RuntimeError('Please define the rad_or_deg: "rad"/"deg"')
        self.p_rotation = p_rotation

        self.shiftZYX = shift_zyx
        self.p_shift = p_shift

    def get_parameters(self, **data_dict) -> dict:
        dim = data_dict['image'].ndim - 1
        # grid center must be in [-1, 1] as required by grid_sample
        shape = data_dict['image'].shape[1:]

        do_squeeze = np.random.uniform() < self.p_squeeze
        do_rotation = np.random.uniform() < self.p_rotation
        do_shift = np.random.uniform() < self.p_shift

        if do_squeeze:
            squeeze = []
            for d in range(dim):
                squeeze.append(np.random.uniform(1 - self.squeezingXYZ[d], 1 + self.squeezingXYZ[d]))
        else:
            squeeze = [1] * dim


        if do_rotation:
            angles = []
            for d in range(dim):
                angles.append(np.random.uniform(-self.rotation_sag_cor_ax[d], self.rotation_sag_cor_ax[d]))
        else:
            angles = [0] * dim


        if do_shift:
            if not self.asynchron:
                triplet = []
                center_location_in_pixels = []
                for d in range(dim):
                    triplet.append(shape[d] / 2 + np.random.randint(-self.shiftZYX[d], self.shiftZYX[d] + 1))
                center_location_in_pixels.append(triplet)
            else:
                raise NotImplementedError
        else:
            center_location_in_pixels = [[i / 2 for i in shape]]


        # affine matrix
        if do_squeeze or do_rotation:
            if dim == 3:
                affine = create_affine_matrix_3d(angles, squeeze)
            elif dim == 2:
                affine = create_affine_matrix_2d(angles[0], squeeze)
            else:
                raise RuntimeError(f'Unsupported dimension: {dim}')
        else:
            affine = None  # this will allow us to detect that we can skip computations

        offsets = None
        return {
            'affine': affine,
            'elastic_offsets': offsets,
            'center_location_in_pixels': center_location_in_pixels
        }

    def _apply_to_image(self, img: torch.Tensor, **params) -> torch.Tensor:
        shape = img.shape[1:]
        if params['affine'] is None and params['elastic_offsets'] is None:
            # No spatial transformation is being done. Round grid_center and crop without having to interpolate.
            # This saves compute.
            # cropping requires the center to be given as integer coordinates

            if not self.asynchron:
                for ch in self.im_channels_2_misalign:
                    img[ch, ...] = crop_tensor(img[ch, ...].unsqueeze(0), [math.floor(i) for i in params['center_location_in_pixels'][0]], shape, pad_mode='constant',
                                      pad_kwargs={'value': 0})
            else:
                raise NotImplementedError
            return img
        else:
            grid = _create_identity_grid(shape)

            if not self.asynchron:
                # we deform first, then rotate
                if params['elastic_offsets'] is not None:
                    grid += params['elastic_offsets']
                if params['affine'] is not None:
                    grid = torch.matmul(grid, torch.from_numpy(params['affine']).float())

                # we center the grid around the center_location_in_pixels. We should center the mean of the grid, not the center position
                mn = grid.mean(dim=list(range(img.ndim - 1)))
                new_center = torch.Tensor(
                    # [(j / (i / 2) - 1) for i, j in zip(img.shape[1:], params['center_location_in_pixels'][0])])
                    [(j / (i / 2) - 1) for i, j in zip(img.shape[1:], [i / 2 for i in shape])])

                grid += - mn + torch.flip(new_center, dims=[0])

                for ch in self.im_channels_2_misalign:
                    img[ch, ...] = grid_sample(img[ch, ...].unsqueeze(0).unsqueeze(0), grid[None], mode='bilinear', padding_mode="zeros", align_corners=False)[0]
                    img[ch, ...] = crop_tensor(img[ch, ...].unsqueeze(0), [math.floor(i) for i in params['center_location_in_pixels'][0]], shape, pad_mode='constant',
                                               pad_kwargs={'value': 0})
            else:
                raise NotImplementedError

            return img


    def _apply_to_segmentation(self, segmentation: torch.Tensor, **params) -> torch.Tensor:
        return segmentation.contiguous()

    def _apply_to_regr_target(self, regression_target, **params) -> torch.Tensor:
        return self._apply_to_image(regression_target, **params)

    def _apply_to_keypoints(self, keypoints, **params):
        raise NotImplementedError

    def _apply_to_bbox(self, bbox, **params):
        raise NotImplementedError


def create_affine_matrix_3d(rotation_angles, scaling_factors):
    # Rotation matrices for each axis
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(rotation_angles[0]), -np.sin(rotation_angles[0])],
                   [0, np.sin(rotation_angles[0]), np.cos(rotation_angles[0])]])

    Ry = np.array([[np.cos(rotation_angles[1]), 0, np.sin(rotation_angles[1])],
                   [0, 1, 0],
                   [-np.sin(rotation_angles[1]), 0, np.cos(rotation_angles[1])]])

    Rz = np.array([[np.cos(rotation_angles[2]), -np.sin(rotation_angles[2]), 0],
                   [np.sin(rotation_angles[2]), np.cos(rotation_angles[2]), 0],
                   [0, 0, 1]])

    # Scaling matrix
    S = np.diag(scaling_factors)

    # Combine rotation and scaling
    RS = Rz @ Ry @ Rx @ S
    return RS


def create_affine_matrix_2d(rotation_angle, scaling_factors):
    # Rotation matrix
    R = np.array([[np.cos(rotation_angle), -np.sin(rotation_angle)],
                  [np.sin(rotation_angle), np.cos(rotation_angle)]])

    # Scaling matrix
    S = np.diag(scaling_factors)

    # Combine rotation and scaling
    RS = R @ S
    return RS


def _create_identity_grid(size: List[int]) -> Tensor:
    space = [torch.linspace((-s + 1) / s, (s - 1) / s, s) for s in size[::-1]]
    grid = torch.meshgrid(space, indexing="ij")
    grid = torch.stack(grid, -1)
    spatial_dims = list(range(len(size)))
    grid = grid.permute((*spatial_dims[::-1], len(size)))
    return grid


class Misalign_fix(BasicTransform):
    def __init__(self,
                 im_channels_2_misalign: Tuple[int,] = [0, ],

                 squeezing_zyx: Tuple[float, ...] = (0.0, 0.0, 0.0),
                 p_squeeze: float = 0.0,

                 rotation_ax_cor_sag: Tuple[float, ...] = (5, 5, 5),
                 rad_or_deg = "deg",
                 p_rotation: float = 0.1,

                 shift_zyx: Tuple[int, ...] = (0, 2, 2),
                 p_shift: float = 0.1,
                 ):
        super().__init__()
        self.im_channels_2_misalign = im_channels_2_misalign

        self.squeezingZYX = squeezing_zyx
        self.p_squeeze = p_squeeze

        if rad_or_deg == "rad":
            if any(rot > np.pi/12 for rot in rotation_ax_cor_sag):
                raise Warning("The rotation is probably too big")
            if any(rot > np.pi for rot in rotation_ax_cor_sag):
                raise ValueError("The rotation is probably in deg or bigger than 180°")
            self.rotation_ax_cor_sag = rotation_ax_cor_sag
        elif rad_or_deg == "deg":
            self.rotation_ax_cor_sag = [rot/360*(2*np.pi) for rot in rotation_ax_cor_sag]
        else:
            raise RuntimeError('Please define the rad_or_deg: "rad"/"deg"')
        self.p_rotation = p_rotation

        self.shiftZYX = shift_zyx
        self.p_shift = p_shift

    def get_parameters(self, **data_dict) -> dict:
        dim = data_dict['image'].ndim - 1

        do_squeeze = np.random.uniform() < self.p_squeeze
        do_rotation = np.random.uniform() < self.p_rotation
        do_shift = np.random.uniform() < self.p_shift
        do_deform = False

        # Squeeze
        if do_squeeze:
            squeezes = [np.random.uniform(1 - self.squeezingZYX[i], 1 + self.squeezingZYX[i]) for i in range(dim)]
        else:
            squeezes = [1] * dim

        # Rotation
        if do_rotation:
            angles = [np.random.uniform(-self.rotation_ax_cor_sag[i], self.rotation_ax_cor_sag[i]) for i in range(dim)]
        else:
            angles = [0] * dim

        # affine matrix
        if do_squeeze or do_rotation:
            if dim == 3:
                affine = create_affine_matrix_3d(angles, squeezes)
            elif dim == 2:
                affine = create_affine_matrix_2d(angles[-1], squeezes)
            else:
                raise RuntimeError(f'Unsupported dimension: {dim}')
        else:
            affine = None  # this will allow us to detect that we can skip computations

        # elastic deformation. We need to create the displacement field here
        # we use the method from augment_spatial_2 in batchgenerators
        if do_deform:
            if np.random.uniform() <= self.p_synchronize_def_scale_across_axes:
                deformation_scales = [
                    sample_scalar(self.elastic_deform_scale, image=data_dict['image'], dim=None, patch_size=self.patch_size)
                    ] * dim
            else:
                deformation_scales = [
                    sample_scalar(self.elastic_deform_scale, image=data_dict['image'], dim=i, patch_size=self.patch_size)
                    for i in range(0, 3)
                    ]

            # sigmas must be in pixels, as this will be applied to the deformation field
            sigmas = [i * j for i, j in zip(deformation_scales, self.patch_size)]

            magnitude = [
                sample_scalar(self.elastic_deform_magnitude, image=data_dict['image'], patch_size=self.patch_size,
                              dim=i, deformation_scale=deformation_scales[i])
                for i in range(0, 3)]
            # doing it like this for better memory layout for blurring
            offsets = torch.normal(mean=0, std=1, size=(dim, *self.patch_size))

            # all the additional time elastic deform takes is spent here
            for d in range(dim):
                # fft torch, slower
                # for i in range(offsets.ndim - 1):
                #     offsets[d] = blur_dimension(offsets[d][None], sigmas[d], i, force_use_fft=True, truncate=6)[0]

                # fft numpy, this is faster o.O
                tmp = np.fft.fftn(offsets[d].numpy())
                tmp = fourier_gaussian(tmp, sigmas[d])
                offsets[d] = torch.from_numpy(np.fft.ifftn(tmp).real)

                mx = torch.max(torch.abs(offsets[d]))
                offsets[d] /= (mx / np.clip(magnitude[d], a_min=1e-8, a_max=np.inf))
            offsets = torch.permute(offsets, (1, 2, 3, 0))
        else:
            offsets = None

        # shape = data_dict['image'].shape[1:]
        # if do_shift:
        #     for i in shape:
        #         print(i)
        #     center_location_in_pixels = [i / 2 + np.random.randint(self.shiftXYZ[j], self.shiftXYZ[j]+1) for i, j in zip(shape, range(dim - 1, -1, -1))][::-1]
        # else:
        #     center_location_in_pixels = [i / 2 for i in shape][::-1]

        shape = data_dict['image'].shape[1:]
        if not do_shift:
            center_location_in_pixels = [i / 2 for i in shape]
        else:
            center_location_in_pixels = [shape[i] / 2 + np.random.randint(-self.shiftZYX[i], self.shiftZYX[i]+1) for i in range(dim)]

        return {
            'affine': affine,
            'elastic_offsets': offsets,
            'center_location_in_pixels': center_location_in_pixels
        }

    def _apply_to_image(self, img: torch.Tensor, **params) -> torch.Tensor:
        im_shape = img.shape[1:]
        if params['affine'] is None and params['elastic_offsets'] is None:
            for ch in self.im_channels_2_misalign:
                img[ch, ...] = crop_tensor(img[ch, ...].unsqueeze(0), [math.floor(i) for i in params['center_location_in_pixels']], im_shape,
                                           pad_mode='constant', pad_kwargs={'value': 0})
            return img
        else:
            grid = _create_centered_identity_grid2(im_shape)

            # we deform first, then rotate
            if params['elastic_offsets'] is not None:
                grid += params['elastic_offsets']
            if params['affine'] is not None:
                grid = torch.matmul(grid, torch.from_numpy(params['affine']).float())

            # we center the grid around the center_location_in_pixels. We should center the mean of the grid, not the center position
            # only do this if we elastic deform
            if params['elastic_offsets'] is not None:
                mn = grid.mean(dim=list(range(img.ndim - 1)))
            else:
                mn = 0

            # new_center = torch.Tensor([c - s / 2 for c, s in zip(params['center_location_in_pixels'], img.shape[1:])])
            new_center = torch.Tensor([0, 0, 0])
            grid += (new_center - mn)

            for ch in self.im_channels_2_misalign:
                img[ch, ...] = grid_sample(img[ch, ...].unsqueeze(0).unsqueeze(0), _convert_my_grid_to_grid_sample_grid(grid, img.shape[1:])[None],
                               mode='bilinear', padding_mode="zeros", align_corners=False)[0]
                img[ch, ...] = crop_tensor(img[ch, ...].unsqueeze(0), [math.floor(i) for i in params['center_location_in_pixels']], im_shape,
                                           pad_mode='constant', pad_kwargs={'value': 0})
            return img

    def _apply_to_segmentation(self, segmentation: torch.Tensor, **params) -> torch.Tensor:
        return segmentation.contiguous()

    def _apply_to_regr_target(self, regression_target, **params) -> torch.Tensor:
        return self._apply_to_image(regression_target, **params)

    def _apply_to_keypoints(self, keypoints, **params):
        raise NotImplementedError

    def _apply_to_bbox(self, bbox, **params):
        raise NotImplementedError

def _create_centered_identity_grid2(size: Union[Tuple[int, ...], List[int]]) -> torch.Tensor:
    space = [torch.linspace((1 - s) / 2, (s - 1) / 2, s) for s in size]
    grid = torch.meshgrid(space, indexing="ij")
    grid = torch.stack(grid, -1)
    return grid


def _convert_my_grid_to_grid_sample_grid(my_grid: torch.Tensor, original_shape: Union[Tuple[int, ...], List[int]]):
    # rescale
    for d in range(len(original_shape)):
        s = original_shape[d]
        my_grid[..., d] /= (s / 2)
    my_grid = torch.flip(my_grid, (len(my_grid.shape) - 1, ))
    # my_grid = my_grid.flip((len(my_grid.shape) - 1,))
    return my_grid