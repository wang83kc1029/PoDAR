import numpy as np
from second.core.point_cloud.point_cloud_ops import points_to_voxel


class VoxelGenerator:
    def __init__(self,
                 voxel_size,
                 point_cloud_range,
                 max_num_points,
                 max_voxels=20000):
        point_cloud_range = np.array(point_cloud_range, dtype=np.float32)
        # [0, -40, -3, 70.4, 40, 1]
        voxel_size = np.array(voxel_size, dtype=np.float32)
        grid_size = (
            point_cloud_range[3:] - point_cloud_range[:3]) / voxel_size
        grid_size = np.round(grid_size).astype(np.int64)

        self._voxel_size = voxel_size
        self._point_cloud_range = point_cloud_range
        self._max_num_points = max_num_points
        self._max_voxels = max_voxels
        self._grid_size = grid_size

    def generate(self, points, max_voxels):
        return points_to_voxel(
            points, self._voxel_size, self._point_cloud_range,
            self._max_num_points, True, max_voxels)

    @property
    def voxel_size(self):
        return self._voxel_size

    @property
    def max_num_points_per_voxel(self):
        return self._max_num_points


    @property
    def point_cloud_range(self):
        return self._point_cloud_range

    @property
    def grid_size(self):
        return self._grid_size

class VoxelGenerator_rp:
    def __init__(self,
                 voxel_size,
                 point_cloud_range,
                 max_num_points,
                 max_voxels=20000):
        point_cloud_range = np.array(point_cloud_range, dtype=np.float32)
        # [0, -0.78539816, -3, 80.0, 0.78539816, 1] / r, phi, h
        voxel_size = np.array(voxel_size, dtype=np.float32)

        grid_size = (
            point_cloud_range[3:] - point_cloud_range[:3]) / voxel_size
        grid_size = np.round(grid_size).astype(np.int64)
        self._voxel_size = voxel_size
        self._point_cloud_range = point_cloud_range
        self._max_num_points = max_num_points
        self._max_voxels = max_voxels
        self._grid_size = grid_size

    def _points_to_rp(self, points):
        r = np.linalg.norm(points[:, :2], axis=1)
        points[:, 1] = np.arctan2(points[:, 1], points[:, 0])
        points[:, 0] = r
        return points

    def _random_occlusion(self, pts):
        q_hor = ((pts[:, 1] - (-np.pi/4)) / (np.pi/2) * 512).astype(np.int32)
        max_block = np.random.choice([16, 32, 64])
        block_size = np.random.randint(max_block//2)
        block_shift = np.random.randint(max_block)
        block_idx = ((q_hor-block_shift)%max_block)<block_size
        pts[block_idx, 0] /= pts[block_idx][:, 0]
        pts[block_idx, 2] /= pts[block_idx][:, 0]
        return pts

    def generate(self, points, max_voxels):
        pts = self._points_to_rp(points)
        return points_to_voxel(
            pts, self._voxel_size, self._point_cloud_range,
            self._max_num_points, True, max_voxels) 

    @property
    def voxel_size(self):
        return self._voxel_size

    @property
    def max_num_points_per_voxel(self):
        return self._max_num_points


    @property
    def point_cloud_range(self):
        return self._point_cloud_range

    @property
    def grid_size(self):
        return self._grid_size