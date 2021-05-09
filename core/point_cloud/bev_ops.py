import numba
import numpy as np
import math


@numba.jit(nopython=True)
def _points_to_bevmap_reverse_kernel(points,
                                     voxel_size,
                                     coors_range,
                                     coor_to_voxelidx,
                                     # coors_2d,
                                     bev_map,
                                     height_lowers,
                                     phi_lowers,
                                     # density_norm_num=16,
                                     with_reflectivity=False,
                                     max_voxels=40000):
    # put all computations to one loop.
    # we shouldn't create large array in main jit code, otherwise
    # reduce performance
    N = points.shape[0]
    ndim = points.shape[1] - 1
    # ndim = 3
    ndim_minus_1 = ndim - 1
    grid_size = (coors_range[3:] - coors_range[:3]) / voxel_size
    # np.round(grid_size)
    # grid_size = np.round(grid_size).astype(np.int64)(np.int32)
    grid_size = np.round(grid_size, 0, grid_size).astype(np.int32)
    height_slice_size = voxel_size[-1]
    phi_slice_size = voxel_size[-2]
    coor = np.zeros(shape=(3, ), dtype=np.int32)  # DHW
    voxel_num = 0
    failed = False
    for i in range(N):
        failed = False
        for j in range(ndim):
            c = np.floor((points[i, j] - coors_range[j]) / voxel_size[j])
            if c < 0 or c >= grid_size[j]:
                failed = True
                break
            coor[ndim_minus_1 - j] = c
        if failed:
            continue
        voxelidx = coor_to_voxelidx[coor[0], coor[1], coor[2]]
        if voxelidx == -1:
            voxelidx = voxel_num
            if voxel_num >= max_voxels:
                break
            voxel_num += 1
            coor_to_voxelidx[coor[0], coor[1], coor[2]] = voxelidx
            # coors_2d[voxelidx] = coor[1:]
        bev_map[-1, coor[1], coor[2]] += 1
        height_norm = bev_map[coor[0], coor[1], coor[2]]
        incomimg_height_norm = (
            points[i, 2] - height_lowers[coor[0]]) / height_slice_size
        if incomimg_height_norm > height_norm:
            incomimg_phi_norm = (
                points[i, 1] - phi_lowers[coor[1]]) / phi_slice_size
            bev_map[coor[0], coor[1], coor[2]] = incomimg_height_norm
            bev_map[coor[0] + grid_size[-1], coor[1], coor[2]] = incomimg_phi_norm
            if with_reflectivity:
                bev_map[-2, coor[1], coor[2]] = points[i, 3]
    # return voxel_num


def points_to_bev(points,
                  voxel_size,
                  coors_range,
                  with_reflectivity=False,
                  density_norm_num=16,
                  max_voxels=40000):
    """convert kitti points(N, 4) to a bev map. return [C, H, W] map.
    this function based on algorithm in points_to_voxel.
    takes 5ms in a reduced pointcloud with voxel_size=[0.1, 0.1, 0.8]

    Args:
        points: [N, ndim] float tensor. points[:, :3] contain xyz points and
            points[:, 3] contain reflectivity.
        voxel_size: [3] list/tuple or array, float. xyz, indicate voxel size
        coors_range: [6] list/tuple or array, float. indicate voxel range.
            format: xyzxyz, minmax
        with_reflectivity: bool. if True, will add a intensity map to bev map.
    Returns:
        bev_map: [num_height_maps + 1(2), H, W] float tensor. 
            `WARNING`: bev_map[-1] is num_points map, NOT density map, 
            because calculate density map need more time in cpu rather than gpu. 
            if with_reflectivity is True, bev_map[-2] is intensity map. 
    """
    if not isinstance(voxel_size, np.ndarray):
        voxel_size = np.array(voxel_size, dtype=points.dtype)
    if not isinstance(coors_range, np.ndarray):
        coors_range = np.array(coors_range, dtype=points.dtype)
    voxelmap_shape = (coors_range[3:] - coors_range[:3]) / voxel_size
    voxelmap_shape = tuple(np.round(voxelmap_shape).astype(np.int32).tolist())
    voxelmap_shape = voxelmap_shape[::-1]  # DHW format
    coor_to_voxelidx = -np.ones(shape=voxelmap_shape, dtype=np.int32)
    # coors_2d = np.zeros(shape=(max_voxels, 2), dtype=np.int32)
    bev_map_shape = list(voxelmap_shape)
    bev_map_shape[0] *= 2
    bev_map_shape[0] += 1
    height_lowers = np.linspace(
        coors_range[2], coors_range[5], voxelmap_shape[0], endpoint=False)
    phi_lowers = np.linspace(
        coors_range[1], coors_range[4], voxelmap_shape[1], endpoint=False)
    if with_reflectivity:
        bev_map_shape[0] += 1
    bev_map = np.zeros(shape=bev_map_shape, dtype=points.dtype)
    _points_to_bevmap_reverse_kernel(
        points, voxel_size, coors_range, coor_to_voxelidx, bev_map,
        height_lowers, phi_lowers, with_reflectivity, max_voxels)
    return bev_map

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    p = np.fromfile('C:/KITTI/3D-Object-Detection-Baseline/data/velodyne/006034.bin', np.float32).reshape(-1, 4)
    p = p[p[:, 0]>0]
    r = np.linalg.norm(p[:, :2], axis=1)
    p[:, 1] = np.arctan2(p[:, 1], p[:, 0])
    p[:, 0] = r
    p = np.ones((100, 4))
    p[:, 0] = np.arange(100)* 0.2
    p[:, 1] = 0.0030679615758 * 0
    p[:, 2] = 0
    bev = points_to_bev(p, [0.1, 0.0030679615758, 0.4], [0, -0.78539816, -3, 80, 0.78539816, 1],
        True, max_voxels=40960000)
    print(bev.shape)
    plt.scatter(p[:,0], p[:,1], s=0.1, c=p[:,2])
    #plt.axis('equal')
    plt.figure()
    plt.imshow(np.max(bev[10:20], 0))
    plt.show()