import numpy as np

from second.core.voxel_generator import VoxelGenerator, VoxelGenerator_rp
from second.protos import voxel_generator_pb2


def build(voxel_config):
    """Builds a tensor dictionary based on the InputReader config.

    Args:
        input_reader_config: A input_reader_pb2.InputReader object.

    Returns:
        A tensor dict based on the input_reader_config.

    Raises:
        ValueError: On invalid input reader proto.
        ValueError: If no input paths are specified.
    """
    if not isinstance(voxel_config, (voxel_generator_pb2.VoxelGenerator)):
        raise ValueError('input_reader_config not of type '
                         'input_reader_pb2.InputReader.')
    if voxel_config.voxel_size[1]<0.1:
        voxel_generator = VoxelGenerator_rp(
            voxel_size=list(voxel_config.voxel_size),
            point_cloud_range=list(voxel_config.point_cloud_range),
            max_num_points=voxel_config.max_number_of_points_per_voxel,
            max_voxels=20000)        
    else:
        voxel_generator = VoxelGenerator(
            voxel_size=list(voxel_config.voxel_size),
            point_cloud_range=list(voxel_config.point_cloud_range),
            max_num_points=voxel_config.max_number_of_points_per_voxel,
            max_voxels=20000)
    return voxel_generator
