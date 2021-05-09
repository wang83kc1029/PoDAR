# PoDAR
Polar-based Object Detector for LiDAR Point Clouds with Direction-invariant

code from PointPillars (https://github.com/nutonomy/second.pytorch)

key idea: object detection by transforming point clouds into polar coordinate

========

modifications: 

----pytorch----

- models/pointpillars.py: modify pillar feature net to process on polar coordinate

- models/voxelnet.py: add low level conv blocks to bev_extractor (only used when using projected bev map as input)

- core/box_torch_ops.py: implement pytorch bounding box encode and decode on polar coordinates

----core----

- anchor_generaror.py: implement anchor generator on polar coordinate

- voxel_generator.py: implement class VoxelGenerator_rp for generating voxels (pillars) on polar coordinate

- region_similarity.py: modify class DistanceSimilarity for anchor matching

- box_np_ops.py: implement numpy bounding box encode and decode on polar coordinates and calculation of distance similarity

- target_ops.py: implement distance anchor matching method

----data----

preprocess.py: implement the generation of polar coordinate bev map

========

3D object detection performance on KITTI validation set

Car (mAP)

             Easy  Mod.  Hard
             
PointPillars 85.19 74.84 69.77 

PoDAR        85.01 74.05 69.01 

Pedestrian (mAP)

             Easy  Mod.  Hard
             
PointPillars 63.10 58.51 53.78 

PoDAR        70.17 63.68 58.88 

Cyclist (mAP)

             Easy  Mod.  Hard
             
PointPillars 79.44 62.03 58.40 

PoDAR        82.42 61.23 56.48 


=> greatly improve pedestrians performance
 
