import numpy as np
import open3d as o3d

# TRANSLATION MULTIPLIER
X = 2000.0

# READ POINT CLOUD FILE (.ply)
pcd1 = o3d.io.read_point_cloud("20240521_image2_5.ply")     # FRONT
pcd2 = o3d.io.read_point_cloud("20240521_image2_6.ply")     # LEFT
pcd3 = o3d.io.read_point_cloud("20240521_image2_7.ply")     # BACK
pcd4 = o3d.io.read_point_cloud("20240521_image2_8.ply")     # RIGHT

# ==================== ARRANGE POINT CLOUDS ======================= #

# ------------------- FRONT -------------------- #
# REMOVE OUTLIER POINTS
cl, ind = pcd1.remove_statistical_outlier(nb_neighbors=300, std_ratio=0.5)
pcd1 = pcd1.select_by_index(ind)
cl, ind = pcd1.remove_radius_outlier(nb_points=10, radius=300)
pcd1 = pcd1.select_by_index(ind)

# TRANSFORM POINT CLOUD
pcd1.translate((0, 0, 0))
pcd1.rotate(
    pcd1.get_rotation_matrix_from_xyz((
        0     - (np.pi * 16/128),    # X Rotation   
        np.pi + (np.pi *  3/128),    # Y Rotation   
        np.pi + (np.pi *  2/128)     # Z Rotation   
    )),
    center=pcd1.get_center()
)
pcd1.scale(1.0, center=pcd1.get_center())

# GET FINAL BOUNDING BOX
box = pcd1.get_axis_aligned_bounding_box()
box.color = (1, 0, 0)

# DISPLAY RESULTS
# o3d.visualization.draw_geometries([pcd1, box])


# ------------------- BACK -------------------- #
# REMOVE OUTLIER POINTS
cl, ind = pcd3.remove_statistical_outlier(nb_neighbors=300, std_ratio=0.5)
pcd3 = pcd3.select_by_index(ind)
cl, ind = pcd3.remove_radius_outlier(nb_points=10, radius=300)
pcd3 = pcd3.select_by_index(ind)

# TRANSFORM POINT CLOUD
pcd3.translate((-0.4 * X, -0.5 * X, -5.2 * X))
pcd3.rotate(
    pcd3.get_rotation_matrix_from_xyz((
        0     + (np.pi * 1/128),    # X Rotation
        0     + (np.pi * 3/128),    # Y Rotation
        np.pi + (np.pi * 3/128)     # Z Rotation
    )),
    center=pcd3.get_center()
)
pcd3.scale(1.0, center=pcd3.get_center())

# WARP POINT CLOUD
pcd3.transform([[1, 0, 0, 0], 
                [0, 1, 0, 0], 
                [0, -0.05, 1, 0], 
                [0, 0, 0, 1]])

# DISPLAY RESULTS
# o3d.visualization.draw_geometries([pcd3, box])


# ------------------- LEFT -------------------- #
# REMOVE OUTLIER POINTS
cl, ind = pcd2.remove_statistical_outlier(nb_neighbors=300, std_ratio=0.25)
pcd2 = pcd2.select_by_index(ind)
cl, ind = pcd2.remove_radius_outlier(nb_points=10, radius=250)
pcd2 = pcd2.select_by_index(ind)


# TRANSFORM POINT CLOUD
pcd2.translate((1.5 * X, -0.5 * X, -6 * X))
pcd2.rotate(
    pcd2.get_rotation_matrix_from_xyz((
        0     - (np.pi * 106/128),    # X Rotation
        np.pi + (np.pi *  56/128),    # Y Rotation
        np.pi - (np.pi * 102/128)     # Z Rotation
    )),
    center=pcd2.get_center()
)
pcd2.scale(0.9, center=pcd2.get_center())

# DISPLAY RESULTS
# o3d.visualization.draw_geometries([pcd2, box])


# ------------------- RIGHT -------------------- #
# REMOVE OUTLIER POINTS
cl, ind = pcd4.remove_statistical_outlier(nb_neighbors=300, std_ratio=0.25)
pcd4 = pcd4.select_by_index(ind)
cl, ind = pcd4.remove_radius_outlier(nb_points=7, radius=240)
pcd4 = pcd4.select_by_index(ind)

# TRANSFORM POINT CLOUD
pcd4.translate((-3.3 * X, -1 * X, -14 * X))
pcd4.rotate(
    pcd4.get_rotation_matrix_from_xyz((
        0     - (np.pi *  6/128),    # X Rotation   
        np.pi - (np.pi * 54/128),    # Y Rotation   
        np.pi + (np.pi *  4/128)     # Z Rotation   
    )),
    center=pcd4.get_center()
)
pcd4.scale(0.68, center=pcd4.get_center())

# WARP POINT CLOUD
pcd4.transform([[1, 0.1, 0, 0], 
                [0, 1, 0, 0], 
                [0, 0.1, 1, 0], 
                [0, 0, 0, 1]])

# DISPLAY RESULTS
# o3d.visualization.draw_geometries([pcd4, box])


# ==================== MERGE POINT CLOUDS ======================= #

# MERGE ALL 4 POINT CLOUDS
merged_pcd = pcd1 + pcd2 + pcd3 + pcd4
o3d.visualization.draw_geometries([merged_pcd])

# # REMOVE REMAINING OUTLIERS
cl, ind = merged_pcd.remove_statistical_outlier(nb_neighbors=100, std_ratio=1.75)
merged_pcd = merged_pcd.select_by_index(ind)
cl, ind = merged_pcd.remove_radius_outlier(nb_points=10, radius=300)
merged_pcd = merged_pcd.select_by_index(ind)
# o3d.visualization.draw_geometries([merged_pcd])

# SAVE MERGED POINT CLOUD
o3d.io.write_point_cloud("merged_pcd.ply", merged_pcd)

# ==================== CONVERT POINT CLOUDS TO MESH ======================= #

# DOWN SAMPLE POINT CLOUD
down_pcd = merged_pcd.voxel_down_sample(voxel_size=1000)
# o3d.visualization.draw_geometries([down_pcd])

# ESTIMATE POINT CLOUD NORMALS
down_pcd.normals = o3d.utility.Vector3dVector(np.zeros((1, 3)))
down_pcd.estimate_normals()
o3d.visualization.draw_geometries([down_pcd], point_show_normal=True)

# TO CONVERT POINT COULD TO A 3D MESH, THERE ARE CURRENTLY TWO METHODS I TESTED TO BE WORKING:

# METHOD 1: CREATE MESH FROM CONVEX HULL
# Result: Creates a fully-formed 3D Mesh, however the colors from the point clouds was not retained
hull_mesh, _ = down_pcd.compute_convex_hull()
hull_mesh.compute_vertex_normals()
o3d.visualization.draw_geometries([hull_mesh], mesh_show_back_face=True)

# METHOD 2: CREATE MESH FROM BALL PIVOTING ALGORITHM
# Result: Creates a 3D Mesh with the colors retained, however there are holes in the geometry that must be filled
radii = 2.0 * np.mean(down_pcd.compute_nearest_neighbor_distance())
bp_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(down_pcd, o3d.utility.DoubleVector([radii, radii * 10]))
o3d.visualization.draw_geometries([bp_mesh], mesh_show_back_face=True)

# SAVE MESH FILES
o3d.io.write_triangle_mesh("mesh_hull.ply", hull_mesh)
o3d.io.write_triangle_mesh("mesh_bp.ply", bp_mesh)