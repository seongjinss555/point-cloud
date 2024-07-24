""" OBJECTIVE: Generate a 3D Mesh (.ply file) using Stereo Images
    Objective 1: DONE               [06/20/24]
    Objective 2: DONE               [07/04/24]
    Objective 3: DONE               [06/20/24]
    Objective 4: To be optimized... [06/20/24]
    Objective 5: To be optimized... [07/07/24]
    Objective 6: To be optimized... [07/07/24]
    Objective 7: To be added...
    Objective 8: To be added...
""" # STATUS: COMPLETED! [06/11/2024]

# IMPORT DEPENDENCIES
import os
import torch
import numpy as np
import cv2 as cv
import open3d as o3d
from matplotlib import pyplot as plt
# pip install timm

# GLOBAL VARIABLES
cam_calibration_dir_path = ".\calibration_images_new"
stereo_images_dir_path = ".\sample_stereo_images"
processed_images_dir_path = ".\processed"

# FUNCTION DEFINITIONS
def get_image_pairs(dir_path: str, left_substr="image2", right_substr="image1"):
    '''Retrieve one or more image pairs (in '.jpg' or '.png' format) from the target directory based on their substring keywords (ex. \"imageX\"), 
       and each image has a file name and image data.
    '''
    image_pairs = []

    for file in os.listdir(dir_path):
        if (left_substr in file) and (".jpg" or ".png" in file):
            img1 = [[file, cv.imread(os.path.join(dir_path, file))], None]
            image_pairs.append(img1)

    for img1 in image_pairs:
        image_name = img1[0][0]
        img2 = image_name.replace(left_substr, right_substr)
        for file in os.listdir(dir_path):
            if file == img2:
                img1[1] = [file, cv.imread(os.path.join(dir_path, file))]
    
    return image_pairs

def get_calib_images(dir_path, additional_path=""):
    '''Retrieve one or more calibration images (in '.jpg' or '.png' format) from the target directory, specifically its file name and image data.\n
       Optional Args: additional_path = add a more specific path to the main directory 'dir_path'
    '''
    images = []

    # Retrieve all files located inside specific directory
    file_path = os.path.join(dir_path, additional_path)
    print(f" >>> Searching directory: \"{file_path}\"")
    for file_name in os.listdir(file_path):
            # If file has a format of .jpg or .png and is not null, append to images[]
            if (".jpg" or ".png") in file_name:
                img = cv.imread(os.path.join(file_path, file_name))
                if img is not None:
                    images.append(img)
    
    print(f" >>> Found {len(images)} images for calibration...")
    return images

def calibrate_stereo_cam(cam_calibration_dir_path, chessboard_size=(10,7), frame_size=(640, 480)):
    '''Generate stereo camera calibration parameters from a set of calibration images for each camera\n
       Optional Args: chessboard_size = specify the number of corners between the first and second row, and between the first and 
                                                             second columns, respectively
                               frame_size = specify the width and height of the calibration images (must be uniform for all calibration images)
    
    '''
    # Declare constants & initialize variables
    SIZE_OF_CHESSBOARD_SQUARES_MM = 20  # Size of each Chessboard Square (mm)
    images_left = []                    # Images to be used for Calibration (left CCamera)
    images_right = []                   # Images to be used for Calibration (Right Camera)
    obj_points = []                     # 3D Point in Real World Space
    img_points_left = []                # 2D Points in Image Plane (Left Camera)
    img_points_right = []               # 2D Points in Image Plane (Right Camera)
    
    # Define Termination Criteria; If number of iterations=30 reached or error is less than epsilon=0.001, calibration is terminated
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Prepare Object Points
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
    objp = objp * SIZE_OF_CHESSBOARD_SQUARES_MM

    # Retrieve Images for Calibration
    images_left = get_calib_images(cam_calibration_dir_path, "left_cam")     # Retrieve Images For Left Camera
    images_right = get_calib_images(cam_calibration_dir_path, "right_cam")   # Retrieve Images For Right Camera

    # Find the Chessboard Corners for Left Camera & Right Camera Images
    print(" >>> Detecting Chessboard Corners...")

    img_count = 0
    for img_left, img_right in zip(images_left, images_right):
        # For each image pair, convert to grayscale for better results
        imgL = img_left
        imgR = img_right
        grayL = cv.cvtColor(imgL, cv.COLOR_BGR2GRAY)
        grayR = cv.cvtColor(imgR, cv.COLOR_BGR2GRAY)

        # Find the Chessboard Corners; If found, append object points and image points to respective lists
        retL, cornersL = cv.findChessboardCorners(grayL, chessboard_size, None)
        retR, cornersR = cv.findChessboardCorners(grayR, chessboard_size, None)

        if retL and retR == True:
            obj_points.append(objp)

            cornersL = cv.cornerSubPix(grayL, cornersL, (11,11), (-1,-1), criteria)
            img_points_left.append(cornersL)

            cornersR = cv.cornerSubPix(grayR, cornersR, (11,11), (-1,-1), criteria)
            img_points_right.append(cornersR)

            # Optional: Draw and Display the Corners
            # cv.drawChessboardCorners(imgL, chessboard_size, cornersL, retL)
            # cv.imshow("Left Camera: Image " + str(img_count + 1), imgL)
            # cv.drawChessboardCorners(imgR, chessboard_size, cornersR, retR)
            # cv.imshow("Right Camera: Image " + str(img_count + 1), imgR)
            # cv.waitKey(0)   # Press any key or close window directly
        
        else:
             print(f"   * ERROR: Unable to find corners for Image {img_count + 1}")
        img_count += 1

    cv.destroyAllWindows()

    # Calibrate the Left and Right Cameras
    retL, cameraMatrixL, distL, rvecsL, tvecsL = cv.calibrateCamera(obj_points, img_points_left, frame_size, None, None)
    heightL, widthL, channelsL = imgL.shape
    newCameraMatrixL, roi_L = cv.getOptimalNewCameraMatrix(cameraMatrixL, distL, (widthL, heightL), 1, (widthL, heightL))

    retR, cameraMatrixR, distR, rvecsR, tvecsR = cv.calibrateCamera(obj_points, img_points_right, frame_size, None, None)
    heightR, widthR, channelsR = imgR.shape
    newCameraMatrixR, roi_R = cv.getOptimalNewCameraMatrix(cameraMatrixR, distR, (widthR, heightR), 1, (widthR, heightR))

    # Fix Intrinsic Camera Matrices, so that only Rot, Trns, Emat and Fmat are calculated
    flags = 0
    flags |= cv.CALIB_FIX_INTRINSIC

    criteria_stereo = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001) # Same as termination criteria above
    retStereo, newCameraMatrixL, distL, newCameraMatrixR, distR, rot, trans, essentialMatrix, fundamentalMatrix = cv.stereoCalibrate(obj_points, img_points_left, img_points_right, newCameraMatrixL, distL, newCameraMatrixR, distR, grayL.shape[::-1], criteria_stereo, flags)

    # Optional: Calculate Reprojection Error
    mean_error = 0
    for i in range(len(obj_points)):
        imgpoints2, _ = cv.projectPoints(obj_points[i], rvecsR[i], tvecsR[i], newCameraMatrixR, distR)
        error = cv.norm(img_points_right[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
        mean_error += error

    # Optional: Display Total Error
    print(f" >>> Total Error: {mean_error/len(obj_points)}")

    # Rectify Stereo Camera Maps
    rectify_scale = 1
    rectL, rectR, projMatrixL, projMatrixR, Q, roi_L, roi_R= cv.stereoRectify(newCameraMatrixL, distL, newCameraMatrixR, distR, grayL.shape[::-1], rot, trans, rectify_scale,(0,0))

    stereoMapL = cv.initUndistortRectifyMap(newCameraMatrixL, distL, rectL, projMatrixL, grayL.shape[::-1], cv.CV_16SC2)
    stereoMapR = cv.initUndistortRectifyMap(newCameraMatrixR, distR, rectR, projMatrixR, grayR.shape[::-1], cv.CV_16SC2)

    # Save Calibration Parameters in a single .xml
    cv_file = cv.FileStorage('stereoMap.xml', cv.FILE_STORAGE_WRITE)

    cv_file.write('stereoMapL_x',stereoMapL[0])
    cv_file.write('stereoMapL_y',stereoMapL[1])
    cv_file.write('stereoMapR_x',stereoMapR[0])
    cv_file.write('stereoMapR_y',stereoMapR[1])
    cv_file.write('q', Q)
    cv_file.release()

def create_pcd(vertices, colors, filename):
	colors = colors.reshape(-1,3)
	vertices = np.hstack([vertices.reshape(-1,3),colors])

	ply_header = '''ply
		format ascii 1.0
		element vertex %(vert_num)d
		property float x
		property float y
		property float z
		property uchar red
		property uchar green
		property uchar blue
		end_header
		'''
	with open(filename, 'w') as f:
		f.write(ply_header %dict(vert_num=len(vertices)))
		np.savetxt(f,vertices,'%f %f %f %d %d %d')

# ================================= PROGRAM START ================================== #

# --------------------------------- OBJECTIVE-1 --------------------------------- #
print(f"\nRETRIEVING STEREO IMAGES FROM: \"{stereo_images_dir_path}\"")     # DEBUG: Display process & target directory

image_pairs_raw = get_image_pairs(stereo_images_dir_path)

print(f" >>> Found {len(image_pairs_raw)} image pairs...")                  # DEBUG: Display number of image pairs

# Optional: Display Raw Images
# for image_pair in image_pairs_raw:                                          # DEBUG: Display each image filename & image data
#     cv.imshow(image_pair[1][0], image_pair[1][1])
#     cv.waitKey(0)
#     cv.imshow(image_pair[0][0], image_pair[0][1])
#     cv.waitKey(0)

# ---------------------------------- OBJ-1 END ---------------------------------- #


# --------------------------------- OBJECTIVE-2 --------------------------------- #

print(f"\nRETRIEVING CALIBRATION IMAGES FROM: \"{cam_calibration_dir_path}\"")     # DEBUG: Display process & target directory

calibrate_stereo_cam(cam_calibration_dir_path)

print(f"CAMERA PARAMETERS GENERATED WITH FILENAME: \"stereoMap.xml\"")             # DEBUG

# ---------------------------------- OBJ-2 END ---------------------------------- #


# --------------------------------- OBJECTIVE-3 --------------------------------- #

print(f"\nUNDISTORTING {len(image_pairs_raw)} IMAGE PAIRS:")     # DEBUG: Display process & target directory

cam_params = cv.FileStorage("stereoMap.xml", cv.FILE_STORAGE_READ)

if cam_params.isOpened():                                                               # DEBUG: Confirm whether camera parameters exist
    print(" >>> Using Camera Parameters with filename: \"stereoMap.xml\"")
else:
    print(" >>> No Camera Parameters found! Perform Camera Calibration first.")
    exit()

stereoMapL_x = cam_params.getNode("stereoMapL_x").mat()
stereoMapL_y = cam_params.getNode("stereoMapL_y").mat()
stereoMapR_x = cam_params.getNode("stereoMapR_x").mat()
stereoMapR_y = cam_params.getNode("stereoMapR_y").mat()

matrixQ = cam_params.getNode("q").mat()

image_pairs_undistorted = []
for imp in image_pairs_raw:
    undistortedL = cv.remap(imp[1][1], stereoMapL_x, stereoMapL_y, cv.INTER_LANCZOS4, cv.BORDER_CONSTANT, 0)
    undistortedR = cv.remap(imp[0][1], stereoMapR_x, stereoMapR_y, cv.INTER_LANCZOS4, cv.BORDER_CONSTANT, 0)

    imp_undistorted = [[imp[0][0], undistortedR], [imp[1][0], undistortedL]]
    image_pairs_undistorted.append(imp_undistorted)

print(f"PERFORMED UNDISTORTION ON {len(image_pairs_undistorted)} IMAGE PAIRS")  # DEBUG: Display number of undistorted image pairs

# Optional: Display Undistorted Images
# for image_pair in image_pairs_undistorted:                                              # DEBUG: Display each image filename & image data
#     cv.imshow(image_pair[1][0], image_pair[1][1])
#     cv.waitKey(0)
#     cv.imshow(image_pair[0][0], image_pair[0][1])
#     cv.waitKey(0)

# ---------------------------------- OBJ-3 END ---------------------------------- #


# --------------------------------- OBJECTIVE-4 --------------------------------- #

#DEBUG: Skip Calibration & Undistortion for the meantime
# image_pairs_undistorted = image_pairs_raw

print(f"\nCREATING OBJECT MASKS FOR {len(image_pairs_undistorted)} IMAGE PAIRS:")     # DEBUG: Display process & target directory

image_masks = []

for im in image_pairs_undistorted:
    img_raw = im[0][1]   # Switch between im[0][1] and im[1][1] accordingly
    # cv.imshow("RAW", img_raw)
    # cv.waitKey(0)

    # Optional: Resize/Crop Image especially if large resolution, reduces processing time
    # img_raw = img_raw[0:640, 0:480]
    # img_raw = cv.resize(img_raw, None, fx=1.0, fx=1.0)

    # Convert Images to Grayscale, and perform thresholding for each image
    img_gray = cv.cvtColor(img_raw, cv.COLOR_BGR2GRAY)

    min_thres = 105     # Adjust accordingly
    max_thres = 200     # Adjust accordingly
    _, th2 = cv.threshold(img_gray, min_thres, max_thres, cv.THRESH_BINARY)

    # Find Image Contours
    contours, hierarchy = cv.findContours(th2, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    # Retrieve only contours with a target minimum number of points (initial = 700)
    new_contours = []
    contour_size = 700  # Adjust Accordingly

    for i in range(len(contours)):
        if len(contours[i]) > contour_size:
            new_contours.append(contours[i])

    # Optional: Display remaining contours
    # cv.drawContours(img_raw, new_contours, -1, (0, 255, 0), thickness=2)
    # cv.imshow(im[1][0] + "Contour Result", img_raw)
    # cv.waitKey(0)

    # Create Mask from each image
    img_mask = np.zeros((img_raw.shape[0], img_raw.shape[1]), dtype="uint8")
    obj_mask = cv.drawContours(img_mask, new_contours, -1, 255, cv.FILLED)
    image_masks.append(obj_mask)

    # Optional: Display Mask Process Result
    # cv.imshow(im[1][0] + "Mask Result", obj_mask)
    # cv.waitKey(0)

print(f" >>> Created {len(image_masks)} Object Masks...")     # DEBUG: Display process & target directory

# ---------------------------------- OBJ-4 END ---------------------------------- #


# --------------------------------- OBJECTIVE-5 --------------------------------- #

print(f"\nCREATING DEPTH MAPS FOR {len(image_pairs_undistorted)} IMAGE PAIRS:")     # DEBUG: Display process & target directory

# Download MiDaS
# midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
midas = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid")
# midas = torch.hub.load("intel-isl/MiDaS", "DPT_Large")
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)
midas.eval()

#Input Transformations
transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = transforms.small_transform

output_points = None
output_colors = None

for im, mask in zip(image_pairs_undistorted, image_masks):
    
    img_color = cv.cvtColor(im[0][1], cv.COLOR_BGR2RGB)
    img_batch = transform(img_color).to("cpu")

    with torch.no_grad():
        prediction = midas(img_batch)

        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img_color.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    depth_map = prediction.cpu().numpy()
    img_depth = cv.normalize(depth_map, None, 0, 1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
    # cv.imshow("midas prediction", img_depth)
    # cv.waitKey(0)
    # print("Depth Map Generated!")

# ---------------------------------- OBJ-5 END ---------------------------------- #


# --------------------------------- OBJECTIVE-6 --------------------------------- #

    points_3D = cv.reprojectImageTo3D(img_depth, matrixQ, handleMissingValues=False)
    # print("3D Points Generated!")

    mask_map = mask > 0.4
    output_points = points_3D[mask_map]
    output_colors = img_color[mask_map]
    # print("Depth Map Masked!")
    # output_points = points_3D
    # output_colors = img_color

    img_depth = (img_depth*255).astype(np.uint8)
    img_depth = cv.applyColorMap(img_depth , cv.COLORMAP_MAGMA)
    # print("Depth Map Colorized!")

    output_file = im[0][0].replace(im[0][0][17:], ".ply")
    fig = plt.figure(figsize=(10, 7))
    fig.suptitle(output_file)

    fig.add_subplot(1, 3, 1)
    plt.imshow(im[0][1])
    plt.axis("off")
    plt.title("Color")

    fig.add_subplot(1, 3, 2)
    plt.imshow(img_depth)
    plt.axis("off")
    plt.title("Depth")

    fig.add_subplot(1, 3, 3)
    plt.imshow(mask)
    plt.axis("off")
    plt.title("Mask")
    plt.show()
    
    #Generate point cloud 
    create_pcd(output_points, output_colors, output_file)
    # print("Successful PCD Generation!")

    pcd = o3d.io.read_point_cloud(output_file)
    o3d.visualization.draw_geometries([pcd])

# ---------------------------------- OBJ-6 END ---------------------------------- #




# ADD OBJECTIVE 7, 8 CODE HERE


# ================================== PROGRAM END =================================== #