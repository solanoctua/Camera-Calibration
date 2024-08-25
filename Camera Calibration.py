#**************************************************************************
# Description : Photogrammetric Camera Calibration script for Pinhole and Fisheye lens cameras. 
# Version     : V1.0 
# Usage       : Put this script where the calibration images located, adjust the parameters according to physical checkerboard pattern , and run the script
#               Script will create 3 folders to where it is located
#               -Corners: Script will output the images with inner corners detected and drawed for every image given for calibration.
#               -Poses: If calibration is done, script will output the images (used in calibration) with poses drawed.
#               -Undistorted: If calibration is done, script will output the images (used in calibration) with distortion is reversed.
#                 and when the calibration is successfull, the script will output an .txt file including camera calibration results.
#
# Inputs      : -User must specify the parameter CHECKERBOARD = (18,27) according to how many inner squares the used checkerboard pattern have 
#               -User must specify the parameter edge_of_square = 0.03 in meters according to size an edge of the one inner square in the used checkerboard
#               -User must specify the width and height information of the sensor of the camera (to be able to calculate focal lenght in milimeters)
# Outputs     : Camera intrinsic matrix and coefficients of the (polynomial) lens distortion model 
#               
#               intrinsic matrix =  [f_x  , 0    , c_x ]
#                                   [0    , f_y  , c_y ]
#                                   [0    , 0    , 1   ]
#               Example output:
#                   Found 34 valid images for calibration 
#                   Mean re-projection error: 0.2386850192324747 px
#                   Resolution: (960, 720)
#                   K = np.array([[658.8019713561096, 0.0, 490.53904897795724], [0.0, 658.425293474817, 353.18201098329763], [0.0, 0.0, 1.0]])
#                   D = np.array([[-0.0037400202963716375], [-0.017349074746381112], [0.05038129356367704], [-0.04298264284373254]])
#
# Errors      : The script will give an error (ending the calibration process) or just warning (without ending the calibration process) 
#                if some of the images aqre not suitable (Ill Conditioned) for the calibration.
#               For example:
#                - If corners are not detected, script will print image_name.png is not suitable for calibration (calibration will be done excluding these images).
#                - If corners are detected but the checker board is too near the edges of the image frame (in this case error will point out the erroneous image).        
#                
# Accuracy    : -Do this calibration with sufficient lightning to increase the accuracy of the detection of the corners in checkerboard.
#               -Print checkerboard in retroreflective material if available to eliminate light reflections which prevents corner detection.
#               -Do not use the pictures where checkerboard is tilted more than 45 degrees with respect to the camera,
#                 otherwise checkerboard corners may not be detected due to foreshortening.
#               -To model the lens distortion well, try to cover all of the image frame by moving the checkerboard to the sides (also not too near to the frame edges)
#               -Check allcorners.png image to see if all the frame is covered or not, if not covered collect new images for those areas.
#               -By looking at the re-projection error per image and the mean re-projection error, 
#                 decide whether if some images have much more re-projection error compared to other ones, 
#                 then isolate these images and simply do not use them in the calibration.
#               -After discarting some images, if some parts of the frame is not covered, collect new images for those areas.
#               -Generally if the re-projection error is under 1 pixel, calibration is successfull.
#               -To be sure, check undistorted output images with an eye if undistortion is correctly done, and poses whether if they are correctly drawn,
#                 if not probably not all frame is covered by checkerboard images, or re-projection error is too high.
#                
# Warning     :  This procedure is not suited for one with insufficient information about the science of camera calibration.
#                When lens is twisted or replaced with another, distortion model is no more.
#                Camera calibration is specific to the resolution of the camera. When the resolution is changed, intrinsic parameters can not be used directly, 
#                they must be scaled properly, or a new calibration with desired resolution is needed.
#                fx = focal length(mm) ∗ frame width(px) / sensor width(mm)
#                fy = focal length(mm) ∗ frame height(px) / sensor height(mm)
#**************************************************************************

# Useful Links
# https://docs.opencv.org/4.5.4/dc/dbb/tutorial_py_calibration.html
# https://www.baeldung.com/cs/correcting-fisheye-images
# https://wiki.panotools.org/Fisheye_Projection
# https://docs.nvidia.com/vpi/sample_fisheye.html

import datetime
import glob
import os
import shutil 
import cv2
import matplotlib.pyplot as plt
import numpy as np


print("cv2.__version__: ", cv2.__version__)
current_path = os.getcwdb()
print("Current Path: ",current_path)

#*********************************************************************************
camera_model = "fisheye"
#camera_model = "pinhole"
CHECKERBOARD = (6,9)
edge_of_square = 0.055 # in meters  30mm = 0.03 m
sensor_width = 5.7 # in mm
sensor_height = 4.3 # in mm
#*********************************************************************************

subpix_criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_CHECK_COND + cv2.fisheye.CALIB_FIX_SKEW

print("\nCreating output folders..")
# CREATE SAVE PATHS IF NOT ALREADY EXISTS
if os.path.exists(f"./Corners {camera_model}"): 
    shutil.rmtree(f"./Corners {camera_model}", ignore_errors=False, onerror=None)
    print(f"old ./Corners {camera_model} file deleted")
    os.mkdir(f"./Corners {camera_model}")
    print(f"./Corners {camera_model} file created")
else:
    os.mkdir(f"./Corners {camera_model}")
    print(f"./Corners {camera_model} file created")

if os.path.exists(f"./Poses {camera_model}"): 
    shutil.rmtree(f"./Poses {camera_model}", ignore_errors=False, onerror=None)
    print(f"old ./Poses {camera_model} file deleted")
    os.mkdir(f"./Poses {camera_model}")
    print(f"./Poses {camera_model} file created")
else:
    os.mkdir(f"./Poses {camera_model}")
    print(f"./Poses {camera_model} file created")
    
if os.path.exists(f"./Undistorted {camera_model}"): 
    shutil.rmtree(f"./Undistorted {camera_model}", ignore_errors=False, onerror=None)
    print(f"old ./Undistorted {camera_model} file deleted")
    os.mkdir(f"./Undistorted {camera_model}")
    print(f"./Undistorted {camera_model} file created")
else:
    os.mkdir(f"./Undistorted {camera_model}")
    print(f"./Undistorted {camera_model} file created")
    
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(9,6,0) for the 9x6 checkerboard  
objp = np.zeros((1, CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[1], 0:CHECKERBOARD[0]].T.reshape(-1, 2) # create mxn matrix ,which each entry will(currently all 0) contain (x,y,z) coordinates of corners of chessboard in real(3D) world    
objp = objp * edge_of_square  # to find location of corners in real world we multiply the matrix with edge_of_square.
    
_img_shape = None
good_img_names = []
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
image_corners = []
images = glob.glob('*.png')
success_count = 0
print("\nFinding chessboard corners..")
for fname in images:
    img = cv2.imread(fname)

    if _img_shape == None:
        _img_shape = img.shape[:2]
    else:
        assert _img_shape == img.shape[:2], "All images must share the same size."
   
    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(img, (CHECKERBOARD[1],CHECKERBOARD[0]), cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK) # cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE
    image_corners.append(corners)
    if ret == True:
        good_img_names.append(fname)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        if success_count == 0:
            frame_cover_check = np.ones(gray.shape,  np.float32) * 255
            
        print(f"image index: {len(objpoints)} image name: {fname}, corners found: {ret}")
        # If found, add object points, image points (after refining them)
        objpoints.append(objp)
        # Find sub-pixel accurate location of the corners
        cv2.cornerSubPix(gray,
                         corners,
                         (3,3), #(3,3)
                         (-1,-1),
                         subpix_criteria)

        imgpoints.append(corners)
        cv2.drawChessboardCorners(img, (CHECKERBOARD[1],CHECKERBOARD[0]), corners, ret)
        # Draw all detected corners for every individual checkerboard, to later check if all camera frame is covered by these corners.
        # To precisely model the distortion of the camera, all camera frame should be covered by these corners.
        cv2.drawChessboardCorners(frame_cover_check, (CHECKERBOARD[1],CHECKERBOARD[0]), corners, ret)
        success_count += 1
    else:
        print(f"{fname} is not suitable for calibration")
    
    splited = fname.split("/")
    cv2.imwrite(f"./Corners {camera_model}/"+splited[-1] +"_corners.png",img)
    cv2.imshow('img', img)
    
    key = cv2.waitKey(20)
    if key == 27:
        cv2.destroyAllWindows()
    
if success_count == 0:
    print("None of the images are suitable for calibration..")
    print("Session terminated.")
    exit()

cv2.destroyAllWindows()
cv2.imwrite(f"./Corners {camera_model}/allcorners.png", frame_cover_check)

N_OK = len(objpoints)
K = np.zeros((3, 3))

rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]    
tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]

# Calibrate the camera
try:
    print("\nCalibrating..")
    print("Camera model: ", camera_model)
    if camera_model == "fisheye":
        D = np.zeros((4, 1))
        # https://github.com/opencv/opencv/blob/4.x/modules/calib3d/src/fisheye.cpp#L1558
        rms, K, D, rvecs, tvecs = cv2.fisheye.calibrate(objpoints,
                                                        imgpoints,
                                                        gray.shape[::-1],
                                                        K,
                                                        D,
                                                        rvecs,
                                                        tvecs,
                                                        calibration_flags,
                                                        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6))
    if camera_model == "pinhole":
        D = np.zeros((5, 1))
        rms, K, D, rvecs, tvecs = cv2.calibrateCamera(objpoints, 
                                                      imgpoints, 
                                                      gray.shape[::-1],
                                                      K,
                                                      D,
                                                      rvecs,
                                                      tvecs,
                                                      None, #cv2.CALIB_FIX_PRINCIPAL_POINT
                                                      (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)) # 
except Exception as error:
    print("Error: ",error)
    
date = datetime.datetime.now()  # We will use this date for all of the file namings
with open(f"{date.day}.{date.month}.{date.year}_{date.hour}.{date.minute}.{date.second}_calibration_output_{camera_model}.txt", "w") as file:
    np.savetxt(file, K, delimiter=',', header="self.mtx: ",)
    np.savetxt(file, D, delimiter=',', header="self.dist: ",)
    file.write(f"\nK: {K}\n")
    file.write(f"D: {D}\n")
    file.close()
    print("Calibration output saved.")

axis_length = edge_of_square * 3

print("Undistorting images and calculating poses..")

i = 0
axes3D = np.float32([[1,0,0],[0,1,0],[0,0,-1]]).reshape(-1,3) * axis_length
for fname in images:
    
    distorted_img = cv2.imread(fname)
    ret, corners = cv2.findChessboardCorners(distorted_img, (CHECKERBOARD[1],CHECKERBOARD[0]), cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK)#cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE
    if ret:
        ret, rvec, tvec = cv2.solvePnP(objp, imgpoints[i], K, D, flags = cv2.SOLVEPNP_ITERATIVE) 
        pose = distorted_img.copy()
        
        # Draw number of each corner
        """
        j = 0
        for point in imgpoints[i]:
            print("point: ", (int(point[0][0]), int(point[0][1])))
            cv2.circle(pose, (int(point[0][0]), int(point[0][1])), 3, (0,0,255),1)
            cv2.putText(pose, str(j), (int(point[0][0]), int(point[0][1])-4), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (255, 0, 255), 1, cv2.LINE_AA)
            j+= 1
        """
        
        # Draw axes
        """
        axes2D, jacobian = cv2.projectPoints(axes3D, rvecs_2, tvecs_2, K, D)
        corner_chessboard = tuple(imgpoints[i][0].ravel())
        x_axis_end_point = tuple(axes2D[0].ravel())
        y_axis_end_point = tuple(axes2D[1].ravel())
        z_axis_end_point = tuple(axes2D[2].ravel())

        pose = cv2.line(pose, (int(corner_chessboard[0]), int(corner_chessboard[1])), (int(x_axis_end_point[0]), int(x_axis_end_point[1])), (255,0,0), 5)
        pose = cv2.line(pose, (int(corner_chessboard[0]), int(corner_chessboard[1])), (int(y_axis_end_point[0]), int(y_axis_end_point[1])), (0,255,0), 5)
        pose = cv2.line(pose, (int(corner_chessboard[0]), int(corner_chessboard[1])), (int(z_axis_end_point[0]), int(z_axis_end_point[1])), (0,0,255), 5)
        cv2.imwrite("Poses/"+fname+"_pose.png", pose)
        """
    
        # Undistort images
        if camera_model == "fisheye":
            if i == 0:
                new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D, gray.shape[::-1], np.eye(3), balance=1) # balance=0 for cropped, balance=1 for normal
                mapx, mapy = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), new_K, gray.shape[::-1], cv2.CV_16SC2)
            undistorted = cv2.remap(distorted_img, mapx, mapy, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            #undistorted = cv2.fisheye.undistortImage (distorted_img, K, D, K)
        if camera_model == "pinhole":
            if i == 0:
                new_K, roi = cv2.getOptimalNewCameraMatrix(K, D, gray.shape[::-1], alpha = 1, newImgSize = gray.shape[::-1])
                #print("roi: \n", roi)
                x,y,w,h = roi
                mapx, mapy = cv2.initUndistortRectifyMap(K, D, np.eye(3), new_K, gray.shape[::-1], 5) # 5--> CV_32F C1
                 
            undistorted = cv2.remap(distorted_img, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            cv2.rectangle(undistorted, (x,y), (x + w, y + h), (0,0,255), 1)
            #undistorted = cv2.undistort(distorted_img, K, D, None, new_K)
            #undistorted  = undistorted[y:y+h, x:x+w] # for pinhole
            
        
        # Project 3D cube points to 2D image
        if camera_model == "pinhole":
            cube3D = np.float32([[0,0,0], [0,1,0], [1,1,0], [1,0,0], [0,0,-1], [0,1,-1], [1,1,-1], [1,0,-1]]).reshape(-1,3) * axis_length
            cube2D, jacobian = cv2.projectPoints(cube3D, rvec, tvec, K, D)
        if camera_model == "fisheye":
            cube3D = np.float32([[[0,0,0], [0,1,0], [1,1,0], [1,0,0], [0,0,-1], [0,1,-1], [1,1,-1], [1,0,-1]]]) * axis_length
            cube2D, jacobian = cv2.fisheye.projectPoints(cube3D, rvec, tvec, K, D, alpha = 0)
     
        cube2D = np.int32(cube2D).reshape(-1,2)
        # Draw cube
        # draw ground floor in green
        pose = cv2.drawContours(pose, [cube2D[:4]], -1, (0,255,0), -3)
        # draw pillars in blue color
        for k,l in zip(range(4),range(4,8)):
            cube_point_1 = tuple(cube2D[k])
            #print("cube_point_1: ", cube_point_1)
            cube_point_2 = tuple(cube2D[l])
            pose = cv2.line(pose, cube_point_1, cube_point_2, (255), 2)
        # draw top layer in red color
        pose = cv2.drawContours(pose, [cube2D[4:]], -1, (0,0,255), 2)
        
        i+= 1
        cv2.imwrite(f"Undistorted {camera_model}/"+fname+"_undistorted.png", undistorted)
        cv2.imwrite(f"Poses {camera_model}/"+fname+"_pose.png", pose)
        cv2.imshow("Pose", pose)
        key = cv2.waitKey(20)
        if key == 27:
            cv2.destroyAllWindows()
            break

# CALCULATE REPROJECTION ERRORS
print("Calculating reprojection errors..")
reprojection_errors = []
total_error = 0
with open(f"{date.day}.{date.month}.{date.year}_{date.hour}.{date.minute}.{date.second}_calibration_output_{camera_model}.txt", "a") as file:
    file.write("\n\nFound " + str(N_OK) + " valid images for calibration\n")
    file.write(f"Resolution: {str(_img_shape[::-1])}\n")
    file.write(f"RMS: {rms} px\n")
    print("Calculating re-projection error per image..")
    print(f"Re-projecting object points w.r.t. {camera_model} camera model..")
    for i in range(len(objpoints)):
        
        if camera_model == "fisheye":
            # https://github.com/opencv/opencv/blob/4.x/modules/calib3d/src/fisheye.cpp#L1558
            projection_points, _ = cv2.fisheye.projectPoints(objectPoints = objpoints[i], rvec = rvecs[i], tvec = tvecs[i], K = K, D = D, alpha = 0)
        if camera_model == "pinhole":
            projection_points, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, D)
                
        projection_points = projection_points.reshape(imgpoints[i].shape)
        
        error_1 = np.sum(np.abs(imgpoints[i] - projection_points)**2) 
        #print(f"\nerror_1: {error_1}")
        error_2 = error_1 / len(projection_points)
        #print(f"error_2: {error_2}")
        error_3 = np.sqrt(error_2)
        #print(f"error_3: {error_3}")
        total_error += error_1
       
        reprojection_errors.append(error_3)
        file.write(f"{i}th image ({good_img_names[i]}) re-projection error: {error_3} px\n")
            
    mean_error = np.sqrt(total_error/(len(objpoints)*len(projection_points)))
    print("mean_error: ",mean_error)
    
    print("\nRESULTS:")
    print("_______________________________________________________________________________________________________________________________")
    print("Found " + str(N_OK) + " valid images for calibration")
    print(f"Mean re-projection error: {rms} px ")
    print("Resolution: " + str(_img_shape[::-1]))
    print("K = np.array(" + str(K.tolist()) + ")")
    print(f"K{K.shape}:\n{K}\n")
    f_x = K[0,0] * sensor_width / gray.shape[::-1][0]
    f_y = K[1,1] * sensor_height / gray.shape[::-1][1]
    #f_z = np.sqrt(f_x^2 + f_y^2)
    print(f"f_x: {f_x} mm")
    print(f"f_y: {f_y} mm")
    #print(f"f_z: {f_z}\n")
    print("D = np.array(" + str(D.tolist()) + ")")
    print("_______________________________________________________________________________________________________________________________")
    
    file.close()

# Plot relative positions of checkerboard patterns and the camera

fig = plt.figure(figsize=plt.figaspect(2.))#, facecolor="gray"
fig.set_size_inches(8, 10, forward=True)
ax = fig.add_subplot(2, 1, 1)
ax.set_xlabel("Image Number")
ax.set_ylabel("Re-Projection Error (px)")
ax.set_title("Re-Projection Error per Image")
ax.axhline(y=rms, color="red", linestyle='-')
ax.grid(color="black", linestyle="--", linewidth=1, axis="y", alpha=1)
ax.bar(list(range(0,len(reprojection_errors))), reprojection_errors, color = "darkviolet")
ax.set_xticks(np.arange(0, len(reprojection_errors), step=1)) 
ax.xaxis.set_tick_params(labelsize=5)

ax2 = fig.add_subplot(2, 1, 2, projection='3d')

ax2.set_title("POSES")
ax2.set_xlabel("$X$", fontsize=20, rotation=0)
ax2.set_ylabel("$Y$", fontsize=20, rotation=0)
ax2.set_zlabel("$Z$", fontsize=20, rotation=0)

for i in range(len(objpoints)):
    obj_points_in_camera_coord_system = []
    point_no = 0
    rotation_matrix, jacobian = cv2.Rodrigues(rvecs[i])
    
    for j in range(len(objpoints[i][0])):
        point_no += 1
        
        homogeneous = np.array([objpoints[i][0][j]])
        camera_frame_coordinate = rotation_matrix @ homogeneous.T + tvecs[i]
        flat = (camera_frame_coordinate * np.array([[1, -1, -1]]).T).flatten()
        obj_points_in_camera_coord_system.append(flat)
        
    obj_points_in_camera_coord_system = np.array(obj_points_in_camera_coord_system)
    ax2.plot_trisurf(obj_points_in_camera_coord_system[:,0],obj_points_in_camera_coord_system[:,1], obj_points_in_camera_coord_system[:,2], alpha=0.2, color = np.random.rand(3,))
    ax2.text(obj_points_in_camera_coord_system[-1][0], obj_points_in_camera_coord_system[-1][1], obj_points_in_camera_coord_system[-1][2], "{}".format(i), fontsize=10, fontweight="bold",  color = "blue")

plt.show()
print("\nSession terminated.")