# Camera-Calibration

Description : Photogrammetric Camera Calibration script for Pinhole and Fisheye lens cameras. 
Usage       : Put this script where the calibration images located, adjust the parameters according to physical checkerboard pattern , and run the script
               Script will create 3 folders to where it is located
               -Corners: Script will output the images with inner corners detected and drawed for every image given for calibration.
               -Poses: If calibration is done, script will output the images (used in calibration) with poses drawed.
               -Undistorted: If calibration is done, script will output the images (used in calibration) with distortion is reversed.
                 and when the calibration is successfull, the script will output an .txt file including camera calibration results.

Inputs      : -User must specify the parameter CHECKERBOARD = (18,27) according to how many inner squares the used checkerboard pattern have 
               -User must specify the parameter edge_of_square = 0.03 in meters according to size an edge of the one inner square in the used checkerboard
               -User must specify the width and height information of the sensor of the camera (to be able to calculate focal lenght in milimeters)

Outputs     : Camera intrinsic matrix and coefficients of the (polynomial) lens distortion model 

               intrinsic matrix =  [f_x  , 0    , c_x ]
                                   [0    , f_y  , c_y ]
                                   [0    , 0    , 1   ]
               Example output:
                   Found 34 valid images for calibration 
                   Mean re-projection error: 0.2386850192324747 px
                   Resolution: (960, 720)
                   K = np.array([[658.8019713561096, 0.0, 490.53904897795724], [0.0, 658.425293474817, 353.18201098329763], [0.0, 0.0, 1.0]])
                   D = np.array([[-0.0037400202963716375], [-0.017349074746381112], [0.05038129356367704], [-0.04298264284373254]])

Errors      : The script will give an error (ending the calibration process) or just warning (without ending the calibration process) 
              if some of the images aqre not suitable (Ill Conditioned) for the calibration.
               For example:
                - If corners are not detected, script will print image_name.png is not suitable for calibration (calibration will be done excluding these images)  
                - If corners are detected but the checker board is too near the edges of the image frame (in this case error will point out the erroneous image)           
                
Accuracy    : -Do this calibration with sufficient lightning to increase the accuracy of the detection of the corners in checkerboard.
               -Print checkerboard in retroreflective material if available to eliminate light reflections which prevents corner detection.
               -Do not use the pictures where checkerboard is tilted more than 45 degrees with respect to the camera,
                 otherwise checkerboard corners may not be detected due to foreshortening.
               -To model the lens distortion well, try to cover all of the image frame by moving the checkerboard to the sides (also not too near to the frame edges)
               -Check allcorners.png image to see if all the frame is covered or not, if not covered collect new images for those areas.
               -By looking at the re-projection error per image and the mean re-projection error, 
                 decide whether if some images have much more re-projection error compared to other ones, 
                 then isolate these images and simply do not use them in the calibration.
               -After discarting some images, if some parts of the frame is not covered, collect new images for those areas.
               -Generally if the re-projection error is under 1 pixel, calibration is successfull.
               -To be sure, check undistorted output images with an eye if undistortion is correctly done, and poses whether if they are correctly drawn,
                 if not probably not all frame is covered by checkerboard images, or re-projection error is too high.
                
Warning     :  This procedure is not suited for one with insufficient information about the science of camera calibration.
                When lens is twisted or replaced with another, distortion model is no more
                Camera calibration is specific to the resolution of the camera. When the resolution is changed, intrinsic parameters can not be used directly, 
                they must be scaled properly, or a new calibration with desired resolution is needed.
                fx = focal length(mm) ∗ frame width(px) / sensor width(mm)
                fy = focal length(mm) ∗ frame height(px) / sensor height(mm)
