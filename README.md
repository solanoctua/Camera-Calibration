## Description
Photogrammetric Camera Calibration script for Pinhole and Fisheye lens cameras.

## Usage
Place this script in the directory where the calibration images are located. Adjust the parameters according to the physical checkerboard pattern and run the script. The script will create three folders in its directory:
- **Corners:** Script will output the images with inner corners detected and drawed for every image given for calibration.
- **Poses:** If calibration is successful, this folder contains the images (used in calibration) with poses drawn.
- **Undistorted:** If calibration is successful, this folder contains the images (used in calibration) with distortion reversed.

When the calibration is successful, the script will output a `.txt` file with camera calibration results.

## Inputs
- Specify the parameter `CHECKERBOARD = (18,27)` according to the number of inner squares in the checkerboard pattern.
- Specify the parameter `edge_of_square = 0.03` (in meters) according to the size of an edge of one inner square in the checkerboard.
- Specify the width and height of the camera sensor (to calculate focal length in millimeters).

## Outputs
Camera intrinsic matrix and coefficients of the (polynomial) lens distortion model.

### Intrinsic Matrix Example:
```
intrinsic matrix =  [f_x  , 0    , c_x ]
                   [0    , f_y  , c_y ]
                   [0    , 0    , 1   ]

focal lengths   : fx, fy
aspect ratio    : a = fy/fx
principal point : cx, cy

distCoeffs = [k1,k2,p1,p2,k3,k4,k5,k6,s1,s2,s3,s4,taux,tauy]

radial distortion     : k1, k2, k3
tangential distortion : p1, p2
rational distortion   : k4, k5, k6
thin prism distortion : s1, s2, s3, s4
tilted distortion     : taux, tauy (ignored here)
```

### Example Output:
```
Found 34 valid images for calibration 
Mean re-projection error: 0.2386850192324747 px
Resolution: (960, 720)
K = np.array([[658.8019713561096, 0.0, 490.53904897795724], [0.0, 658.425293474817, 353.18201098329763], [0.0, 0.0, 1.0]])
D = np.array([[-0.0037400202963716375], [-0.017349074746381112], [0.05038129356367704], [-0.04298264284373254]])
```

## Errors
The script may produce an error (terminating the calibration process) or just a warning (without terminating the process) if some images are not suitable (ill-conditioned) for calibration. 

For example:
- If corners are not detected, the script will print `image_name.png is not suitable for calibration` (calibration will proceed without these images).
- If corners are detected but the checkerboard is too close to the image frame edges, an error will indicate the problematic image.

## Accuracy
- Ensure sufficient lighting during calibration to increase the accuracy of corner detection in the checkerboard.
- If possible, print the checkerboard on retroreflective material to eliminate light reflections that prevent corner detection.
- Avoid using images where the checkerboard is tilted more than 45 degrees relative to the camera, as corners may not be detected due to foreshortening.
- To accurately model lens distortion, try to cover the entire image frame by moving the checkerboard to the edges (but not too close to the frame edges).
- Check the `allcorners.png` image to verify full frame coverage; if not, collect new images for uncovered areas.
- By reviewing the re-projection error per image and the mean re-projection error, identify any images with significantly higher errors, isolate them, and exclude them from calibration.
- After discarding some images, if some parts of the frame are not covered, collect new images for those areas.
- Generally, if the re-projection error is under 1 pixel, calibration is successful.
- To ensure accuracy, visually inspect the undistorted output images and check if poses are drawn correctly. If not, this may indicate incomplete frame coverage or a high re-projection error.

## Warning
This procedure is not suitable for those with insufficient knowledge of camera calibration science. 

- When the lens is twisted or replaced, the distortion model becomes invalid.
- Camera calibration is specific to the camera's resolution. If the resolution changes, the intrinsic parameters cannot be used directly and must be scaled properly, or a new calibration with the desired resolution is needed.

```
fx = focal length(mm) ∗ frame width(px) / sensor width(mm)
fy = focal length(mm) ∗ frame height(px) / sensor height(mm)
```

## DETAILED GUIDE

#### Importance of Camera Calibration

A camera captures the 3D world and converts it into 2D pixel coordinates. However, the images produced often do not accurately represent the actual scene, due to the inherent characteristics of camera lenses or errors during the assembly or production of camera components. Even if the captured images seem fine, understanding the abstract camera model is crucial for tasks like image processing and computer vision applications. To achieve this, one must know the intrinsic camera parameters and the lens distortion model to bridge the gap between the physical camera and its abstract representation.

Ignoring lens distortions or intrinsic camera parameters during processes such as distance estimation, 3D object pose and size extraction, or homography can lead to significant errors or render such tasks unfeasible. Therefore, camera calibration is a critical step in 3D computer vision to accurately extract metric information from 2D images.

In calibration, we aim to estimate the intrinsic and extrinsic parameters of the camera model, which describe how a point in the 3D world is mapped to a corresponding point on the image plane through perspective projection. Distortion parameters, which can be estimated during calibration, are considered part of the intrinsic parameters of the camera.

#### Calibration Goal

The goal of the calibration process is to determine the 3×3 intrinsic camera matrix, the 3×3 rotation matrix, and the 3×1 translation vector using a set of known 3D world points and their corresponding pixel coordinates. Once these intrinsic and extrinsic parameters are obtained, the camera is considered calibrated.

To project a 3D object point onto the image plane, the point must first be transformed from the world coordinate system to the camera coordinate system using the extrinsic parameters. Then, using the intrinsic parameters of the camera, the point can be projected onto the image plane.

- **Intrinsic (Internal) Parameters**: These include the focal length, optical center, and lens distortion coefficients.
- **Extrinsic (External) Parameters**: These refer to the orientation (rotation and translation) of the camera relative to a world coordinate system, which could be aligned with a calibration pattern such as a checkerboard.

#### Photogrammetric Calibration Procedure

##### Choosing a Calibration Pattern

Calibration is performed by observing a calibration object with known 3D geometry. The most commonly used calibration objects are checkerboards, circle grids, fiducial markers, and custom detector patterns. A checkerboard pattern, which consists of alternating black and white squares of equal size, is commonly used. The corners of these squares serve as control points and can be automatically detected in 2D images using a corner detection algorithm. Since the squares are of equal size, their 3D world coordinates can be determined by knowing the square size of the checkerboard.

We use checkerboard patterns because they are distinct, easy to detect, and their corners provide ideal localization points due to sharp gradients in two directions. Assuming the X (right) and Y (down) axes are along the pattern plane and the Z-axis is perpendicular to it, all points on the checkerboard lie on the XY plane (Z = 0). The (X, Y) coordinates of each 3D point can be easily defined by selecting one reference point (0, 0) and defining the remaining corners relative to it.

##### Preparing the Checkerboard

When creating the checkerboard pattern, consider the following:

1. **Pattern Size**: The pattern should be large enough to cover the entire camera field of view in a few passes. This can be determined by the camera’s field of view at certain distances. For example, a camera with given sensor sizes and an approximate focal length will have a 2.85-meter horizontal and 2.15-meter vertical field of view at a 2-meter distance. Using an A3 paper (297 x 420 mm) checkerboard pattern, you can cover the camera's view in approximately 35 passes from a 2-meter distance.

2. **Unit Square Size**: For a camera with the above parameters, a 3.3-centimeter (0.033 meters) square appears as nearly 15 pixels at a 2-meter distance. This size is adequate in good lighting conditions without blurring. To ensure successful images, consider increasing the unit square size.

3. **Number of Squares**: Since the overall checkerboard size and the unit square size are decided, simply add as many squares as possible to cover the printing area. It is preferred to use a checkerboard with an even number of squares along one edge and an odd number along the other, with two black corner squares on one side and two white corner squares on the opposite side. This orientation helps certain calibration algorithms determine the pattern's orientation and origin, with the longer side typically assigned as the x-direction.

You may use the following sites to create a checkerboard suitable for your camera and application:
- [Ninox Calibration Targets](https://www.ninox360.com/calibration-targets)
- [Calib.io Camera Calibration Pattern Generator](https://calib.io/pages/camera-calibration-pattern-generator)

We use a 7x10 checkerboard (6x9 inner corners) with squares having 55 mm edges.

#### Camera Model

A camera model describes the mathematical relationship between a point in the 3D world and its projection onto the 2D image plane.

Before starting the calibration script, decide which camera model to use. For cameras with a field of view between 100 and 180 degrees, the fisheye model provides a more accurate distortion model and precise focal length calculation. For angles under 90 degrees, the pinhole model is more accurate; however, it becomes less effective for wider angles. The pinhole model cannot accurately calibrate cameras with wide or fisheye lenses, as it cannot project a hemispherical field of view onto a finite image plane via perspective projection. Therefore, choosing the correct camera model is crucial.

#### Starting Calibration

There are several methods to perform calibration. One common approach is to fix the camera and change the checkerboard's position and orientation while capturing images or video.

Set up the camera and a screen so the person moving the checkerboard can see if it stays within the camera's view and is visible enough for calibration. Move the pattern across the entire camera frame while tilting and rotating it in all three directions.

The accuracy of calibration depends heavily on the selection of camera poses from which images of the calibration object are acquired. For satisfactory calibration, ensure that the target successively covers the entire image area, or else the estimation of radial distortion and other parameters may remain suboptimal.

#### Calibration Script Execution and Error Checking

After collecting sufficient images, start the calibration script. If some checkerboard corners are not detected in certain images, those images will be automatically discarded.

Once the script completes, proceed with the following checks, unless an error occurs, in which case refer to the Troubleshooting section:

1. **Frame Coverage**: Ensure that most of the camera frame is covered by the calibration images. Manually check the `allcorners.png` file saved by the script to see whether the entire frame is covered by the detected corners.

2. **Undistorted Images**: Immediately check if the undistorted images look correct. Verify that straight lines in real life appear straight in the undistorted images. If they do, the calibration is likely successful.

If most of the frame is covered but the undistorted images are unsatisfactory, some corners may not have been correctly detected, or the data may be suboptimal. Manually inspect the images with drawn corners to identify and remove problematic ones from the calibration set. After removing these images, repeat the calibration to achieve better results.

Additionally, check the individual re-projection errors of the images. If an image’s re-projection error is significantly higher than the mean, it may be an outlier. This could be due to concentrating too many images in one part of the frame. For example, if 10 images have the checkerboard at the center, then the 11th image with the checkerboard near the corners might become an outlier, disrupting the model constructed by the other images. However, if one image is an outlier, this indicates that the distortion model is not representing some part of the camera frame well. This should be detected by judging the straightness of lines in the undistorted images. Another possibility is an issue with the image itself, which should be detected during manual corner inspection.

If all drawn corners are fine but the undistorted images still appear incorrect, the problem is likely not with corner detection.

Once the calibration process is complete, the distortion can be modeled using the `VisualizePinholeDistortionModel.m` script.

#### Improving Calibration Accuracy

Here are practical considerations to improve calibration accuracy:

1. **Calibration Pattern (Checkerboard)**:
   - Print the checkerboard on retroreflective material, if available, to eliminate light reflections that can hinder corner detection. Laminating the pattern may cause glare, leading to incorrect corner detection.
   - Mount the target on a rigid, flat surface. Any warping will reduce calibration accuracy.
   - Ensure white margins around the checkerboard pattern, as these are crucial for corner detection algorithms in MATLAB and OpenCV. Without these borders, the algorithms may fail to detect any corner points.

2. **Motion Blur**:
   - Motion blur occurs when the target moves too quickly during exposure, causing the object’s position to vary between the start and end of exposure. This blur can cause corner detection to fail.
   - Since the camera is fixed in our setup, avoid touching it during image acquisition to prevent camera movement-induced blur. Use a tripod or other support to keep the camera steady.
   - If recording a video, hold the checkerboard still for 2-3 seconds before capturing frames. Use only the frames without motion blur.
   - If motion blur persists, the corner detection algorithm may deviate from the actual corners. Manually inspect images with drawn corners to detect any deviations.

3.

 **Frame Coverage**:
   - Ensure that the entire camera frame is covered by calibration images. This is particularly important for cameras with significant lens distortion and wide fields of view.
   - Manually verify frame coverage by inspecting the `allcorners.png` file saved by the calibration script.

4. **Lighting**:
   - Ensure even illumination across the scene and avoid overexposure.
   - Ensure that the checkerboard is sufficiently visible with good contrast. Without good contrast, corner detection may fail.

5. **Recording Multiple Checkerboard Passes**:
   - It is not necessary to calibrate the camera with only one set of images. Collect additional checkerboard pass images if necessary.
