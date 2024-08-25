## Description
Photogrammetric Camera Calibration script for Pinhole and Fisheye lens cameras.

## Usage
Place this script in the directory where the calibration images are located. Adjust the parameters according to the physical checkerboard pattern and run the script. The script will create three folders in its directory:
- **Corners:** Contains images with inner corners detected and drawn for each image used in calibration.
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
