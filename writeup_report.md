## Writeup Template

**Advanced Lane Finding Project**

The different steps I used in this project are the following:

1. First we compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
1. Then, we apply the distortion correction to raw images.
1. We look for different color transforms, gradients, etc., and combine them to create a thresholded binary image.
1. We then apply a perspective transform to rectify binary image ("birds-eye view").
1. In that birds-eye view we look lane pixels using histogram and fit to find the lane boundary.
1. Then we determine the curvature of the lane and vehicle position with respect to center.
1. Finally we warp the detected lane boundaries back onto the original image.
1. We output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position. We save the outputs in the output_images folder.

[//]: # (Image References)

[chessboard_undistortion]: ./output_images/chessboard_undist.png "Chessboard undist"
[test_image_undistortion]: ./output_images/test_image_undist.png "Test image undist"
[combined_binary_image]: ./output_images/straight_lines2_undistorted_binary.png "Binary Example"
[image4]: ./output_images/straight_lines2_undistorted_binary_warped.png "Warp Example"
[image5]: ./output_images/straight_lines2_undistorted_binary_warped_lane_fit.png "Fit Visual"
[image6]: ./output_images/straight_lines2_final_output.png "Output"
[video1]: ./project_video.mp4 "Video"

### Here I will consider all [rubric points](https://review.udacity.com/#!/rubrics/571/view) individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  

The current file is the Writeup report.
Note : All code was written in one IPython notebook : **P2-advanced_lane_finding_project.ipynb**.

### Camera Calibration

#### 1. Calculation of camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the following cells:

* 1/ Get distortion coefficients and camera matrix from camera calibration using chessboard
* 2/ Apply distortion to raw images to get rectified images.

I calibrated the camera using all chessboard images located in the *camera_cal/* folder. Two methods where used to fill the `objpoints`, 3D points in real world space, and `imgpoints`, 2D points in image plane, lists :

* convert image to Grayscale  using *cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)*
* find the chessboard on the image using *cv2.findChessboardCorners*

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the chessboard image using the `cv2.undistort()` function. The following result displays the original chessboard image on the left and the undistorted one on the right:

![alt text][chessboard_undistortion]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

Using the camera calibration matrix and its distortion coefficients I applyed the correction using `cv2.undistort()` to one of the test images. The following result displays the original image on the left and the undistorted one on the right:
![alt text][test_image_undistortion]

#### 2. Image filtering using combined thresholding methods

The code for this step is contained in the following cells:

* 3/ Combine thresholds techniques to create binay image
* 3.5/ Combine thresholds

I used a combination of :

* Color space threshold : using the S channel of HLS image.
* Sobel X gradiant threshold
* Magnitude gradiant threshold

to generate a binary image.  Here's an example of my output for this step.

![alt text][combined_binary_image]

#### 3. Perspective transform

The code for this step is contained in the following cells:

* 4/ Apply perspective transform

The code for my perspective transform includes a function called `warper()`:
```python
def warper(img, src, dst):
    M = cv2.getPerspectiveTransform(src, dst) 
    inv_M = cv2.getPerspectiveTransform(dst, src)   
    
    # Grab the image shape
    img_size = (img.shape[1], img.shape[0])  
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
    return warped, M, inv_M
```

 The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
src = np.float32([[300, 700], [1300, 700], [600, 450], [700, 450]])

dst = np.float32([[300, 700], [1180, 700], [300, 0], [1180, 0]]) 
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 300, 700      | 300, 700        | 
| 1300, 700      | 1180, 700      |
| 600, 450     | 300, 0      |
| 700, 450      | 1180, 0        |

I verified that my perspective transform was working as expected verifying that the lines appear parallel in the warped image. Here is the result :

![alt text][image4]

#### 4. Lane-line pixels detection and line polynomial fit

The code for this step is contained in the following cells:

* 5/ Detect lane pixels using Histogram and get lane boundary

Here are the different steps followed to do that using the previous warped binary image:

* Get the bottom half of the image to compute the histogram. 
* Using the histogram find the left and right lane X base by spliting it vertically in two.
* Divide the image in 9 sliding windows.
* From the bottom to top window we identify on left and right lane non zero pixels. We save the indices of those pixels as being the indices for both left and right lanes. Depending on the number of found pixels we recenter the window by updating the left and right lane X base.
* Once pixel from left and right lane identified we fit a 2nd polynomial the estimate the left and right lane lines.

The python method used to do it is : `find_lane_pixels()`.

In prevision of using this algorithm on real world stream another step was implemented to find lane-line pixels. When in a stream we can use previously found lines and just add an horizontal +/- margin to perform again a polynomial fit. This improves a lot the search as it will not go through the complete process.

The python method used to do it is : `search_around_poly()`

The result is the following :

![alt text][image5]

#### 5. Calculation of the radius of curvature of the lane and the position of the vehicle with respect to center.

The code for this step is contained in the following cells:

* 6/ Compute lane curvature and vehicle position to center

Once we computed the polynomial fit for both left and right lanes we can easily compute the radius of curvature using the formulas presented in the courses. The python method used is `calc_curvature_real_world()`. We directly compute the mean_curve (average of both left and right curvature) in real world units : meters. To do that I use the meter per pixel conversion in both X and Y axis :

* 30 meters for 720 pixels on Y vertical axis.
* 3.7 meters for 700 pixels on X horizontal axis.

The vehicle position/offset to center of lane is computed directly using the polynomial fit for both left and right lanes. As specified in the tips and tricks of this project, when assuming that the camera is mounted at the center of the car, meaning that the lane center is the midpoint at the bottom of the image, the vehicle offset is the offset of the lane center from from the center of the image. The python method used is `calc_offset_from_center()`.

Both average curvature and vehicle position to lane center are added to the image.

#### 6. Final result

The code for this step is contained in the following cells:

* 7/ Warp the lane boudaries back onto the original image

Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Link to video pipeline

Here's a [link to my video result](./test_videos_output/project_video_output.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

This version works only on simple images. It is obsvious that any big contrast, shadows, car and probably even simple scratches on the road, will cause some failures.

Something that could easily be done to *robustify* this detection is a bit of intelligent filtering on the lane detection, a low pass filter could help ! Also checking that the lines are parallel and respecting a mean distance between them is obviously the first improvment that needs to be done.

My biggest problem here was to implement every thing in the python notebook. It was fine when working on images. But when starting to think about handling the stream it was a bit complicated to get it all working fine. Next projects will be segmented cleanly in different files to avoid those issues.