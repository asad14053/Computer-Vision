# Computer-Vision
Projects of advanced computer vision and OpenCV
# Project Name: Feature Tracking and Visual Odometery using SIFT, AKAZE and ORB in OpenCV and C++
https://user-images.githubusercontent.com/13154919/233898909-1bc309e4-cb6a-4c12-a7ad-5c3da12ce70a.mp4

## Description
Feature tracking and VO, PCL workflow:

1. Import OpenCV libraries: You'll need to include the necessary OpenCV libraries to your project.
2. Load input images: Load the input images onto which you want to perform the feature tracking. You can use the imread() function of OpenCV for this purpose.
3. Detect features: Use SIFT/ORB/ AKAZE feature detection algorithm to detect features in the input images. You can use the cv::SIFT() or cv::ORB()or, cv::AKAZE() function of OpenCV for this purpose.
4. Match descriptors: Match the descriptors of the detected features in the two input images. We can use any matching algorithm of your choice like brute force matching and FLANN-based matching, etc.
5. Filter matches: Filter out the matches using some criteria to remove false matches. We can use RANSAC algorithm or distance filter or, other filter of your choice.
6. Display output: Display the output by drawing the matched features on the input images using OpenCV functions.
7. Display the output using 8 point algorithm and estimate the trajectory and pointcloud, all are save on ./out folder (you have to create new ./out folder before running it)

## How to run the code:
1. Install openCV 4.7 and VS C++ 2022
2. Import source code and data files
3. Run the code
4. Enter 1 for SIFT based Flann matcher output
   Enter 2 for ORB based Hamming matcher 
   Enter 3 for AKAZE based Hamming matcher output
5. we will see video files based on your input preference
if you entered 1 --> Task_1_sift_flann_boxtracking.avi, Task_2sift_flann_trajectory



.avi, Task_2sift_flann_point_cloud.txt
if you entered 2 --> Task_1_orb_brutforce_Hamming_boxtracking.avi, Task_2orb_brutforce_Hamming_trajectory.avi, Task_2orb_brutforce_Hamming_point_cloud.txt
if you entered 3 --> Task_1_akaze_brutforce_Hamming_boxtracking.avi, Task_2akaze_brutforce_Hamming_trajectory.avi, Task_2akaze_brutforce_Hamming_point_cloud.txt

## The generic details about the codes

1. The distance_filter function takes two vectors of 2D points (pts and pts1) and filters them by computing the Euclidean distance between corresponding points. Points with a distance less than 0.3 times the standard deviation of the distances are kept. This function is used to remove outliers before computing the homography matrix.

2. The rejectOutliers function implements the RANSAC algorithm to estimate the homography matrix between two sets of 2D points (pts and pts1). The RANSAC algorithm selects four random points from the sets and computes the homography matrix using these points. The algorithm then computes the distance between each point in pts and its corresponding point in pts1 using the estimated homography matrix. Points whose distance is less than a given threshold are considered inliers. The algorithm repeats this process for a number of iterations and selects the set of points that produces the largest number of inliers. The function updates the pts and pts1 vectors to contain only the inliers.

3. The track function reads a sequence of images and detects features using the SIFT, AKAZE or ORB feature detectors. It then computes the descriptors of the detected keypoints. For each pair of consecutive images in the sequence, it matches the keypoints using the Brute-Force matcher and filters the matches by computing the distance ratio between the first and second nearest neighbors. It then applies the distance_filter and rejectOutliers functions to remove outliers before computing the homography matrix. Finally, it draws the matches and the homography matrix on the images and displays them.

4. The 8-point algorithm is a fundamental computer vision algorithm used in 3D reconstruction from two or more 2D images. It is used to estimate the essential matrix, which encodes the relative pose of two cameras that have taken images of the same scene.

5. The code assumes that the images are stored in a directory called "data" and that the filenames of the images follow a certain pattern (e.g., "000000.png", "000001.png", etc.). The code also assumes that the first image in the sequence is called "000000.png" and is stored in a directory called "first_200_right"/ "kitti_seq".

##More details about Feature algorithms

1. SIFT, AKAZE, and ORB are feature extraction algorithms commonly used in computer vision and image processing applications. They all aim to identify distinctive features in images that can be used for various tasks such as image matching, object recognition, and tracking.

2. SIFT (Scale-Invariant Feature Transform) algorithm was introduced by David Lowe in 1999. It works by detecting key points in an image at multiple scales and orientations. SIFT features are scale-invariant, which means they can be detected regardless of the size of the object or the image. SIFT uses a Difference of Gaussian (DoG) algorithm to identify local maxima and minima in the image, and then computes the orientation of each key point using gradient information. Finally, it generates a descriptor for each key point based on the distribution of gradient orientations in the surrounding region. SIFT is a robust and widely used algorithm for feature extraction.

3. AKAZE (Accelerated-KAZE) is an extension of the KAZE algorithm, introduced by Pablo F. Alcantarilla et al. in 2012. AKAZE is similar to SIFT in that it detects key points at multiple scales and orientations, but it uses a different approach. Instead of the DoG algorithm, AKAZE uses a non-linear scale space that enhances the detection of corner-like structures in the image. AKAZE also uses a binary descriptor that is faster and more compact than SIFT's descriptor.

4. ORB (Oriented FAST and Rotated BRIEF) is another feature extraction algorithm that was introduced by Ethan Rublee et al. in 2011. ORB is based on two other algorithms: FAST (Features from Accelerated Segment Test) and BRIEF (Binary Robust Independent Elementary Features). FAST is used to detect key points in the image, and BRIEF is used to generate binary descriptors for each key point. ORB adds orientation information to the FAST algorithm to make it rotation-invariant, and also uses a modified version of BRIEF to make it more robust to noise and blur.

In summary, SIFT, AKAZE, and ORB are all feature extraction algorithms that can be used for various computer vision and image processing tasks. SIFT is a robust and widely used algorithm that generates scale-invariant features, while AKAZE is an extension of KAZE that uses a non-linear scale space and a binary descriptor. ORB is based on FAST and BRIEF algorithms, and adds orientation information to make it rotation-invariant. The choice of algorithm depends on the specific application and the requirements for accuracy, speed, and robustness.


**RANSAC (Random Sample Consensus) is a robust method for estimating parameters of a mathematical model from a set of observed data points that may contain outliers. In OpenCV, RANSAC can be used for tasks such as fitting lines or planes to sets of points or finding correspondences between two images.

**FLANN (Fast Library for Approximate Nearest Neighbors) is a library for performing fast nearest neighbor searches in high-dimensional spaces. In OpenCV, FLANN is used for tasks such as image matching and object recognition, where it can be used to find the nearest neighbors of a given feature in a database of features.

**Bruteforce Hamming is a method for matching features in two images based on the Hamming distance between their descriptors. In OpenCV, bruteforce Hamming can be used with binary descriptors such as BRIEF or ORB to find correspondences between two images. It works by computing the Hamming distance between the descriptors of each feature in one image and all the features in the other image, and selecting the closest match based on the distance. This method can be slow for large datasets but is very accurate.

**The 8-point algorithm is a fundamental computer vision algorithm used in 3D reconstruction from two or more 2D images. It is used to estimate the essential matrix, which encodes the relative pose of two cameras that have taken images of the same scene.
