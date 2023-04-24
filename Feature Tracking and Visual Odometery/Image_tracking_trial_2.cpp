// Reference
//https://docs.opencv.org/4.7.0/d8/d9b/group__features2d__match.html
//https://docs.opencv.org/4.7.0/d9/d97/tutorial_table_of_content_features2d.html
// https://towardsdatascience.com/install-and-configure-opencv-4-2-0-in-windows-10-vc-d132c52063a1
// https://github.com/PointCloudLibrary/pcl/issues/4462
//https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html

// Import Header files
#include <iostream>
#include<set>
#include <chrono>
#include <thread>
#include <math.h>
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/features2d.hpp"
#include <opencv2/opencv.hpp>
#include<fstream>
#include<pcl/visualization/cloud_viewer.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/cloud_viewer.h>

using namespace std;
using namespace cv;


// This function takes in two vectors of 2D points (pts and pts1) and two empty vectors (filter_pts and filter_pts1)
// It calculates the L2 distance between each corresponding pair of points in pts and pts1, and stores the distances in a vector called 'distances'
// It then calculates the mean distance and standard deviation of distances
// Finally, it filters out the points where the distance is within 2 standard deviations of the mean, and stores them in filter_pts and filter_pts1 respectively.
// The filtering criterion used is that the distance should be less than 0.3 times the standard deviation away from the mean.

void distance_filter(vector<Point2f>& pts, vector<Point2f>& pts1, vector<Point2f>& filter_pts, vector<Point2f>& filter_pts1)
{
    // Initialize variables for mean, variance, and standard deviation
    vector<float> distances;
    float distance, mean = 0., var = 0., std;
    int N = int(pts.size());


    // Compute L2 distance between corresponding points and calculate mean distance
    for (int i = 0; i < N; i++)
    {
        float distance = sqrt((pts[i].x - pts1[i].x) * (pts[i].x - pts1[i].x) + (pts[i].y - pts1[i].y) * (pts[i].y - pts1[i].y));
        distances.push_back(distance);
        mean += distance;
    }
    mean /= N;

    // Compute variance and standard deviation of distances
    for (int i = 0; i < N; i++) {
        var += (distances[i] - mean) * (distances[i] - mean);
    }
    var /= N;
    std = sqrt(var);

    // filter points whose distances are within 0.35 times the standard deviation from the mean
    for (int i = 0; i < N; i++) {
        if (((distances[i] - mean) / std) < 0.5) {
            filter_pts.push_back(pts[i]);
            filter_pts1.push_back(pts1[i]);
        }
    }
}

// This function reject outliers using homography and custom RANSAC
void Custom_RANSAC(vector<Point2f>& pts, vector<Point2f>pts1, double threshold)
{
    // Define output inliers
    vector<Point2f> inliers1;
    vector<Point2f> inliers2;

    // Define algorithm parameters
    int num_iterations = 100; // number of RANSAC iterations
    //double threshold = 5.0; // RANSAC threshold distance
    int min_num_inliers = 4; // minimum number of inliers required to compute homography

    // Initialize best homography and number of inliers
    Mat best_homography;
    int best_num_inliers = 0;

    // Run manual RANSAC algorithm
    for (int i = 0; i < num_iterations; i++) {
        // Randomly select four input points
        vector<Point2f> rand_pts;
        vector<Point2f> rand_pts1;
        for (int j = 0; j < 4; j++) {
            int idx = rand() % pts.size();
            rand_pts.push_back(pts[idx]);
            rand_pts1.push_back(pts1[idx]);
        }

        // Compute homography using selected points
        Mat homography = findHomography(rand_pts, rand_pts1);

        // Compute inliers using threshold distance
        vector<Point2f> curr_inliers1;
        vector<Point2f> curr_inliers2;
        for (int j = 0; j < pts.size(); j++) {
            Mat pt = Mat::ones(3, 1, CV_64F);
            pt.at<double>(0, 0) = pts[j].x;
            pt.at<double>(1, 0) = pts[j].y;
            Mat pt1 = Mat::ones(3, 1, CV_64F);
            pt1.at<double>(0, 0) = pts1[j].x;
            pt1.at<double>(1, 0) = pts1[j].y;
            Mat pt1_est = homography * pt;
            pt1_est /= pt1_est.at<double>(2, 0);
            double dist = norm(pt1 - pt1_est);
            if (dist < threshold) {
                curr_inliers1.push_back(pts[j]);
                curr_inliers2.push_back(pts1[j]);
            }
        }

        // Update best homography and number of inliers
        if (curr_inliers1.size() > best_num_inliers && curr_inliers1.size() >= min_num_inliers) {
            best_homography = findHomography(curr_inliers1, curr_inliers2);
            best_num_inliers = curr_inliers1.size();
            inliers1 = curr_inliers1;
            inliers2 = curr_inliers2;
        }
    }

    // Update input points and corresponding points with inliers
    pts = inliers1;
    pts1 = inliers2;
}

// Define a function named track() to do feature tracking
// SIFT/ ORB/ AKAZE feature extraction and FLANN/ Hamming-based descriptor matching. 
// It reads in a sequence of grayscale images, detects and computes SIFT/ ORB/ AKAZE descriptors for each image, 
// matches the descriptors between consecutive images, applies a distance filter to the matches, 
// draws bounding boxes around the filtered keypoints, and writes the output frames to a video file. 
// The function outputs a message indicating that the tracking is complete.

void vo(int a, String data, int N_frames, Mat& K) {

    // Create a SIFT, ORB, AKAZE feature extractor
    // SIFT, AKAZE and ORB is a feature detection and description algorithm used for computer vision tasks
    Ptr<Feature2D> fextractor;

    // Select feature algorithm based on user input preference
    if (a == 1)
        fextractor = SIFT::create();
    else if (a == 2)
        fextractor = ORB::create();
    else
        fextractor = AKAZE::create();
    // Read the first image and compute its features
    // Load an image in grayscale mode
    Mat img = imread(data + "Kitti_Seq_07/" + "000000.png", IMREAD_GRAYSCALE);

    // Detect keypoints and compute SIFT/ ORB/ AKAZE descriptors for the first image
    vector<KeyPoint> kps;
    Mat descriptors;
    fextractor->detectAndCompute(img, Mat(), kps, descriptors);

    // Create a video writer object to save the output video
    // Define the codec, frame rate and size of the output video
    int codec = VideoWriter::fourcc('M', 'J', 'P', 'G');
    VideoWriter video;

    // Select video writer based on user input preference
    if (a == 1)
        video = VideoWriter(data + "/out/Task_1_sift_flann_boxtracking.avi", codec, 24.0, img.size());
    else if (a == 2)
        video = VideoWriter(data + "/out/Task_1_orb_brutforce_Hamming_boxtracking.avi", codec, 24.0, img.size());
    else
        video = VideoWriter(data + "/out/Task_1_akaze_brutforce_Hamming_boxtracking.avi", codec, 24.0, img.size());


    // Define 8-point Algorithm for estimatimating trajectory
    Mat trajectory = Mat::zeros(600, 600, CV_8UC3);
    Mat R, t, pts_3d;
    Mat R_i, t_i;

    String name = "";

    if (a == 1)
        name = "sift_flann";
    else if (a == 2)
        name = "orb_brutforce_Hamming";
    else name = "akaze_brutforce_Hamming";

    VideoWriter video_task3 = VideoWriter(data + "/out/Task_2" + name + "_trajectory.avi", VideoWriter::fourcc('M', 'J', 'P', 'G'), 24.0, trajectory.size());

    ofstream pcout;
    pcout.open(data + "/out/Task_2" + name + "_point_cloud.txt");

    // Loop over the remaining images and track the features
    for (int i = 1; i <= 2; i++)
    {
        // Read the next image and compute its features
        // Load the next image in grayscale mode
        String filename = format("%06d.png", i);
        Mat img1 = imread(data + "Kitti_Seq_07/" + filename, IMREAD_GRAYSCALE);

        // Detect keypoints and compute SIFT, ORB and AKAZE descriptors for the next image
        vector<KeyPoint> kps1;
        Mat descriptors1;
        fextractor->detectAndCompute(img1, Mat(), kps1, descriptors1);

        // Match the features between the current and previous images
        // Use a FLANN-based descriptor matcher to find the best matching features
        vector<DMatch> matches;
        Ptr<DescriptorMatcher> matcher;

        // Select feature descriptor based on user input preference
        if (a == 1)
            matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
        else
            matcher = DescriptorMatcher::create("BruteForce-Hamming");

        // Apply matcher on two image sample
        matcher->match(descriptors, descriptors1, matches, Mat());

        // Filter the matches using distance thresholding
        // Extract the keypoints from the matches and apply a distance-based filter
        vector<Point2f> pts, pts1;
        for (DMatch match : matches)
        {
            pts.push_back(kps[match.queryIdx].pt);
            pts1.push_back(kps1[match.trainIdx].pt);
        }

        // Reject outliers for ORB/ AKAZE brutforce matcher
        // RANSAC for outlier rejection
        Mat ransac_mask;
        Mat F = findFundamentalMat(pts, pts1, FM_RANSAC, 1, 0.99, ransac_mask);
        Mat E = K.t() * F * K;
        vector<Point2f> ransac_pts, ransac_pts1;

        // Just check 1 feature algorithm (SIFT) without RANSAC  
        if (a != 1)
        {
            // Custom_RANSAC(pts, pts1, 5.0);
            for (int j = 0; j < ransac_mask.rows; j++) {
                int mask_val = (int)ransac_mask.at<uchar>(j, 0);
                if (mask_val != 0) {
                    ransac_pts.push_back(pts[j]);
                    ransac_pts1.push_back(pts1[j]);
                }
            }
        }
        else
        {
            ransac_pts = pts;
            ransac_pts1 = pts1;
        }

        // Distance filter based on mean and standard deviation and decide threshold
        // Apply a distance filter based on the mean and standard deviation of the distances
        // Between the keypoints in the current and previous images
        vector<Point2f> filter_pts, filter_pts1;
        distance_filter(ransac_pts, ransac_pts1, filter_pts, filter_pts1);



        // Draw bounding boxes around the tracked features
        // Create a new image to draw the tracked features
        Mat img_out;

        // Convert the grayscale image to RGB color space
        cvtColor(img1, img_out, COLOR_GRAY2BGR);

        // Draw bounding boxes around the tracked keypoints
        for (int i = 0; i < filter_pts.size(); i++)
        {
            // Rect bbox(filter_pts[i].x - 10, filter_pts1[i].y - 10, 5, 5);
            // rectangle(img_out, bbox, Scalar(94, 138, 94), .0000002);

            // Draw a line connecting the keypoints in the current and previous images
            // Uncomment the line below to draw a line between the matched keypoints
            line(img_out, filter_pts[i], filter_pts1[i], Scalar(0, 255, 0), 1, LINE_AA);
        }

        // Write the output frame to the video file
        video.write(img_out);
        // imshow("win0", img_out);
        // waitKey(0);

        // Task:2 Motion estimation --> trajectory
        Mat pose_mask;
        recoverPose(E, filter_pts, filter_pts1, K, R, t, 2000.0, pose_mask, pts_3d);

        if (i == 1) {
            t_i = t.clone();
            R_i = R.clone();
        }
        else {
            t_i = t_i + R_i * t;
            R_i = R * R_i;
        }

        int x = int(t_i.at<double>(0)) + 300;
        int z = int(t_i.at<double>(2)) + 300;
        circle(trajectory, Point(x, z), 1, Scalar(255, 0, 0));
        video_task3.write(trajectory);
        imshow("trajectory", trajectory);
        waitKey(1);

        // Task: 2 3d point cloud output
        for (int j = 0; j < pose_mask.rows; j++) {
            int mask_val = (int)pose_mask.at<uchar>(j, 0);
            if (mask_val != 0) {
                pcout << i << " " << t_i.at<double>(0) << " " << t_i.at<double>(1) << " " << t_i.at<double>(2) << " "
                    << pts_3d.at<double>(0, j) / pts_3d.at<double>(3, j)
                    << " "
                    << pts_3d.at<double>(1, j) / pts_3d.at<double>(3, j)
                    << " "
                    << pts_3d.at<double>(2, j) / pts_3d.at<double>(3, j)
                    << " " << endl;
            }
        }




        // Update the current image and features for the next iteration
        img = img1;
        kps = kps1;
        descriptors = descriptors1;
    }
    pcout.close();

    // Print a message indicating that tracking is complete
    cout << "done" << endl;
}

int main()
{
    int a;
    cout << "Enter 1: SIFT feature with Flann matcher \nEnter 2: ORB feature with Hamming matcher\nEnter 3: AKAZE feature with Hamming matcher" << endl;
    cout << "Enter number: ";
    cin >> a;
    //track(a);

    String data = "H:/Test/data/kitti_Seq/";

    // Camera Intrinsic
    double K_data[] = { 7.070493000000e+02, 0.000000000000e+00, 6.040814000000e+02,
                        0.000000000000e+00, 7.070493000000e+02, 1.805066000000e+02,
                        0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00 };

    // K is a matrix contains focal length and image origin
    Mat K = Mat1d(3, 3, K_data);
    vo(a, data, 1100, K);
    // vo(data, 100, K);
    cout << "All done!" << endl;
}

