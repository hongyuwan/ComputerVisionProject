/*
  Course: Computer Vision - 5330 S22
  Project 3: Real-time Object 2-D Recognition
  Name: Sida Zhang and Hongyu Wan
  Febuary 26, 2022

  This header file declears all functions for recognition.cpp.
*/

#ifndef RECOGNITION_H
#define RECOGNITION_H

using namespace cv;

// the two functions threashold an image to a binary image by the thresh level
// it also darken a pixel if the pixel has high saturation.
// thresholding for Mat input
// the first one threshold objects from camera
// the second one threshold objects from database
int thresholding(cv::Mat &src, int thresh, cv::Mat &dst);
int thresholding_d(char *imageFile, int thresh, cv::Mat &dst);

// clean up by growing and shrinking
int cleanup(cv::Mat &src, cv::Mat &dst);

// Compute Connected Component with Segment Region Filter
// and return the biggest 3 objects from camera.
int ccAnalysis(cv::Mat &src, cv::Mat &dst, double &num);
// Compute Connected Component with Segment Region Filter
// and return the biggest 1 object from the dataset.
int ccAnalysis_d(cv::Mat &src, cv::Mat &dst);
// draw connected component on video stream;
int drawfeatures(cv::Mat &src, cv::Mat &dst, double &num);

// feature compute for video stream and
// draw centroid, bounding box, and axis.
int featureCompute(cv::Mat &src, cv::Mat &dst);
// store feature data from the camera to a vector
int featureCompute_c(cv::Mat &src, cv::Mat &dst, std::vector<std::pair<std::string,std::vector<float>>> &featureData);
// store feature data from the dataset to a vector
int featureCompute_d(cv::Mat &src, std::string &objectLabel, cv::Mat &dst, std::vector<std::pair<std::string,std::vector<float>>> &featureData);

// collect data to database, csv file.
int collectData(std::vector<std::pair<std::string,std::vector<float>>> &featureData);
// calculating distance metric by taking objects' feature and compare with features in database.
int distanceMetric(cv::Mat &src, std::string &output, double &num);
// calculate kNearestNeighbor matching by taking object's features and comepare with features in database by its frequency.
int kNearestNeighbor(cv::Mat &src, std::string &output, double &num);

#endif