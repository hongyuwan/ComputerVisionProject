#ifndef FILTERS_H
#define FILTERS_H

#include <iostream>
#include <opencv2/opencv.hpp>
#include <typeinfo>

using namespace cv;

int greyscale( cv::Mat &src, cv::Mat &dst );
int blur5x5( cv::Mat &src, cv::Mat &dst );
int sobelX3x3( cv::Mat &src, cv::Mat &dst );
int sobelY3x3( cv::Mat &src, cv::Mat &dst );
int magnitude( cv::Mat &sx, cv::Mat &sy, cv::Mat &src, cv::Mat &dst);
int blurQuantize( cv::Mat &src, cv::Mat &dst, int levels );
int cartoon(cv::Mat &src, cv::Mat&dst, int levels, int magThreshold );
int brightnessAdjust(cv::Mat &src, cv::Mat&dst, int upOrDown, int adjVal);
int blueLineChallenge(cv::Mat &src, cv::Mat &src2, cv::Mat&dst);
int readMe();

#endif