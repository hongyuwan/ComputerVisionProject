#ifndef CALIBRATION_H
#define CALIBRATION_H

#include <iostream>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <dirent.h>
#include <opencv2/opencv.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <string>

using namespace cv;

int detectDrawChessboardCorners(cv::Mat &src, cv::Size &chessboardPattern,  std::vector<cv::Point2f> &corner_set);
int selectCalibrationImages(cv::Mat &src, cv::Size &chessboardPattern,
                    std::vector<cv::Point2f> &corner_set,
                    std::vector<std::vector<cv::Vec3f> > &point_list,
                    std::vector<std::vector<cv::Point2f> > &corner_list );
int calibrateFrame(cv::Mat &src, std::vector<std::vector<cv::Vec3f> > &point_list,
                std::vector<std::vector<cv::Point2f> > &corner_list,
                cv::Mat &camera_matrix, cv::Mat &disCoefficients,
                std::vector<cv::Mat> &rotationVec, std::vector<cv::Mat> &translationVec, int &num);
int writeIntrinsicParameters(cv::Mat &camera_matrix, cv::Mat &disCoefficients);
int readIntrinsicParameters(char* matrix_file, char* dist_file, cv::Mat &camera_matrix, cv::Mat &disCoefficients);
int axes3D(cv::Mat &src, cv::Mat &rotationVec, cv::Mat &translationVec, cv::Mat &camera_matrix, cv::Mat &disCoefficients);
int createVirtualObj(cv::Mat &src, cv::Mat &rotationVec, cv::Mat &translationVec, cv::Mat &camera_matrix, cv::Mat &disCoefficients);
int createVirtualObj2(cv::Mat &src, cv::Mat &rotationVec, cv::Mat &translationVec, cv::Mat &camera_matrix, cv::Mat &disCoefficients, int &scale);
int harrisCorner(cv::Mat &src);
int loadObj(char *path, std::vector<cv::Vec3f> &vertices, std::vector<int> &faces);
int createVirtualObjExtension(cv::Mat &src, cv::Mat &rotationVec, cv::Mat &translationVec, cv::Mat &camera_matrix, cv::Mat &disCoefficients, int &arg);
int readStaticImageARsystem(char* staticImageme, cv::Mat &camera_matrix, cv::Mat &disCoefficients)

#endif