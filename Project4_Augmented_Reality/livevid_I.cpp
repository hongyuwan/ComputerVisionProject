#include <iostream>
#include <filesystem>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <dirent.h>
#include <opencv2/opencv.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include "calibration.h"

/*
  Course: Computer Vision - 5330 S22
  Project 4: Calibration and Augmented Reality
  Name: Sida Zhang and Hongyu Wan
  March 10, 2022

  This file finds the checkerboard and do calibration on the checkerboard.
  This is the first part of the project that writes data to txt file
    in order to run AR system on the second part of the project.

*/
using namespace cv;

int main(int argc, char *argv[])
{ 
  // remove all calibration images in data/CameraCalibration
  char dirname[256];
  char buffer[256];
  DIR *dir_path;
  struct dirent *dp;
  strcpy(dirname, "C:\\Users\\sidaz\\Desktop\\5330ComputerVision\\Projects\\project4\\data\\CameraCalibration");

  dir_path = opendir( dirname );
  while( (dp = readdir(dir_path)) != NULL ) {
    strcpy(buffer, dirname);
    strcat(buffer, "\\"); // changed from "/" to "\\"
    strcat(buffer, dp->d_name);
    std::remove(buffer);
  }

  cv::VideoCapture *capdev;

  // open the video device
  capdev = new cv::VideoCapture(0);
  if( !capdev->isOpened() ) {
    printf("Unable to open video device\n");
    return(-1);
  }

  // get some properties of the image
  cv::Size refS( (int) capdev->get(cv::CAP_PROP_FRAME_WIDTH ),
                  (int) capdev->get(cv::CAP_PROP_FRAME_HEIGHT));

  cv::namedWindow("Calibration Window", 1); // identifies a window
  cv::Mat frame;
  cv::Size chessboardPattern(9,6);

  std::vector<cv::Point2f> corner_set;
  // std::vector<cv::Vec3f> point_set;
	std::vector<std::vector<cv::Vec3f> > point_list;    // vector for points lists for each calibration frame.
	std::vector<std::vector<cv::Point2f> > corner_list; // vector for corner lists for each calibration frame.
  int num = 0; //saving calibration frames;

  double camera_matrix3X3[3][3] = {
        {1, 0, frame.cols/2.0},
        {0, 1, frame.rows/2.0},
        {0, 0, 1}
  };
  cv::Mat camera_matrix(3, 3, CV_64FC1, camera_matrix3X3);
  cv::Mat disCoefficients = cv::Mat::zeros(8, 1, CV_64F);
  std::vector<cv::Mat> rotationVec, translationVec;

  for(;;) {
    *capdev >> frame;
    // User manual text shown in bottom
    std::string text = "s or S to save | c or C to calibrate | q or Q to quit";
    cv::putText(frame, text, cv::Point(20, refS.height - 20), cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(0, 0, 255), 1, cv::LINE_8, false);

    // Task 1: Detect and Extract Chessboard Corners
    // Speed up the performance by giving CALIB_CB_FAST_CHECK flag
    detectDrawChessboardCorners(frame, chessboardPattern, corner_set);
    
    int key = cv::waitKey(10);
    if (key == 83 || key == 115) {          // "s" or "S" to save image
      // Task 2: Select Calibration Images
      // check if corner_set is empty
      if(!corner_set.empty()){
        selectCalibrationImages(frame, chessboardPattern, corner_set, point_list, corner_list);
        // Save images to data/CameraCalibration
        std::string file = "C:\\Users\\sidaz\\Desktop\\5330ComputerVision\\Projects\\project4\\data\\CameraCalibration\\clibration_" + std::to_string(num) + ".jpg";
        cv::imwrite(file, frame);
        std::cout << "Calibration images are successfully saved to data/CameraCalibration/clibration_" << std::to_string(num) << ".jpg" << std::endl;
        num++;
      } else {
        std::cout << "Calibration not found, please try again" << std::endl;
      }

    } else if (key == 67 || key == 99){     //  "c" or "C" to start calibrate
      // Task 3: Calibrate the Camera
      calibrateFrame(frame, point_list, corner_list, camera_matrix,
               disCoefficients, rotationVec, translationVec, num);
      // write camera_matrix and distoration_ceofficients
      writeIntrinsicParameters(camera_matrix, disCoefficients);
    } else if(key == 81 || key == 113){     // "q" or "Q" or "ESC" to terminate the program
      cv::destroyWindow("Calibration Window");
      break;
    }
  }
  
  printf("\nTerminating\n");
  delete capdev;
  return(0);
}