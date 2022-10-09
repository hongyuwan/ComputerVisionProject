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
  
  This file first read the camera matrix and dist coeffs in order to 
    find the rotation and translation vectors.
    We have used flags for different tasks including A|V|I|E|H
    The instruction is also printed in ever frame in live video stream.
*/

using namespace cv;

int main(int argc, char *argv[])
{ 
    // Task 4: Calculate Current Position of the Camera
    char camera_matrix_data[256];
    char dist_coeffs_data[256];
    int intervals = 0;            // used for showing results in frames
    int flag = 0;                 // used for draw 3d or objects.
    strcpy(camera_matrix_data, "C:\\Users\\sidaz\\Desktop\\5330ComputerVision\\Projects\\project4\\data\\camera_matrix.txt");
    strcpy(dist_coeffs_data, "C:\\Users\\sidaz\\Desktop\\5330ComputerVision\\Projects\\project4\\data\\dist_coeffs.txt");


    cv::Mat camera_matrix(3, 3, CV_64FC1);
    cv::Mat disCoefficients = cv::Mat::zeros(8, 1, CV_64F);

    // Task 4 Part I: Reading Intrinstic Parameters
    readIntrinsicParameters(camera_matrix_data, dist_coeffs_data, camera_matrix, disCoefficients);

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

    cv::namedWindow("AR system", 1); // identifies a window
    cv::Mat frame;
    cv::Size chessboardPattern(9,6);

    // Extension IIII: preloaded image
    char staticImageDir[256];
    char staticImage[256];
    DIR *dir_path;
    struct dirent *dp;
    strcpy(staticImageDir, "C:\\Users\\sidaz\\Desktop\\5330computerVision\\Projects\\project4\\data\\StaticImages");
    
    dir_path = opendir( staticImageDir );

    // it will automatically check if there is any picture in data/StaticImages folder.
    // If yes, it will run cv::findChessboardCorners to find corners; if a chessboard is found, it will show the after effect in a new window.
    if( (dp = readdir(dir_path)) == NULL ){
      std::cout << "No pre-loaded static image found in data/StaticImages" <<  std::endl;
    } else {
      std::cout << "Pre-loaded static image found, please check the AR effects" << std::endl;
    }
    while( (dp = readdir(dir_path)) != NULL ) {
      if( strstr(dp->d_name, ".jpg") || strstr(dp->d_name, ".png") || strstr(dp->d_name, ".ppm") || strstr(dp->d_name, ".tif") ) {
        strcpy(staticImage, staticImageDir);
        strcat(staticImage, "\\");
        strcat(staticImage, dp->d_name);

        readStaticImageARsystem(staticImage, camera_matrix, disCoefficients);
      }
    }


    // start detecting rotation and translation from the saved files in data folder
    std::cout << "Detecting Rotation&Translation Data on frames..." << std::endl;
    for(;;){
      *capdev >> frame;
      // User manual text shown in every frame in bottom
      // https://www.geeksforgeeks.org/python-opencv-cv2-puttext-method/
      std::string text, text2, text3;
      if(flag == 0){
        text = "a or A to draw 3D axes | v or V to draw a Diamond ";
        text2 = "i or I to draw an Icosahedron | e or E to draw an Aircraft";
        text3 = "h or H to draw Harris Corner | q or Q to quit";
      } else if(flag == 1){
        text = "";
        text2 = "";
        text3 = "ESC to return to the menu";
      } else if(flag == 2){
        text = "";
        text2 = "*Hard-coded data.";
        text3 = "ESC to return to the menu";
      } else if(flag == 3){
        text = "";
        text2 = "*Hard-coded data.";
        text3 = "ESC to return to the menu";
      } else if(flag == 4){
        text = "";
        text2 = "*Loaded from .obj file";
        text3 = "ESC to return to the menu";
      } else if(flag == 5){
        text = "";
        text2 = "";
        text3 = "ESC to return to the menu";
      }
      cv::putText(frame, text, cv::Point(20, refS.height - 60), cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(0, 0, 255), 1, cv::LINE_8, false);
      cv::putText(frame, text2, cv::Point(20, refS.height - 40), cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(0, 0, 255), 1, cv::LINE_8, false);
      cv::putText(frame, text3, cv::Point(20, refS.height - 20), cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(0, 0, 255), 1, cv::LINE_8, false);


      std::vector<cv::Point2f> corner_set;
      std::vector<cv::Point3f> point_set;
      for(int i = 0; i < chessboardPattern.height; i++){
          for (int j = 0; j < chessboardPattern.width; j++){
              point_set.push_back(cv::Vec3f(j, -i, 0));
          }
      }

      cv::Mat rotationVec, translationVec;

      // CALIB_CB flags provide much faster speed when searching for chessboard
      // https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html
      bool found = cv::findChessboardCorners(frame, chessboardPattern, corner_set, 
                  cv::CALIB_CB_FAST_CHECK);

      if(found){
        // Task 4 Part II: getting chessboard poses
        cv::solvePnP(point_set, corner_set, camera_matrix, disCoefficients, rotationVec, translationVec);
        if(intervals%25 == 0){
          
        // Print Rotation Data and Translation Data every 25 frames in real time
          std::cout << "Rotation: \t";
          std::cout << rotationVec.at<double>(0) << "\t" << rotationVec.at<double>(1) << "\t"  << rotationVec.at<double>(2) << std::endl;
          std::cout << "Translation: \t";
          std::cout << translationVec.at<double>(0) << "\t"  << translationVec.at<double>(1) << "\t"  << translationVec.at<double>(2) << std::endl;
        }
        
        // Task 5: Project Outside Corners or 3D Axes
        if(flag == 1){
          // Draw Axes on top left outside corner
          axes3D(frame, rotationVec, translationVec, camera_matrix, disCoefficients);
        }
        // Task 6: Create a Virtual Object
        if(flag == 2){
          // Draw a Diamond
          createVirtualObj(frame, rotationVec, translationVec, camera_matrix, disCoefficients);
        }
        if(flag == 3){
          // Draw an Icosahedron
          int scale = 5;
          createVirtualObj2(frame, rotationVec, translationVec, camera_matrix, disCoefficients, scale);
        }
        if(flag == 4){
          // Draw Cool stuffs
          
          // 1 as image 0 as video;
          int arg = 0;
          createVirtualObjExtension(frame, rotationVec, translationVec, camera_matrix, disCoefficients, arg);
        }
      }
    
      // Task 7: Detect Robust Features: Harris Corner
      if(flag == 5){
        harrisCorner(frame);
      }

      
      cv::imshow("AR system", frame);
      intervals++;

      
      int key = cv::waitKey(10);
      if(key == 65 || key == 97){             // press "a" or "A" to draw 3d axes on frames.
        flag = 1;
      } else if(key == 86 || key == 118) {    // press "v" or "V" to draw a dimond.
        // Diamond
        flag = 2;
      } else if(key == 73 || key == 105){     // press "i" or "I" to draw an Icosahedron.
        // Icosahedron
        flag = 3;
      } else if(key == 69 || key == 101) {    // press "e" or "E" to draw an Aircraft.
        // Aircraft
        flag = 4;
      } else if(key == 72 || key == 104) {    // press "h" or "H" to show Harris Corners.
        // Harris Corners
        flag = 5;
      } else if (key == 27){                  // press "ECS" to clean objects on frames.
        flag = 0;
      } else if(key == 81 || key == 113){     // press "q" or "Q" or "ESC" to terminate the program
        cv::destroyWindow("Live Video");
        break;
      }
    }
    
    printf("\nTerminating\n");
    delete capdev;
    return(0);
}