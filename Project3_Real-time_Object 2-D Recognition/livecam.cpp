#include <iostream>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <dirent.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <cmath>
#include <iomanip>
#include "csv_util.h"
#include "recognition.h"
// One (and only one) of your C++ files must define CVUI_IMPLEMENTATION
// before the inclusion of cvui.h to ensure its implementaiton is compiled.
#define CVUI_IMPLEMENTATION
#include "cvui.h"
using namespace cv;
#define WINDOW_NAME "Frame"
/*
  Course: Computer Vision - 5330 S22
  Project 3: Real-time Object 2-D Recognition
  Name: Sida Zhang and Hongyu Wan
  Febuary 26, 2022

  This file contains contains everything for GUI and Outputs
  The main function contains every buttons and output for matching
  User can also adjust the Thresh value on the GUI, default set to 60
*/

int main(int argc, char *argv[])
{
  char dirname[256]; // image dataset location
  char buffer[256];
  char csv[256];
  char print[256];
  FILE *fp; // csv file location
  DIR *dirp;
  struct dirent *dp;
  int i;
  // !set the directory path to default.
  strcpy(dirname, "C:\\Users\\sidaz\\Desktop\\5330computerVision\\Projects\\project3\\data\\trainingSet");
  // printf("Processing directory %s\n", dirname );

  // empty csv;
  fp = fopen("C:\\Users\\sidaz\\Desktop\\5330computerVision\\Projects\\project3\\data\\objectFeatures.csv", "a");

  // open the directory
  dirp = opendir(dirname);
  // if directory unavailable
  if (dirp == NULL)
  {
    printf("Cannot open directory %s\n", dirname);
    exit(-1);
  }
  char begin[256] = "Program has started.";
  strcpy(print, begin);
  printf("Program has started\n");

  cv::Mat video, provideo, showconnect;
  cv::Mat dst, morpho, connectedComponent, dst1;
  std::vector<std::pair<std::string, std::vector<float>>> featureData;
  cv::Mat threshold = cv::Mat::zeros(dst.rows, dst.cols, CV_8UC1);
  double thresh = 60;
  int mode = 1;
  int photoid = 1;
  bool collect = 0;
  bool Matching = 0;
  int showmatch = 0;
  double value = 60;

  std::string disMatch, knnMatch;
  cv::VideoCapture *capdev;
  char c[256];
  // open the video device
  capdev = new cv::VideoCapture(0);
  if (!capdev->isOpened())
  {
    printf("Unable to open video device\n");
    return (-1);
  }
  // get some properties of the image
  cv::Size refS((int)capdev->get(cv::CAP_PROP_FRAME_WIDTH),
                (int)capdev->get(cv::CAP_PROP_FRAME_HEIGHT));
  cv::Mat frame = cv::Mat(700, 1200, CV_8UC3);

  // Init cvui and tell it to create a OpenCV window, i.e. cv::namedWindow(WINDOW_NAME).
  cvui::init(WINDOW_NAME);
  for (;;)
  {
    // button
    // camera video
    if (cvui::button(frame, 850, 20, "Camera"))
    {
      mode = 1;
    }
    // Task1&&2: Thresholding
    if (cvui::button(frame, 850, 60, "Task 1 && 2: Thresholding and Clean up the binary image"))
    {
      mode = 2;
    }
    // Task3: Segment the image into regions
    if (cvui::button(frame, 850, 100, "Task 3: Segment the image into regions"))
    {
      mode = 3;
    }
    // Task4: Compute features
    if (cvui::button(frame, 850, 140, "Task 4: Compute features"))
    {
      mode = 4;
    }
    // Save Image
    if (cvui::button(frame, 850, 180, "Save Image (take 3 images for a single object)."))
    {
      if (mode < 5)
      {
        char save_image[256], photo_name[256], temp[64], temp1[64];
        char outPutPath[256] = "C:\\Users\\sidaz\\Desktop\\5330computerVision\\Projects\\project3\\data\\newImage\\";
        std::cin >> temp1;
        sprintf(temp, "%d", mode);
        char b[20] = "mode";
        char c[20] = ".jpg";
        char d[20] = "\\";
        strcat(outPutPath, b);
        strcat(outPutPath, temp);
        mkdir(outPutPath);
        strcat(outPutPath, d);
        strcat(outPutPath, temp1);
        strcat(outPutPath, c);
        if (mode == 1)
          imwrite(outPutPath, video);
        if (mode == 2)
          imwrite(outPutPath, morpho);
        if (mode == 3)
          imwrite(outPutPath, showconnect);
        if (mode == 4)
          imwrite(outPutPath, dst1);
      }
    }
    // Task5: Collect training data from Video
    if (cvui::button(frame, 850, 220, "Task 5:(1) Collect training data from Video"))
    { // save an image of current Video
      if (mode == 4)
      {
        {
          char save_image[256], photo_name[256], temp[64], temp1[64];
          char outPutPath[256] = "C:\\Users\\sidaz\\Desktop\\5330computerVision\\Projects\\project3\\data\\newImage\\";
          std::cout << "Please input the image name." << std::endl;
          std::cin >> temp1;
          char b[20] = "new_dataset";
          char c[20] = ".jpg";
          char d[20] = "\\";
          strcat(outPutPath, b);
          mkdir(outPutPath);
          strcat(outPutPath, d);
          strcat(outPutPath, temp1);
          strcat(outPutPath, c);
          imwrite(outPutPath, video);
        }
        { // print
          collect = 1;
          char a[256] = "Collect data from Video finished.";
          strcpy(print, a);
        }
      }
    }
    // Task5: Collect training data from dataset
    if (cvui::button(frame, 850, 260, "Task 5:(2) Collect training data from dataset"))
    {

      int counter = 0;
      std::string objectLabel;
      int idx = 1;
      char a[256] = "Collect data from dataset finished.";
      strcpy(print, a);

      while ((dp = readdir(dirp)) != NULL)
      {
        // check if the file is an image
        if (strstr(dp->d_name, ".jpg") || strstr(dp->d_name, ".png") || strstr(dp->d_name, ".ppm") || strstr(dp->d_name, ".tif") ||
            strstr(dp->d_name, ".JPG") || strstr(dp->d_name, ".PNG") || strstr(dp->d_name, ".PPM") || strstr(dp->d_name, ".TIF"))
        {
          // build the overall filename
          strcpy(buffer, dirname);
          strcat(buffer, "\\"); // changed from "/" to "\\"
          strcat(buffer, dp->d_name);

          if (counter == 0)
          {
            std::cout << "Please label object " << idx << " from the dataset." << std::endl;
            std::cin >> objectLabel;
          }
          counter++;
          if (counter == 3)
          {
            counter = 0;
            idx++;
          }

          cv::Mat threshold1, morpho1, connectedComponent1, moment1;
          std::vector<std::pair<std::string, std::vector<float>>> featureData1;
          // cv::Mat src = cv::imread(buffer);
          // task1
          thresholding_d(buffer, thresh, threshold1);
          // task2
          cleanup(threshold1, morpho1);
          // task3
          ccAnalysis_d(morpho1, connectedComponent1);
          // task4
          featureCompute_d(connectedComponent1, objectLabel, moment1, featureData1);
          // task5
          collectData(featureData1);
        }
      }
    }
    // clear training data
    if (cvui::button(frame, 850, 300, "Clear database"))
    {
      fp = fopen("C:\\Users\\sidaz\\Desktop\\5330computerVision\\Projects\\project3\\data\\objectFeatures.csv", "w");
    }
    // task 6
    if (cvui::button(frame, 850, 340, "Task 6: Euclidean Distance Metric Classifier"))
    {
      showmatch = 1;
      distanceMetric(video, disMatch, value);
      char a[256] = "Euclidean Distance Metric Classifier finished.";
      strcpy(print, a);
    }
    // task 7
    if (cvui::button(frame, 850, 380, "Task 7: K-Nearest Neighbor Classifier"))
    {
      showmatch = 2;
      kNearestNeighbor(video, knnMatch, value);
      char a[256] = "K-Nearest Neighbor Classifier finished.";
      strcpy(print, a);
    }
    // Segment Region Filter control
    cvui::window(frame, 140, 10, 240, 70, "Thresh Value");
    {
      cvui::trackbar(frame, 140, 30, 220, &thresh, (double)32.0, (double)128.0);
    }

    *capdev >> video; // get a new frame from the camera, treat as a stream

    // updated gui
    cvui::update();
    // show camera
    if (mode == 1)
    { 
      cvui::image(frame, 100, 100, video);
    }
    // show threshold and clean up
    if (mode == 2)
    { 
      thresholding(video, thresh, threshold);
      cleanup(threshold, morpho);
      cv::imshow("video", morpho);
    }
    // show connectedComponent
    if (mode == 3)
    { 
      thresholding(video, thresh, threshold);
      cleanup(threshold, morpho);
      drawfeatures(morpho, showconnect, value);
      cvui::image(frame, 100, 100, showconnect);
    }
    // show featureCompute
    if (mode == 4)
    { 
      thresholding(video, thresh, threshold);
      cleanup(threshold, morpho);
      ccAnalysis(morpho, connectedComponent, value);

      if (collect == 0)
      {
        featureCompute(connectedComponent, dst1);
      }
      else
      {
        featureCompute_c(connectedComponent, dst1, featureData);
        collectData(featureData);
        collect = 0;
      }

      cvui::image(frame, 100, 100, dst1);
    }

    // Output window
    cvui::window(frame, 780, 460, 450, 100, "Output");
    {
      if (showmatch == 1)
        cvui::printf(frame, 800, 485, 0.4, 0x00ff00, "The Euclidean's best match for this object is %s.", disMatch.c_str());
      if (showmatch == 2)
        cvui::printf(frame, 800, 525, 0.4, 0x00ff00, "The Knn's best match for this object is %s.", knnMatch.c_str());
    }

    // Terminal window
    cvui::window(frame, 780, 600, 450, 50, "Terminal");
    {
      cvui::printf(frame, 800, 625, 0.5, 0x00ff00, print);
    }

    cvui::imshow(WINDOW_NAME, frame);

    if (cv::waitKey(20) == 27)
    {

      break;
    }
  }

  delete capdev;
  return (0);
}