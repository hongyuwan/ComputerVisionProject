#include <iostream>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <dirent.h>
#include <opencv2/opencv.hpp>
#include "csv_util.h"
#include "matching.h"
#include "filter.h" //filter for Sobel Magnitude calculation from project 1.
/*
  Course: Computer Vision - 5330 S22
  Project 2: Content-based Image Retrieval
  Name: Sida Zhang and Hongyu Wan
  Febuary 7, 2022
  
  This file iterate all images and calculate its histogram
  or feature and store them to individual csv files to the
  data folder. And run each tasks and extensions to perform
  distance metric matching.

*/


/*
  Given a directory on the command line, scans through the directory for image files.
 */
int main(int argc, char *argv[]) {
  char dirname[256];
  char buffer[256];
  char csv[256]; //csv file location
  FILE *fp;
  DIR *dirp;
  struct dirent *dp;
  int i;

  // !set the directory path to default.
  strcpy(dirname, "C:\\Users\\sidaz\\Desktop\\5330computerVision\\Projects\\project2\\data\\olympus");
  // printf("Processing directory %s\n", dirname );

  // open the directory
  dirp = opendir( dirname );
  if( dirp == NULL) {
    printf("Cannot open directory %s\n", dirname);
    exit(-1);
  }
  
  printf("Program has started\n");

  // loop over all the files in the image file listing
  while( (dp = readdir(dirp)) != NULL ) {
    // check if the file is an image
    if( strstr(dp->d_name, ".jpg") || strstr(dp->d_name, ".png") || strstr(dp->d_name, ".ppm") || strstr(dp->d_name, ".tif") ) {
      // build the overall filename
      strcpy(buffer, dirname);
      strcat(buffer, "\\"); // changed from "/" to "\\"
      strcat(buffer, dp->d_name);


      // initial images' data
      std::vector<float> baselineData, histData, multiData, sobelDataT4, rgbDataT4, sobelDataT5, rgbDataT5, gaborDataEx1, rgbDataEx1, HsvDataEx2, RgbDataEx2;
      //iterate images to 9x9 baseline vector to float.
      getBaseline(buffer, baselineData);
      // //iterate images to rg chromaticity histogram to vector float.
      getHistMatch(buffer, histData);
      // //iterate images to two RGB histograms to vector float.
      getMultiMatch(buffer, multiData);
      // //iterate images to one color and one textrue histograms to vector float.
      getRGBsobel(buffer, sobelDataT4, rgbDataT4);
      //iterate images to rg chromaticity histogram and a texture histogram to vector float.
      getRgSobel(buffer, sobelDataT5, rgbDataT5);
      //Extension 01: use gabor feature and RGB color histograms to re-do matching task 4.
      getGaborRgbMatch(buffer, gaborDataEx1, rgbDataEx1);
      //Extension 02: use a HVI color histogram to re-do matcching task 2.
      getHviMatch(buffer, HsvDataEx2, RgbDataEx2);


      if(strstr(dp->d_name, "pic.0001.jpg")){
        // reset file if filename is pic.0001.jpg and append 0001 data to csv.
        // Task 1 to open as write mode to csv (Baseline)
        append_image_data_csv("C:\\Users\\sidaz\\Desktop\\5330computerVision\\Projects\\project2\\data\\baselineData.csv", "pic.0001.jpg", baselineData, true);
        // Task 2 to open as write mode to csv (rg)
        append_image_data_csv("C:\\Users\\sidaz\\Desktop\\5330computerVision\\Projects\\project2\\data\\histData.csv", "pic.0001.jpg", histData, true);
        // Task 3 to open as write mode to csv (RGB and RGB)
        append_image_data_csv("C:\\Users\\sidaz\\Desktop\\5330computerVision\\Projects\\project2\\data\\multiData.csv", "pic.0001.jpg", multiData, true);
        // Task 4 to open as write mode to csv (Sobel and RGB)
        append_image_data_csv("C:\\Users\\sidaz\\Desktop\\5330computerVision\\Projects\\project2\\data\\sobelDataT4.csv", "pic.0001.jpg", sobelDataT4, true);
        append_image_data_csv("C:\\Users\\sidaz\\Desktop\\5330computerVision\\Projects\\project2\\data\\rgbDataT4.csv", "pic.0001.jpg", rgbDataT4, true);
        // Task 5 to open as write mode to csv (Sobel and RGB)
        append_image_data_csv("C:\\Users\\sidaz\\Desktop\\5330computerVision\\Projects\\project2\\data\\sobelDataT5.csv", "pic.0001.jpg", sobelDataT5, true);
        append_image_data_csv("C:\\Users\\sidaz\\Desktop\\5330computerVision\\Projects\\project2\\data\\rgbDataT5.csv", "pic.0001.jpg", rgbDataT5, true);
        // Extension 01 to open as write mode to csv (Gabor and RGB)
        append_image_data_csv("C:\\Users\\sidaz\\Desktop\\5330computerVision\\Projects\\project2\\data\\gaborDataEx1.csv", "pic.0001.jpg", gaborDataEx1, true);
        append_image_data_csv("C:\\Users\\sidaz\\Desktop\\5330computerVision\\Projects\\project2\\data\\rgbDataEx1.csv", "pic.0001.jpg", rgbDataEx1, true);
        // Extension 02 to open as write mode to csv (HSV and RGB)
        append_image_data_csv("C:\\Users\\sidaz\\Desktop\\5330computerVision\\Projects\\project2\\data\\HsvDataEx2.csv", "pic.0001.jpg", HsvDataEx2, true);
        append_image_data_csv("C:\\Users\\sidaz\\Desktop\\5330computerVision\\Projects\\project2\\data\\RgbDataEx2.csv", "pic.0001.jpg", RgbDataEx2, true);
      } else {
        // //append data to csv.
        append_image_data_csv("C:\\Users\\sidaz\\Desktop\\5330computerVision\\Projects\\project2\\data\\baselineData.csv", dp->d_name, baselineData, false);
        append_image_data_csv("C:\\Users\\sidaz\\Desktop\\5330computerVision\\Projects\\project2\\data\\histData.csv", dp->d_name, histData, false);
        append_image_data_csv("C:\\Users\\sidaz\\Desktop\\5330computerVision\\Projects\\project2\\data\\multiData.csv", dp->d_name, multiData, false);
        append_image_data_csv("C:\\Users\\sidaz\\Desktop\\5330computerVision\\Projects\\project2\\data\\sobelDataT4.csv", dp->d_name, sobelDataT4, false);
        append_image_data_csv("C:\\Users\\sidaz\\Desktop\\5330computerVision\\Projects\\project2\\data\\rgbDataT4.csv", dp->d_name, rgbDataT4, false);
        append_image_data_csv("C:\\Users\\sidaz\\Desktop\\5330computerVision\\Projects\\project2\\data\\sobelDataT5.csv", dp->d_name, sobelDataT5, false);
        append_image_data_csv("C:\\Users\\sidaz\\Desktop\\5330computerVision\\Projects\\project2\\data\\rgbDataT5.csv", dp->d_name, rgbDataT5, false);
        append_image_data_csv("C:\\Users\\sidaz\\Desktop\\5330computerVision\\Projects\\project2\\data\\gaborDataEx1.csv", dp->d_name, gaborDataEx1, false);
        append_image_data_csv("C:\\Users\\sidaz\\Desktop\\5330computerVision\\Projects\\project2\\data\\rgbDataEx1.csv", dp->d_name, rgbDataEx1, false);
        append_image_data_csv("C:\\Users\\sidaz\\Desktop\\5330computerVision\\Projects\\project2\\data\\HsvDataEx2.csv", dp->d_name, HsvDataEx2, false);
        append_image_data_csv("C:\\Users\\sidaz\\Desktop\\5330computerVision\\Projects\\project2\\data\\RgbDataEx2.csv", dp->d_name, RgbDataEx2, false);
      }
    }
  } 

  std::vector<char *> filenames;
  int match_size = 3; // set Task 1, 2, 3, 4 and 2 extensions with match size of 3.
  int match_size_10 = 10; // set Task 5 of 10.
  std::vector<std::pair<std::string,float>> result; //print or show result(0-match_size);
  

  //task 1:
  std::vector<std::vector<float>> baselineData;
  strcpy(csv, "C:\\Users\\sidaz\\Desktop\\5330computerVision\\Projects\\project2\\data\\baselineData.csv");
  read_image_data_csv(csv, filenames, baselineData, false);
  strcpy(buffer, dirname);
  strcat(buffer, "\\");
  strcat(buffer, "pic.1016.jpg");
  // Distance Matching
  distMetric(filenames, baselineData, buffer, match_size, result);

  std::cout << "\n The top 3 results for 1016 matching are" << std::endl;
  for(int i = 0; i <match_size; i++){
    std::cout << result[i].first << std::endl;
  }

  //task 2:
  std::vector<std::vector<float>> histData;
  strcpy(csv, "C:\\Users\\sidaz\\Desktop\\5330computerVision\\Projects\\project2\\data\\histData.csv");
  read_image_data_csv(csv, filenames, histData, false);
  strcpy(buffer, dirname);
  strcat(buffer, "\\");
  strcat(buffer, "pic.0164.jpg");
  // rg Matching
  histMatching(filenames, histData, buffer, match_size, result);
  
  std::cout << "\n The top 3 results for 0164 matching are" << std::endl;
  for(int i = 0; i <match_size; i++){
    std::cout << result[i].first << std::endl;
  }
  
  // //task 3:
  std::vector<std::vector<float>> multiData;
  strcpy(csv, "C:\\Users\\sidaz\\Desktop\\5330computerVision\\Projects\\project2\\data\\multiData.csv");
  read_image_data_csv(csv, filenames, multiData, false);
  strcpy(buffer, dirname);
  strcat(buffer, "\\");
  strcat(buffer, "pic.0274.jpg");
  // RGB and RGB Matching
  task3Matching(filenames, multiData, buffer, match_size, result);

  std::cout << "\n The top 3 results for 0274 matching are" << std::endl;
  for(int i = 0; i <match_size; i++){
    std::cout << result[i].first << std::endl;
  }

  //task 4:
  std::vector<std::vector<float>> sobelDataT4, rgbDataT4;
  strcpy(csv, "C:\\Users\\sidaz\\Desktop\\5330computerVision\\Projects\\project2\\data\\sobelDataT4.csv");
  read_image_data_csv(csv, filenames, sobelDataT4, false);
  strcpy(csv, "C:\\Users\\sidaz\\Desktop\\5330computerVision\\Projects\\project2\\data\\rgbDataT4.csv");
  read_image_data_csv(csv, filenames, rgbDataT4, false);
  strcpy(buffer, dirname);
  strcat(buffer, "\\");
  strcat(buffer, "pic.0535.jpg");
  // Sobel and RGB Matching
  task4Matching(filenames, sobelDataT4, rgbDataT4, buffer, match_size, result);
  
  std::cout << "\n The top 3 results for 0535 matching are" << std::endl;
  for(int i = 0; i <match_size; i++){
    std::cout << result[i].first << std::endl;
  }

  //task 5:
  // Sobel and RGB Matching
  std::vector<std::vector<float>> sobelDataT5, rgbDataT5;
  strcpy(csv, "C:\\Users\\sidaz\\Desktop\\5330computerVision\\Projects\\project2\\data\\sobelDataT5.csv");
  read_image_data_csv(csv, filenames, sobelDataT5, false);
  strcpy(csv, "C:\\Users\\sidaz\\Desktop\\5330computerVision\\Projects\\project2\\data\\rgbDataT5.csv");
  read_image_data_csv(csv, filenames, rgbDataT5, false);
  strcpy(buffer, dirname);
  strcat(buffer, "\\");
  strcat(buffer, "pic.0334.jpg");
  task5Matching(filenames, sobelDataT5, rgbDataT5, buffer, match_size_10, result);

  std::cout << "\n The top 10 results for 0280 matching are" << std::endl;
  for(int i = 0; i < match_size_10; i++){
    std::cout << result[i].first << std::endl;
  }

  // Extension 1 Use Gabor filter as feature histograms to get Task 4:
  std::vector<std::vector<float>> gaborDataEx1, rgbDataEx1;
  strcpy(csv, "C:\\Users\\sidaz\\Desktop\\5330computerVision\\Projects\\project2\\data\\gaborDataEx1.csv");
  read_image_data_csv(csv, filenames, gaborDataEx1, false);
  strcpy(csv, "C:\\Users\\sidaz\\Desktop\\5330computerVision\\Projects\\project2\\data\\rgbDataEx1.csv");
  read_image_data_csv(csv, filenames, rgbDataEx1, false);
  strcpy(buffer, dirname);
  strcat(buffer, "\\");
  strcat(buffer, "pic.0535.jpg");
  // Gabor and RGB Matching
  ex1GaborMatching(filenames, gaborDataEx1, rgbDataEx1, buffer, match_size, result);
  std::cout << "\n The top 3 results for 0535 matching are \n" << std::endl;
  for(int i = 0; i <match_size; i++){
    std::cout << result[i].first << std::endl;
  }

  // Extension 2 Use HSV and RGB as feature histograms to get Task 4:
  std::vector<std::vector<float>> HsvDataEx2, RgbDataEx2;
  strcpy(csv, "C:\\Users\\sidaz\\Desktop\\5330computerVision\\Projects\\project2\\data\\HsvDataEx2.csv");
  read_image_data_csv(csv, filenames, HsvDataEx2, false);
  strcpy(csv, "C:\\Users\\sidaz\\Desktop\\5330computerVision\\Projects\\project2\\data\\RgbDataEx2.csv");
  read_image_data_csv(csv, filenames, RgbDataEx2, false);
  strcpy(buffer, dirname);
  strcat(buffer, "\\");
  strcat(buffer, "pic.0274.jpg");
  // HSV and RGB Matching
  ex2HsvRgbMatching(filenames, HsvDataEx2, RgbDataEx2, buffer, match_size, result);
  
  std::cout << "\n The top 3 results for 0274 matching are" << std::endl;
  for(int i = 0; i <match_size; i++){
    std::cout << result[i].first << std::endl;
  }

  printf("Program has terminated\n");

  return(0);
}


