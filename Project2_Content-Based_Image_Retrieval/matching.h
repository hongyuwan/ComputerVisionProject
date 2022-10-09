#ifndef FILTERS_H
#define FILTERS_H

#include <iostream>
#include <opencv2/opencv.hpp>
#include <typeinfo>

using namespace cv;

bool cmp(std::pair<std::string, float>& a, std::pair<std::string, float>& b);
int getBaseline(char *imageFile, std::vector<float> &dst);
int getHistMatch(char *imageFile, std::vector<float> &dst);
int histMatching(std::vector<char *> &filenames, std::vector<std::vector<float>> &data, char *imageFile, int &match_size, std::vector<std::pair<std::string,float>> &result);
int getMultiMatch(char *imageFile, std::vector<float> &dst);
int task3Matching(std::vector<char *> &filenames, std::vector<std::vector<float>> &data, char *imageFile, int &match_size, std::vector<std::pair<std::string,float>> &result);
int getRGBsobel(char *imageFile, std::vector<float> &dst, std::vector<float> &dst2);
int task4Matching(std::vector<char *> &filenames, std::vector<std::vector<float>> &sobelData, std::vector<std::vector<float>> &rgbData, char *imageFile, int &match_size, std::vector<std::pair<std::string,float>> &result);
int getRgSobel(char *imageFile, std::vector<float> &dst, std::vector<float> &dst2);
int task5Matching(std::vector<char *> &filenames, std::vector<std::vector<float>> &sobelData, std::vector<std::vector<float>> &rgbData, char *imageFile, int &match_size, std::vector<std::pair<std::string,float>> &result);
int getGaborRgbMatch(char *imageFile, std::vector<float> &dst, std::vector<float> &dst2);
int ex1GaborMatching(std::vector<char *> &filenames, std::vector<std::vector<float>> &gaborData, std::vector<std::vector<float>> &rgbData, char *imageFile, int &match_size, std::vector<std::pair<std::string,float>> &result);
int getHviMatch(char *imageFile, std::vector<float> &dst, std::vector<float> &dst2);
int ex2HsvRgbMatching(std::vector<char *> &filenames, std::vector<std::vector<float>> &hsvData,  std::vector<std::vector<float>> &rgbData, char *imageFile, int &match_size, std::vector<std::pair<std::string,float>> &result);

#endif