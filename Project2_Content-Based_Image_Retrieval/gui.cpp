#include <iostream>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <dirent.h>
#include <opencv2/opencv.hpp>
#include "csv_util.cpp"
#include "matching.cpp"
#define CVUI_IMPLEMENTATION
#include "cvui.h"

#define WINDOW_NAME "Frame"
#define WINDOW_NAME2 "Target_Image"
#define SHOW_IMAGE "Matching_Result"

/*
  Course: Computer Vision - 5330 S22
  Project 2: Content-based Image Retrieval
  Name: Sida Zhang and Hongyu Wan
  Febuary 7, 2022
  
  This file create a GUi version of matching project. 
  The executable iterates stored csv files to run each
tasks and extensions to perform distance metric matching.

*/
int main(int argc, const char *argv[])
{
    char dirname[256];
    char buffer[256];
    char csv[256]; // csv file location
    char imageFiles[256];
    FILE *fp;
    DIR *dirp;
    struct dirent *dp;
    int i;

    // !set the directory path to default.
    strcpy(dirname, "C:\\Users\\sidaz\\Desktop\\5330computerVision\\Projects\\project2\\data\\olympus");
    std::vector<char *> filenames;
    std::vector<std::pair<std::string, float>> result;

    /******************gui****************************/
    cv::Mat image_dis;
    int Match_size = 3;
    int task = 0;
    int show_img = 0;
    char photo_num[20];
    char target_num[20];
    char show[256];
    char b[10] = ".jpg";
    bool Match = 0;
    bool played = 0;
    bool display = 0;
    cv::Mat task1image_dis;
    ////////////////////////////////////////////////////
    // Create a frame where components will be rendered to.
    cv::Mat frame = cv::Mat(700, 1500, CV_8UC3);

    // Init cvui and tell it to create a OpenCV window, i.e. cv::namedWindow(WINDOW_NAME).
    cvui::init(WINDOW_NAME);
    for (;;)
    {
        // Fill the frame with a nice color
        frame = cv::Scalar(49, 52, 49);

        // Render UI components to the frame

        // button of task 1
        if (cvui::button(frame, 1100, 20, "Task1:Baseline Matching"))
        {
            if (display == 1)
            {
                cv::destroyWindow(WINDOW_NAME2);
                display = 0;
            }
            task = 1;
            char path[100] = "./data/olympus/";
            char a[20] = "pic.";
            char c[20] = ".jpg";
            std::cin >> target_num;
            strcat(a, target_num);
            strcat(a, c);
            strcpy(photo_num, a);
            strcat(path, a);
            image_dis = cv::imread(path);
            display = 1;
        }

        // button of task 2
        if (cvui::button(frame, 1100, 60, "Task2:Histogram Matching"))
        {
            if (display == 1)
            {
                cv::destroyWindow(WINDOW_NAME2);
                display = 0;
            }
            task = 2;
            char path[100] = "./data/olympus/";
            char a[20] = "pic.";
            char c[20] = ".jpg";
            std::cin >> target_num;
            strcat(a, target_num);
            strcat(a, c);
            strcpy(photo_num, a);
            strcat(path, a);
            image_dis = cv::imread(path);
            display = 1;
        }

        // button of task 3
        if (cvui::button(frame, 1100, 100, "Task3:Multi-histogram Matching"))
        {
            if (display == 1)
            {
                cv::destroyWindow(WINDOW_NAME2);
                display = 0;
            }
            task = 3;
            char path[100] = "./data/olympus/";
            char a[20] = "pic.";
            char c[20] = ".jpg";
            std::cin >> target_num;
            strcat(a, target_num);
            strcat(a, c);
            strcpy(photo_num, a);
            strcat(path, a);
            image_dis = cv::imread(path);
            display = 1;
        }

        // button of task 4
        if (cvui::button(frame, 1100, 140, "Task4:Texture and Color"))
        {
            if (display == 1)
            {
                cv::destroyWindow(WINDOW_NAME2);
                display = 0;
            }
            task = 4;
            char path[100] = "./data/olympus/";
            char a[20] = "pic.";
            char c[20] = ".jpg";
            std::cin >> target_num;
            strcat(a, target_num);
            strcat(a, c);
            strcpy(photo_num, a);
            strcat(path, a);
            image_dis = cv::imread(path);
            display = 1;
        }

        // button of task 5
        if (cvui::button(frame, 1100, 180, "Task5:Custom Design"))
        {
            if (display == 1)
            {
                cv::destroyWindow(WINDOW_NAME2);
                display = 0;
            }
            task = 5;
            char path[100] = "./data/olympus/";
            char a[20] = "pic.";
            char c[20] = ".jpg";
            std::cin >> target_num;
            strcat(a, target_num);
            strcat(a, c);
            strcpy(photo_num, a);
            strcat(path, a);
            image_dis = cv::imread(path);
            display = 1;
        }

        // button of Extension 01
        if (cvui::button(frame, 1100, 220, "Entension1:Garbor and RGB Matching"))
        {
            if (display == 1)
            {
                cv::destroyWindow(WINDOW_NAME2);
                display = 0;
            }
            task = 6;
            char path[100] = "./data/olympus/";
            char a[20] = "pic.";
            char c[20] = ".jpg";
            std::cin >> target_num;
            strcat(a, target_num);
            strcat(a, c);
            strcpy(photo_num, a);
            strcat(path, a);
            image_dis = cv::imread(path);
            display = 1;
        }

        // button of Extension 02
        if (cvui::button(frame, 1100, 260, "Entension2:HSV and RGB Matching"))
        {
            if (display == 1)
            {
                cv::destroyWindow(WINDOW_NAME2);
                display = 0;
            }
            task = 7;
            char path[100] = "./data/olympus/";
            char a[20] = "pic.";
            char c[20] = ".jpg";
            std::cin >> target_num;
            strcat(a, target_num);
            strcat(a, c);
            strcpy(photo_num, a);
            strcat(path, a);
            image_dis = cv::imread(path);
            display = 1;
        }

        // button of Start Matching
        if (cvui::button(frame, 1100, 540, "Start Matching"))
        {
            Match = 1;
            show_img = 0;
        }

        //button of Next Image
        if (cvui::button(frame, 1100, 580, "Next Image"))
        {
            show_img++;
            Match = 1;
        }

        //button of Previous Image
        if (cvui::button(frame, 1100, 620, "Previous Image"))
        {
            show_img--;
            Match = 1;
        }

        cvui::counter(frame, 1100, 500, &Match_size);

        cvui::text(frame, 1050, 340, "Please click up and down arrow to change match result size.");
        cvui::text(frame, 1050, 380, "Please enter target image number(E.g:0001) in the terminal.");
        cvui::text(frame, 1050, 420, "Click Start Matching to run match");

        //task 1:
        if (task == 1)
        {
            cv::imshow(WINDOW_NAME2, image_dis);
            cvui::printf(frame, 300, 60, "Task1:Baseline Matching");
            cvui::printf(frame, 0, 0, "Target_Image = pic.%s.jpg", target_num);
            if (Match == 1)
            {
                std::vector<std::vector<float>> baselineData;
                strcpy(csv, "C:\\Users\\sidaz\\Desktop\\5330computerVision\\Projects\\project2\\data\\baselineData.csv");
                read_image_data_csv(csv, filenames, baselineData, false);
                strcpy(buffer, dirname);
                strcat(buffer, "\\");
                strcat(buffer, photo_num);
                // Distance Matching
                distMetric(filenames, baselineData, buffer, Match_size, result);

                strcpy(imageFiles, dirname);
                strcat(imageFiles, "\\");
                if (show_img < Match_size)
                {
                    strcat(imageFiles, result[show_img].first.c_str());
                }

                task1image_dis = cv::imread(imageFiles);
                Match = 0;
                played = 1;
            }
        }

        //task 2:
        if (task == 2)
        {
            cv::imshow(WINDOW_NAME2, image_dis);
            cvui::printf(frame, 300, 60, "Task2:Histogram Matching");
            cvui::printf(frame, 0, 0, "Target_Image = pic.%s.jpg", target_num);
            if (Match == 1)
            {
                std::vector<std::vector<float>> histData;
                strcpy(csv, "C:\\Users\\sidaz\\Desktop\\5330computerVision\\Projects\\project2\\data\\histData.csv");
                read_image_data_csv(csv, filenames, histData, false);
                strcpy(buffer, dirname);
                strcat(buffer, "\\");
                strcat(buffer, photo_num);
                // rg Matching
                histMatching(filenames, histData, buffer, Match_size, result);

                strcpy(imageFiles, dirname);
                strcat(imageFiles, "\\");
                if (show_img < Match_size)
                {
                    strcat(imageFiles, result[show_img].first.c_str());
                }

                task1image_dis = cv::imread(imageFiles);
                Match = 0;
                played = 1;
            }
        }

        //task 3:
        if (task == 3)
        {
            cv::imshow(WINDOW_NAME2, image_dis);
            cvui::printf(frame, 300, 60, "Task3:Multi-histogram Matching");
            cvui::printf(frame, 0, 0, "Target_Image = pic.%s.jpg", target_num);
            if (Match == 1)
            {
                std::vector<std::vector<float>> multiData;
                strcpy(csv, "C:\\Users\\sidaz\\Desktop\\5330computerVision\\Projects\\project2\\data\\multiData.csv");
                read_image_data_csv(csv, filenames, multiData, false);
                strcpy(buffer, dirname);
                strcat(buffer, "\\");
                strcat(buffer, photo_num);
                // RGB and RGB Matching
                task3Matching(filenames, multiData, buffer, Match_size, result);

                strcpy(imageFiles, dirname);
                strcat(imageFiles, "\\");
                if (show_img < Match_size)
                {
                    strcat(imageFiles, result[show_img].first.c_str());
                }

                task1image_dis = cv::imread(imageFiles);
                Match = 0;
                played = 1;
            }
        }

        //task 4:
        if (task == 4)
        {
            cv::imshow(WINDOW_NAME2, image_dis);
            cvui::printf(frame, 300, 60, "Task4:Texture and Colorn");
            cvui::printf(frame, 0, 0, "Target_Image = pic.%s.jpg", target_num);
            if (Match == 1)
            {
                std::vector<std::vector<float>> sobelDataT4, rgbDataT4;
                strcpy(csv, "C:\\Users\\sidaz\\Desktop\\5330computerVision\\Projects\\project2\\data\\sobelDataT4.csv");
                read_image_data_csv(csv, filenames, sobelDataT4, false);
                strcpy(csv, "C:\\Users\\sidaz\\Desktop\\5330computerVision\\Projects\\project2\\data\\rgbDataT4.csv");
                read_image_data_csv(csv, filenames, rgbDataT4, false);
                strcpy(buffer, dirname);
                strcat(buffer, "\\");
                strcat(buffer, photo_num);
                // Sobel and RGB Matching
                task4Matching(filenames, sobelDataT4, rgbDataT4, buffer, Match_size, result);

                strcpy(imageFiles, dirname);
                strcat(imageFiles, "\\");
                if (show_img < Match_size)
                {
                    strcat(imageFiles, result[show_img].first.c_str());
                }

                task1image_dis = cv::imread(imageFiles);
                Match = 0;
                played = 1;
            }
        }


        //task 5:
        if (task == 5)
        {
            cv::imshow(WINDOW_NAME2, image_dis);
            cvui::printf(frame, 300, 60, "Task5:Custon Desig");
            cvui::printf(frame, 0, 0, "Target_Image = pic.%s.jpg", target_num);
            if (Match == 1)
            {
                std::vector<std::vector<float>> sobelDataT5, rgbDataT5;
                strcpy(csv, "C:\\Users\\sidaz\\Desktop\\5330computerVision\\Projects\\project2\\data\\sobelDataT5.csv");
                read_image_data_csv(csv, filenames, sobelDataT5, false);
                strcpy(csv, "C:\\Users\\sidaz\\Desktop\\5330computerVision\\Projects\\project2\\data\\rgbDataT5.csv");
                read_image_data_csv(csv, filenames, rgbDataT5, false);
                strcpy(buffer, dirname);
                strcat(buffer, "\\");
                strcat(buffer, photo_num);
                // Sobel and RGB Matching
                task5Matching(filenames, sobelDataT5, rgbDataT5, buffer, Match_size, result);

                strcpy(imageFiles, dirname);
                strcat(imageFiles, "\\");
                if (show_img < Match_size)
                {
                    strcat(imageFiles, result[show_img].first.c_str());
                }

                task1image_dis = cv::imread(imageFiles);
                Match = 0;
                played = 1;
            }
        }

        // Extension 1 Use Gabor filter as feature histograms to get Task 4:
        if (task == 6)
        {
            cv::imshow(WINDOW_NAME2, image_dis);
            cvui::printf(frame, 300, 60, "Entension1:Garbor and RGB Matching");
            cvui::printf(frame, 0, 0, "Target_Image = pic.%s.jpg", target_num);
            if (Match == 1)
            {
                std::vector<std::vector<float>> gaborDataEx1, rgbDataEx1;
                strcpy(csv, "C:\\Users\\sidaz\\Desktop\\5330computerVision\\Projects\\project2\\data\\gaborDataEx1.csv");
                read_image_data_csv(csv, filenames, gaborDataEx1, false);
                strcpy(csv, "C:\\Users\\sidaz\\Desktop\\5330computerVision\\Projects\\project2\\data\\rgbDataEx1.csv");
                read_image_data_csv(csv, filenames, rgbDataEx1, false);
                strcpy(buffer, dirname);
                strcat(buffer, "\\");
                strcat(buffer, photo_num);
                // Gabor and RGB Matching
                ex1GaborMatching(filenames, gaborDataEx1, rgbDataEx1, buffer, Match_size, result);

                strcpy(imageFiles, dirname);
                strcat(imageFiles, "\\");
                if (show_img < Match_size)
                {
                    strcat(imageFiles, result[show_img].first.c_str());
                }

                task1image_dis = cv::imread(imageFiles);
                Match = 0;
                played = 1;
            }
        }


        // Extension 2 Use HSV and RGB as feature histograms to get Task 4:
        if (task == 7)
        {
            cv::imshow(WINDOW_NAME2, image_dis);
            cvui::printf(frame, 300, 60, "Entension2:HSV and RGB Matchingg");
            cvui::printf(frame, 0, 0, "Target_Image = pic.%s.jpg", target_num);
            if (Match == 1)
            {
                std::vector<std::vector<float>> HsvDataEx2, RgbDataEx2;
                strcpy(csv, "C:\\Users\\sidaz\\Desktop\\5330computerVision\\Projects\\project2\\data\\HsvDataEx2.csv");
                read_image_data_csv(csv, filenames, HsvDataEx2, false);
                strcpy(csv, "C:\\Users\\sidaz\\Desktop\\5330computerVision\\Projects\\project2\\data\\RgbDataEx2.csv");
                read_image_data_csv(csv, filenames, RgbDataEx2, false);
                strcpy(buffer, dirname);
                strcat(buffer, "\\");
                strcat(buffer, photo_num);
                // HSV and RGB Matching
                ex2HsvRgbMatching(filenames, HsvDataEx2, RgbDataEx2, buffer, Match_size, result);

                strcpy(imageFiles, dirname);
                strcat(imageFiles, "\\");
                if (show_img < Match_size)
                {
                    strcat(imageFiles, result[show_img].first.c_str());
                }

                task1image_dis = cv::imread(imageFiles);
                Match = 0;
                played = 1;
            }
        }
        cvui::update();
        
        //print image and image number
        int n = strlen(imageFiles);
        cvui::printf(frame, 300, 650, 0.8, 0x00ff00, "Match[%i] = %.*s", show_img, 12, imageFiles + (n - 12));
        if (played == 1)
        {
            if (show_img < Match_size)
            {
                cvui::image(frame, 10, 100, task1image_dis);
            }
            else
            {
                cvui::printf(frame, 10, 100, "Warning: Out of size number.");
            }
        }

        cvui::imshow(WINDOW_NAME, frame);

        if (cv::waitKey(20) == 27)
        {

            break;
        }
    }

    return 0;
}