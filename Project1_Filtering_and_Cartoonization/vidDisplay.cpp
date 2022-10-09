
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>
#include "filters.hpp"
#include "filters.cpp"
#include <iostream>

using namespace cv;
int quantizeLevel = 15;
int magThreshold = 15;
int display_mode=0;
int frames_per_second = 10;
bool record=0;


int main(int argc, char *argv[]) {
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

    cv::namedWindow("Video", 1); // identifies a window
    cv::Mat frame;
    cv::Mat dst;
    cv::Mat dst_gray;
    *capdev >> frame;
    cv::Mat dst16(frame.rows,frame.cols,CV_16SC3);
    cv::Mat mag16(frame.rows,frame.cols,CV_16SC3);
    cv::Mat tmp16(frame.rows,frame.cols,CV_16SC3);
    cv::Mat magFloatTheta(frame.rows,frame.cols,CV_32FC3);
    dst = frame.clone();
    VideoWriter oVideoWriter("savevideo.avi", VideoWriter::fourcc('M', 'J', 'P', 'G'), frames_per_second, refS, true);
    //Create and initialize the VideoWriter object 

    
                      //Menu buttons:
    std::cout << "Menu:" << std::endl;
    std::cout << "1.Video press 'r' " << std::endl;
    std::cout << "2.Save an image from current video press 's' " << std::endl;
    std::cout << "3.Grayscale video press 'g' " << std::endl;
    std::cout << "4.Alrernative Grayscale video press 'h' " << std::endl;
    std::cout << "5.Blurred video press 'b' " << std::endl;
    std::cout << "6.Sobel in x direction press 'x' " << std::endl;
    std::cout << "7.Sobel in y direction press 'y' " << std::endl;
    std::cout << "8.Sobel Magnitude video press 'm' " << std::endl;
    std::cout << "9.Burred and Quantized Video press 'l' " << std::endl;
    std::cout << "10.Cartoonized video press 'c' " << std::endl;
    std::cout << "11.Up and down brightness press '[' or ']' " << std::endl;
    std::cout << "12.Save short video sequences with the special effects press 'p' to start and stop " << std::endl;
    std::cout << "13.Quit press 'q' :" << std::endl;

    
    for(;;) {
            *capdev >> frame; // get a new frame from the camera, treat as a stream
        
            if( frame.empty() )
            {
              printf("frame is empty\n");
              break;
            }
            char key = cv::waitKey(10);
        
            if (key == 'g') //greyscale
                display_mode = 1;   
            else if(key == 'r') //original video
                display_mode = 2;
            else if(key == 'b')
                display_mode = 3; //blur5x5
            else if(key == 'x')
                display_mode = 4; //sobelX
            else if(key == 'y')
                display_mode = 5;//sobelY
            else if(key == 'm')
                display_mode = 6; //magnitude
            else if(key == 'l')
                display_mode = 7; //blurQuantize
            else if(key == '[')
                display_mode = 8; //downbrightness
            else if(key == ']')
                display_mode = 9; // upbrightness
            else if(key == 'c') //cartoon
            {
                display_mode = 10;
                quantizeLevel = 15;
                magThreshold = 20;
            }
            else if(key == 'h')
                display_mode = 11; //alternative greyscale
            else if(key == 's') // save image
                imwrite("save.jpg",frame);
            else if(key == 'p')// save video
            {
                if(record==0)
                {
                record=1;
                cout<<"start recording"<<endl;
                }
                else
                {
                oVideoWriter.release();
                cout<<"finish recording"<<endl;
                record=0;;
                }
            }
            if(display_mode == 2)
            {    cv::imshow("Video", frame);
                if(record==1) oVideoWriter.write(frame);
            }
            else if (display_mode == 1)
            {
                cvtColor(frame, dst, COLOR_RGBA2GRAY, 0);
                cv::imshow("Video", dst);
                if(record==1) oVideoWriter.write(dst);
            }
            else if (display_mode == 3)
            {
                blur5x5(frame,dst);
                cv::imshow("Video", dst);
                if(record==1) oVideoWriter.write(dst);
            }
            else if (display_mode == 9)
            {
                upbrightness(frame,dst);
                cv::imshow("Video", dst);
                if(record==1) oVideoWriter.write(dst);
            }
            else if (display_mode == 8)
            {
                downbrightness(frame,dst);
                cv::imshow("Video", dst);
                if(record==1) oVideoWriter.write(dst);
            }
            else if (display_mode == 5)
            {
                sobelY3x3(frame,dst16);
                split(dst,dst16);
                cv::imshow("Video", dst);
                if(record==1) oVideoWriter.write(dst);
            }
            else if (display_mode == 4)
            {
                sobelX3x3(frame,dst16);
                split(dst,dst16);
                cv::imshow("Video", dst);
                if(record==1) oVideoWriter.write(dst);
            }
            else if (display_mode ==6)
            {
                sobelX3x3(frame,dst16);
                sobelY3x3(frame,tmp16);
                magnitude(dst16,tmp16,dst);
                cv::imshow("Video", dst);
                    if(record==1) oVideoWriter.write(dst);
            }
            else if (display_mode == 7)
            {
               blurQuantize(frame,dst,quantizeLevel);
                cv::imshow("Video", dst);
                    if(record==1) oVideoWriter.write(dst);
            }
            else if (display_mode == 10)
            {
                cartoon(frame,dst,quantizeLevel,magThreshold);
                cv::imshow("Video", dst);
                    if(record==1) oVideoWriter.write(dst);
            }
            else if (display_mode == 11)
            {
                greyscale(frame,dst);
                cv::imshow("Video", dst);
                    if(record==1) oVideoWriter.write(dst);
            }
            else
            {
                cv::imshow("Video", frame);
                if(record==1) oVideoWriter.write(frame);
            }
            // see if there is a waiting keystroke
    
            if( key == 'q') {
                break;
            }
    }

    delete capdev;
    return(0);
}