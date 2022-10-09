#include <iostream>
using namespace std;

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
int main(int argc, char** argv){
    cv::Mat image;
    image = cv::imread("download.jpg"); //image read
    if(image.data== nullptr)
    {
        cerr<<"Image not found"<<endl;
        return 0;
    }
    for(;;)
    {
    cv::imshow("window",image);
     char key = cv::waitKey(10);
    if( key == 'q') 
        {
         break;
        }
    }
    return 0;
}
