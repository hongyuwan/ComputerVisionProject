#ifndef filters_hpp
#define filters_hpp

#include <opencv2/core/mat.hpp>

namespace private_filters {

int blur5x5( cv::Mat &src, cv::Mat &dst );
int split( cv::Mat &dst, cv::Mat &dst16 );
int sobelX3x3( cv::Mat &src, cv::Mat &dst );
int sobelY3x3( cv::Mat &src, cv::Mat &dst );
int magnitude( cv::Mat &sx, cv::Mat &sy, cv::Mat &dst );
int blurQuantize( cv::Mat &src, cv::Mat &dst, int levels );
int cartoon( cv::Mat &src, cv::Mat&dst, int levels, int magThreshold );
int upbrightness( cv::Mat &src, cv::Mat &dst );
int downbrightness( cv::Mat &src, cv::Mat &dst );
int savevideo( cv::Mat &src, cv::Mat &dst );
int greyscale( cv::Mat &src, cv::Mat &dst );
}


#endif /* filters_hpp */