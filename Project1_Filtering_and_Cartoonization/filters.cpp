
#include"filters.hpp"
#include <opencv2/imgproc.hpp>
#include<cmath>
using namespace std;
using namespace cv;
using namespace private_filters;
int value(int Max, int x)
{
    if(x < 0)
    {
        return -x - 1;
    }
    if(x >= Max)
    {
        return 2*Max - x - 1;
    }

    return x;
}

int private_filters::greyscale( cv::Mat &src, cv::Mat &dst )
{   std::vector<Mat>mv;
	cv::split(src, mv);
	mv[0] = mv[2];
	mv[1] = mv[2];
	cv::merge(mv, dst);
    return 0;
}
int private_filters::split( cv::Mat &dst, cv::Mat &dst16 )
{
    for(int y = 0; y < dst16.rows; y++)
    {
        for(int x = 0; x < dst16.cols; x++)
        {
            cv::Vec3s intensity = dst16.at<Vec3s>(y, x);
            cv::Vec3b dstPix = dst.at<Vec3b>(y, x);
            dstPix.val[0] = (intensity.val[0]);
            dstPix.val[1] = (intensity.val[1]);
            dstPix.val[2] = (intensity.val[2]);
            dst.at<Vec3b>(y, x) = dstPix;
        }
    }
    return 0;
}

int private_filters::blur5x5( cv::Mat &src, cv::Mat &dst )
{
 
      Mat temp;
      float x1, y1;
 
      //Seperable 1D kernel (Normalized)
      double coeffs[] = {0.1, 0.2, 0.4, 0.2, 0.1};
 
      dst = src.clone();
      temp = src.clone();
 
        //Process along y direction
        for(int y = 0; y < src.rows; y++){
            for(int x = 0; x < src.cols; x++){
                float sum[3] = {0.0,0.0,0.0};
                for(int i = -2; i <= 2; i++){
                    y1 =value(src.rows, y - i);
                    
                    cv::Vec3b intensity = src.at<Vec3b>(y1, x);
                    
                    sum[0] = sum[0] + coeffs[i + 2]*intensity.val[0];
                    sum[1] = sum[1] + coeffs[i + 2]*intensity.val[1];
                    sum[2] = sum[2] + coeffs[i + 2]*intensity.val[2];
                    
                }
                cv::Vec3b tempPix = temp.at<Vec3b>(y, x);
                tempPix.val[0] = sum[0];
                tempPix.val[1] = sum[1];
                tempPix.val[2] = sum[2];
                temp.at<Vec3b>(y, x) = tempPix;
            }
        }
 
        //Process along x direction
        for(int y = 0; y < src.rows; y++){
            for(int x = 0; x < src.cols; x++){
                float sum[3] = {0.0,0.0,0.0};
                for(int i = -2; i <= 2; i++){
                    x1 = value(src.cols, x - i);
                    
                    cv::Vec3b intensity = temp.at<Vec3b>(y, x1);
                    
                    sum[0] = sum[0] + coeffs[i + 2]*intensity.val[0];
                    sum[1] = sum[1] + coeffs[i + 2]*intensity.val[1];
                    sum[2] = sum[2] + coeffs[i + 2]*intensity.val[2];
                }
                cv::Vec3b dstPix = dst.at<Vec3b>(y, x);
                dstPix.val[0] = sum[0];
                dstPix.val[1] = sum[1];
                dstPix.val[2] = sum[2];
                dst.at<Vec3b>(y, x) = dstPix;
            }
        }
 
    return 0;
}

int doSobel(cv::Mat &src, cv::Mat &dst, double coeffx[3], double coeffy[3])
{
    cv::Mat temp(src.rows, src.cols, CV_16SC3);
    float x1, y1;
    //Process along y direction
    for(int y = 0; y < src.rows; y++)
    {
      for(int x = 0; x < src.cols; x++)
      {
          float sum[3] = {0.0,0.0,0.0};
          for(int i = -1; i <= 1; i++)
          {
              y1 = value(src.rows, y - i);
              
              cv::Vec3b intensity = src.at<Vec3b>(y1, x);
              
              sum[0] = sum[0] + coeffy[i + 1]*intensity.val[0];
              sum[1] = sum[1] + coeffy[i + 1]*intensity.val[1];
              sum[2] = sum[2] + coeffy[i + 1]*intensity.val[2];
              
          }
          cv::Vec3s tempPix = temp.at<Vec3s>(y, x);
          tempPix.val[0] = abs(sum[0]);
          tempPix.val[1] = abs(sum[1]);
          tempPix.val[2] = abs(sum[2]);
          temp.at<Vec3s>(y, x) = tempPix;
      }
    }

    //Process along x direction
    for(int y = 0; y < src.rows; y++)
    {
      for(int x = 0; x < src.cols; x++)
      {
          float sum[3] = {0.0,0.0,0.0};
          for(int i = -1; i <= 1; i++)
          {
              x1 = value(src.cols, x - i);
              
              cv::Vec3s intensity = temp.at<Vec3s>(y, x1);
              
              sum[0] = sum[0] + coeffx[i + 1]*intensity.val[0];
              sum[1] = sum[1] + coeffx[i + 1]*intensity.val[1];
              sum[2] = sum[2] + coeffx[i + 1]*intensity.val[2];
          }
          cv::Vec3s dstPix = dst.at<Vec3s>(y, x);
          dstPix.val[0] = abs(sum[0]/4);
          dstPix.val[1] = abs(sum[1]/4);
          dstPix.val[2] = abs(sum[2]/4);
          dst.at<Vec3s>(y, x) = dstPix;
      }
    }

  return 0;
}

int private_filters::sobelX3x3( cv::Mat &src, cv::Mat &dst )
{ 
    double coeffx[3] = {1.0,2.0,1.0};
    double coeffy[3] = {1.0,0.0,-1.0};
    convertScaleAbs(src,src,1,0);
    doSobel(src,dst,coeffx,coeffy);
    return 0;
}


int private_filters::sobelY3x3( cv::Mat &src, cv::Mat &dst )
{
    double coeffy[3] = {1.0,2.0,1.0};
    double coeffx[3] = {1.0,0.0,-1.0};
    convertScaleAbs(src,src,1,0);
    doSobel(src,dst,coeffx,coeffy);
    return 0;
}

int private_filters::magnitude( cv::Mat &sx, cv::Mat &sy, cv::Mat &dst )
{
    for(int y = 0; y < sx.rows; y++)
    {
      for(int x = 0; x < sx.cols; x++)
      {
          
          cv::Vec3s sxIntensity = sx.at<Vec3s>(y, x);
          cv::Vec3s syIntensity = sy.at<Vec3s>(y, x);
          cv::Vec3b tempPix = dst.at<Vec3b>(y, x);
          tempPix.val[0] = (unsigned char)sqrt(sxIntensity[0] * sxIntensity[0] + syIntensity[0] * syIntensity[0]);
          tempPix.val[1] = (unsigned char)sqrt(sxIntensity[1] * sxIntensity[1] + syIntensity[1] * syIntensity[1]);
          tempPix.val[2] = (unsigned char)sqrt(sxIntensity[2] * sxIntensity[2] + syIntensity[2] * syIntensity[2]);
          dst.at<Vec3b>(y, x) = tempPix;
      }
    }
    return 0;
}

int private_filters::blurQuantize( cv::Mat &src, cv::Mat &dst, int levels )
{
    private_filters::blur5x5(src,dst);
    int b = 255/levels;
    
    for(int y = 0; y < src.rows; y++)
    {
      for(int x = 0; x < src.cols; x++)
      {
          cv::Vec3b tempPix = dst.at<Vec3b>(y, x);
          tempPix.val[0] /= b;
          tempPix.val[0] *= b;
          tempPix.val[1] /= b;
          tempPix.val[1] *= b;
          tempPix.val[2] /= b;
          tempPix.val[2] *= b;
          
          dst.at<Vec3b>(y, x) = tempPix;
      }
    }
    return 0;
    
}

int private_filters::cartoon( cv::Mat &src, cv::Mat &dst, int levels, int magThreshold )
{
    cv::Mat dst16(src.rows,src.cols,CV_16SC3);
    cv::Mat tmp16(src.rows,src.cols,CV_16SC3);
    cv::Mat magDst;
    magDst = src.clone();
    private_filters::sobelX3x3(src,dst16);
    private_filters::sobelY3x3(src,tmp16);
    private_filters::magnitude(dst16,tmp16,magDst);
    private_filters::blurQuantize(src,dst,levels);
    for(int y = 0; y < src.rows; y++)
    {
      for(int x = 0; x < src.cols; x++)
      {
          cv::Vec3b dstPix = dst.at<Vec3b>(y, x);
          cv::Vec3b magPix = magDst.at<Vec3b>(y, x);
          for(int i = 0; i < 3; i++)
          {
              if(magPix.val[i] > magThreshold)
              {
                  dstPix.val[i] = 0;
              }
          }
          dst.at<Vec3b>(y, x) = dstPix;
      }
    }
    return 0;
}


int private_filters::downbrightness( cv::Mat &src, cv::Mat &dst)
{   
    dst = src + cv::Scalar(-30, -30, -30);
    return 0;
}

int private_filters::upbrightness( cv::Mat &src, cv::Mat &dst)
{   
    dst = src + cv::Scalar(30, 30, 30);
    return 0;
}