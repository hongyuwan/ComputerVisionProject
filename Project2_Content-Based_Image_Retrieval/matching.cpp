#include <iostream>
#include <opencv2/opencv.hpp>
#include <typeinfo>
#include "filter.cpp" //filter for Sobel Magnitude calculation from project 1.

using namespace cv;

// Compare two 2-D vector's second value and return the lower one. (Geeksforgeeks)
bool cmp(std::pair<std::string, float>& a, std::pair<std::string, float>& b){
    return a.second < b.second;
}


int getBaseline(char *imageFile, std::vector<float> &dst){
    cv::Mat image = cv::imread(imageFile);
    int i, j, c;   
    
    // Set grid 9x9 in the middle
    for (i = image.rows/2 - 4 ; i< image.rows/ 2 + 5; i++){
        for(j = image.cols/2 - 4; j < image.cols/ 2 + 5; j++){
            for (c = 0; c < 3 ; c++){
                dst.push_back(image.at<cv::Vec3b>(i, j)[c]);
            }
        }
    }

    return 0;
}

int distMetric(std::vector<char *> &filenames, std::vector<std::vector<float>> &data, char *imageFile, int &match_size, std::vector<std::pair<std::string,float>> &result){
    int i, j;
    std::vector<float> targetData;
    getBaseline(imageFile, targetData);

    // std::map<std::string, float> compareDist;
    std::vector<std::pair<std::string,float>> compareDist;

    for (i = 0; i < data.size(); i++){
        float distanceM = 0;
        for(j = 0; j < data[i].size(); j++){
            // distance metric
            distanceM += (data[i][j] - targetData[j]) * (data[i][j] - targetData[j]);
        }
        // compareDist.insert({filenames[i], distanceM});
        compareDist.push_back(std::make_pair(filenames[i], distanceM));
    }

    sort(compareDist.begin(), compareDist.end(), cmp); // geeksforgeeks

    // return top 3 matches;
    result = {compareDist.begin() + 1, compareDist.begin() + (match_size + 1)};

    return 0;
}

//Histogram Matching - to - CSV 
int getHistMatch(char *imageFile, std::vector<float> &dst){
    cv::Mat image = cv::imread(imageFile);
    float r, g;
    int i, j, rix, gix, bins;
    const int hsize = 16;
    cv::Mat hist2d;
    int dim[2] {hsize, hsize};
    hist2d = cv::Mat::zeros(2, dim, CV_32S);
    bins = 16;

    // rg chromacity calculation
    for (i = 0; i< image.rows; i++){
        for(j = 0; j < image.cols; j++){
            // 0: B;
            // 1: G;
            // 2: R;

            r = ((float)image.at<cv::Vec3b>(i, j)[2]) / (image.at<cv::Vec3b>(i, j)[0] + image.at<cv::Vec3b>(i, j)[1] + image.at<cv::Vec3b>(i, j)[2] + 1);
            g = ((float)image.at<cv::Vec3b>(i, j)[1]) / (image.at<cv::Vec3b>(i, j)[0] + image.at<cv::Vec3b>(i, j)[1] + image.at<cv::Vec3b>(i, j)[2] + 1);
            rix = (int)(r * bins);
            gix = (int)(g * bins);

            hist2d.at<unsigned int>(rix, gix)++;
        }
    }
    
    for(i = 0; i < hsize; i++){
        for(j = 0; j < hsize; j++){
            dst.push_back(hist2d.at<int>(i,j));
        }
    }
    return 0;
}

//Histogram Matching - calculation & sort
int histMatching(std::vector<char *> &filenames, std::vector<std::vector<float>> &data, char *imageFile, int &match_size, std::vector<std::pair<std::string,float>> &result){
    int i, j, k;
    std::vector<float> targetData;
    getHistMatch(imageFile, targetData);

    // std::map<std::string, float> compareDist;
    std::vector<std::pair<std::string,float>> compareHist;

    for (i = 0; i < data.size(); i++){
        float histData = 0;
        for(j = 0; j < data[i].size(); j++){
            //histogram intersection
            histData += targetData[j] - std::min(targetData[j], data[i][j]);
        }
        
        // compareDist.insert({filenames[i], distanceM});
        compareHist.push_back(std::make_pair(filenames[i], histData));
    }

    sort(compareHist.begin(), compareHist.end(), cmp); // geeksforgeeks

    // return top 3 matches;
    result = {compareHist.begin() + 1, compareHist.begin() + (match_size + 1)};

    return 0;
}

//Multi histograms Matching - to - CSV 
int getMultiMatch(char *imageFile, std::vector<float> &dst){
    cv::Mat image = cv::imread(imageFile);
    int i, j, k, rix, gix, bix, bins, divisor;
    const int hsize = 8;
    cv::Mat hist3d_1, hist3d_2;
    int dim[3] {hsize, hsize, hsize};
    hist3d_1 = cv::Mat::zeros(3, dim, CV_32S);
    hist3d_2 = cv::Mat::zeros(3, dim, CV_32S);
    bins = 8;
    divisor = 256/ bins;


    //RGB calculation
    for (i = 0; i< image.rows/2; i++){
        for(j = 0; j < image.cols; j++){
            rix = (int)(image.at<cv::Vec3b>(i, j)[2] / divisor);
            gix = (int)(image.at<cv::Vec3b>(i, j)[1] / divisor);
            bix = (int)(image.at<cv::Vec3b>(i, j)[0] / divisor);

            hist3d_1.at<unsigned int>(rix, gix, bix)++;
        }
    }   
    for (i = image.rows/2; i< image.rows; i++){
        for(j = 0; j < image.cols; j++){
            rix = (int)(image.at<cv::Vec3b>(i, j)[2] / divisor);
            gix = (int)(image.at<cv::Vec3b>(i, j)[1] / divisor);
            bix = (int)(image.at<cv::Vec3b>(i, j)[0] / divisor);

            hist3d_2.at<unsigned int>(rix, gix, bix)++;
        }
    }
    
    
    for(i = 0; i < hsize; i++){
        for(j = 0; j < hsize; j++){
            for(k = 0; k < hsize; k++){
                dst.push_back(hist3d_1.at<int>(i,j,k));
            }
        }
    }

    for(i = 0; i < hsize; i++){
        for(j = 0; j < hsize; j++){
            for(k = 0; k < hsize; k++){
                dst.push_back(hist3d_2.at<int>(i,j,k));
            }
        }
    }

    return 0;
}

//multi histograms Matching - calculation & sort
int task3Matching(std::vector<char *> &filenames, std::vector<std::vector<float>> &data, char *imageFile, int &match_size, std::vector<std::pair<std::string,float>> &result){
    int i, j, k;
    std::vector<float> targetData;
    getMultiMatch(imageFile, targetData);

    // std::map<std::string, float> compareDist;
    std::vector<std::pair<std::string,float>> compareHist;

    for (i = 0; i < data.size(); i++){
        float histData = 0;
        for(j = 0; j < data[i].size(); j++){
            //histogram intersection
            histData += targetData[j] - std::min(targetData[j], data[i][j]);
        }
        compareHist.push_back(std::make_pair(filenames[i], histData));
    }

    sort(compareHist.begin(), compareHist.end(), cmp); // geeksforgeeks

    // return top 3 matches;
    result = {compareHist.begin() + 1, compareHist.begin() + (match_size + 1)};

    return 0;
}

// RGB chrom and texture (magnitute) Matching - to - CSV 
int getRGBsobel(char *imageFile, std::vector<float> &dst, std::vector<float> &dst2){
    cv::Mat image = cv::imread(imageFile);
    float r, g;
    int i, j, k, rix, gix, bix, gray, bins, divisor;
    const int hsize = 8;
    std::vector <float> hist1d_t(8);
    cv::Mat hist3d_c;
    int dim3[3] = {hsize, hsize, hsize};
    hist3d_c = cv::Mat::zeros(3, dim3, CV_32SC3);
    bins = 8;
    divisor = 256/ bins;

    // magnitude image calculation
    cv::Mat SobelX, SobelY, MagnitudeImage;
    sobelX3x3(image, SobelX);
    cv::convertScaleAbs(SobelX, SobelX);
    sobelY3x3(image, SobelY);
    cv::convertScaleAbs(SobelY, SobelY);
    magnitude(SobelX, SobelY, image, MagnitudeImage);
    cv::convertScaleAbs(MagnitudeImage, MagnitudeImage);
    cv::cvtColor(MagnitudeImage, MagnitudeImage, cv::COLOR_RGBA2GRAY);

    for(i = 0; i < MagnitudeImage.rows; i++){
        for(j = 0; j < MagnitudeImage.cols; j++){
            gray = (int)(image.at<uchar>(i, j) / divisor);

            hist1d_t[gray]++;
        }
    }



    // get RGB color histogram 
    for (i = 0; i < image.rows; i++){
        for(j = 0; j < image.cols; j++){
            rix = (int)(image.at<cv::Vec3b>(i, j)[2] / divisor);
            gix = (int)(image.at<cv::Vec3b>(i, j)[1] / divisor);
            bix = (int)(image.at<cv::Vec3b>(i, j)[0] / divisor);

            hist3d_c.at<unsigned int>(rix, gix, bix)++;
        }
    }

    for(i = 0; i < hsize; i++){
        dst.push_back(hist1d_t[i]);
    }
    
        
    for(i = 0; i < hsize; i++){
        for(j = 0; j < hsize; j++){
            for(k = 0; k < hsize; k++){
                dst2.push_back(hist3d_c.at<int>(i,j,k));
            }
        }
    }
    return 0;
}

//Color(RGB chrom) and Texture(magnitute) Matching - calculation & sort
int task4Matching(std::vector<char *> &filenames, std::vector<std::vector<float>> &sobelData, std::vector<std::vector<float>> &rgbData, char *imageFile, int &match_size, std::vector<std::pair<std::string,float>> &result){
    int i, j, k;
    std::vector<float> sobelTarget, rgbTarget;
    getRGBsobel(imageFile, sobelTarget, rgbTarget);

    // std::map<std::string, float> compareDist;
    std::vector<std::pair<std::string,float>> compareHist;

    for (i = 0; i < sobelData.size(); i++){
        float value = 0;
        float sobelHist = 0;
        float rgbHist = 0;
        for(j = 0; j < sobelData[i].size(); j++){
            // sobelHist += std::abs(sobelTarget[j] - sobelData[i][j]);

            //histogram intersection of sobel histograms.
            sobelHist += sobelTarget[j] - std::min(sobelTarget[j], sobelData[i][j]);
        }
        for(j = 0; j < rgbData[i].size(); j++){
            // rgbHist += std::abs(rgbTarget[j] - rgbData[i][j]);

            //histogram intersection of rgb histograms.
            rgbHist += rgbTarget[j] - std::min(rgbTarget[j], rgbData[i][j]);
        }
        value = sobelHist * 0.5 + rgbHist * 0.5;
        
        // compareDist.insert({filenames[i], distanceM});
        compareHist.push_back(std::make_pair(filenames[i], value));
    }

    sort(compareHist.begin(), compareHist.end(), cmp); // geeksforgeeks

    // return top 3 matches;
    result = {compareHist.begin() + 1, compareHist.begin() + (match_size + 1)};

    return 0;
}

// RGB chrom and texture (magnitute) Matching - to - CSV 
int getRgSobel(char *imageFile, std::vector<float> &dst, std::vector<float> &dst2){
    cv::Mat image = cv::imread(imageFile);
    float r, g;
    int i, j, k, rix, gix, bix, gray, bins, divisor;
    const int hsize = 8;
    std::vector <float> hist1d_t(8);
    cv::Mat hist3d_c;
    int dim[3] = {hsize, hsize, hsize};
    hist3d_c = cv::Mat::zeros(3, dim, CV_32SC3);
    bins = 8;
    divisor= 256 / bins;

    // get magnitude histogram of the image.
    cv::Mat SobelX, SobelY, MagnitudeImage;

    sobelX3x3(image, SobelX);
    sobelY3x3(image, SobelY);
    magnitude(SobelX, SobelY, image, MagnitudeImage);
    cv::convertScaleAbs(MagnitudeImage, MagnitudeImage);
    cv::cvtColor(MagnitudeImage, MagnitudeImage, cv::COLOR_RGBA2GRAY);

    for(i = 0; i < MagnitudeImage.rows; i++){
        for(j = 0; j < MagnitudeImage.cols; j++){
            gray = (int)(image.at<uchar>(i, j) / divisor);

            hist1d_t[gray]++;
        }
    }


    // get RGB color histogram 
    for (i = image.rows/2 - 128; i < image.rows/2 + 128; i++){
        for(j = image.cols/2 - 128; j < image.cols/2 + 128; j++){
            // 0: B;
            // 1: G;
            // 2: R;
            rix = (int)(image.at<cv::Vec3b>(i, j)[2] / divisor);
            gix = (int)(image.at<cv::Vec3b>(i, j)[1] / divisor);
            bix = (int)(image.at<cv::Vec3b>(i, j)[0] / divisor);

            hist3d_c.at<unsigned int>(rix, gix, bix)++;
        }
    }
    
    for(i = 0; i < hsize; i++){
        dst.push_back(hist1d_t[i]);
    }

    for(i = 0; i < hsize; i++){
        for(j = 0; j < hsize; j++){
            for(k = 0; k < hsize; k++){
                dst2.push_back(hist3d_c.at<int>(i,j,k));
            }
        }
    }
    return 0;
}

//Color(RGB chrom) and Texture(magnitute) Matching - calculation & sort
int task5Matching(std::vector<char *> &filenames, std::vector<std::vector<float>> &sobelData, std::vector<std::vector<float>> &rgbData, char *imageFile, int &match_size, std::vector<std::pair<std::string,float>> &result){
    int i, j, k;
    std::vector<float> sobelTarget, rgbTarget;
    getRgSobel(imageFile, sobelTarget, rgbTarget);

    // std::map<std::string, float> compareDist;
    std::vector<std::pair<std::string,float>> compareHist;

    for (i = 0; i < rgbData.size(); i++){
        float value = 0;
        float sobelHist = 0;
        float rgbHist = 0;
        for(j = 0; j < sobelData[i].size(); j++){
            // sobelHist += (sobelData[i][j] - sobelTarget[j]) * (sobelData[i][j] - sobelTarget[j]);

            //histogram intersection of two sobel histograms
            sobelHist += sobelTarget[j] - std::min(sobelTarget[j], sobelData[i][j]); 
        }
        for(j = 0; j < rgbData[i].size(); j++){
            // rgbHist += (rgbData[i][j] - rgbTarget[j]) * (rgbData[i][j] - rgbTarget[j]);
            
            //histogram intersection of two color histograms
            rgbHist += rgbTarget[j] - std::min(rgbTarget[j], rgbData[i][j]); 
        }
        // weight 0.1 for texture histograms and 0.9 for color histogram
        value = sobelHist * 0.1 + rgbHist * 0.9;
        
        // compareDist.insert({filenames[i], distanceM});
        compareHist.push_back(std::make_pair(filenames[i], value));
    }

    sort(compareHist.begin(), compareHist.end(), cmp); // geeksforgeeks

    // return top 10 matches;
    result = {compareHist.begin() + 1, compareHist.begin() + (match_size + 1)};

    return 0;
}

// Gabor Texture and RGB Color - to csv:
int getGaborRgbMatch(char *imageFile, std::vector<float> &dst, std::vector<float> &dst2){
    cv::Mat image = cv::imread(imageFile);
    float r, g;
    int i, j, k, rix, gix, bix, gabor, bins, divisor;
    const int hsize = 8;
    std::vector <float> hist1d_t(8);
    cv::Mat hist3d_c;
    int dim3[3] = {hsize, hsize, hsize};
    hist3d_c = cv::Mat::zeros(3, dim3, CV_32SC3);
    bins = 8;
    divisor = 256/ bins;

    // this section is from https://docs.opencv.org/3.4/d4/d86/group__imgproc__filter.html#gae84c92d248183bd92fa713ce51cc3599
    // https://en.wikipedia.org/wiki/Gabor_filter for Gabor implementations
    cv::Mat GImage, gaborFilter;
    GImage.convertTo(gaborFilter, CV_32F);

    // get kernel size
    int filter_size = 31;
    // set Parameters for GaborKernel
    double sigma = 1, theta = 0, lambd = 1.0, gamma = 0.02, psi  = 0;
    cv::Mat filter = cv::getGaborKernel(cv::Size(filter_size,filter_size), sigma, theta, lambd, gamma, psi);
    cv::filter2D(image, gaborFilter, CV_32F, filter);

    Mat Gabor;
    gaborFilter.convertTo(Gabor,CV_8U,1.0/255.0); 

    cv::cvtColor(Gabor, Gabor, cv::COLOR_RGBA2GRAY);
    // cv::imshow("gabor",Gabor);
    // cv::waitKey();

    //////////////////////////////////////////////////// end of citation ////////////////////////////////////////////////////

    for(i = 0; i < Gabor.rows; i++){
        for(j = 0; j < Gabor.cols; j++){
            gabor = (int)(image.at<uchar>(i, j) / divisor);

            hist1d_t[gabor]++;
        }
    }

    // get RGB color histogram 
    for (i = 0; i < image.rows; i++){
        for(j = 0; j < image.cols; j++){
            rix = (int)(image.at<cv::Vec3b>(i, j)[2] / divisor);
            gix = (int)(image.at<cv::Vec3b>(i, j)[1] / divisor);
            bix = (int)(image.at<cv::Vec3b>(i, j)[0] / divisor);

            hist3d_c.at<unsigned int>(rix, gix, bix)++;
        }
    }

    for(i = 0; i < hsize; i++){
        dst.push_back(hist1d_t[i]);
    }
    
        
    for(i = 0; i < hsize; i++){
        for(j = 0; j < hsize; j++){
            for(k = 0; k < hsize; k++){
                dst2.push_back(hist3d_c.at<int>(i,j,k));
            }
        }
    }

    return 0;
}


//Color(RGB chrom) and Texture(Gabor) Matching - calculation & sort
int ex1GaborMatching(std::vector<char *> &filenames, std::vector<std::vector<float>> &gaborData, std::vector<std::vector<float>> &rgbData, char *imageFile, int &match_size, std::vector<std::pair<std::string,float>> &result){
    int i, j, k;
    std::vector<float> gaborTarget, rgbTarget;
    getRgSobel(imageFile, gaborTarget, rgbTarget);

    // std::map<std::string, float> compareDist;
    std::vector<std::pair<std::string,float>> compareHist;

    for (i = 0; i < rgbData.size(); i++){
        float value = 0;
        float gaborHist = 0;
        float rgbHist = 0;
        for(j = 0; j < gaborData[i].size(); j++){
            //histogram intersection of two texture histograms
            gaborHist += gaborTarget[j] - std::min(gaborTarget[j], gaborData[i][j]); 
        }
        for(j = 0; j < rgbData[i].size(); j++){
            //histogram intersection of two color histograms
            rgbHist += rgbTarget[j] - std::min(rgbTarget[j], rgbData[i][j]); 
        }
        // weight half and half for histograms
        value = gaborHist * 0.5 + rgbHist * 0.5;
        
        // compareDist.insert({filenames[i], distanceM});
        compareHist.push_back(std::make_pair(filenames[i], value));
    }

    sort(compareHist.begin(), compareHist.end(), cmp); // geeksforgeeks

    // return top 3 matches;
    result = {compareHist.begin() + 1, compareHist.begin() + (match_size + 1)};
    return 0;
}


//HSV histograms Matching - to CSV
int getHviMatch(char *imageFile, std::vector<float> &dst, std::vector<float> &dst2){
    cv::Mat image = cv::imread(imageFile);
    cv::Mat imageHSV;
    int i, j, k, rix, gix, bix, bins, divisor;
    const int hsize = 8;
    cv::Mat hist3d_1;
    cv::Mat hist3d_2;
    int dim[3] {hsize, hsize, hsize};
    hist3d_1 = cv::Mat::zeros(3, dim, CV_32S);
    hist3d_2 = cv::Mat::zeros(3, dim, CV_32S);
    bins = 8;
    divisor = 256/ bins;

    //get HSV and generate histogram
    cv::cvtColor(image, imageHSV, cv::COLOR_RGB2HSV);

    for (i = 0; i< imageHSV.rows/2; i++){
        for(j = 0; j < imageHSV.cols; j++){
            rix = (int)(imageHSV.at<cv::Vec3b>(i, j)[2] / divisor);
            gix = (int)(imageHSV.at<cv::Vec3b>(i, j)[1] / divisor);
            bix = (int)(imageHSV.at<cv::Vec3b>(i, j)[0] / divisor);

            hist3d_1.at<unsigned int>(rix, gix, bix)++;
        }
    }  
    
    //get RGB and generate histogram
    for (i = image.rows/2; i< image.rows; i++){
        for(j = 0; j < image.cols; j++){
            rix = (int)(image.at<cv::Vec3b>(i, j)[2] / divisor);
            gix = (int)(image.at<cv::Vec3b>(i, j)[1] / divisor);
            bix = (int)(image.at<cv::Vec3b>(i, j)[0] / divisor);

            hist3d_2.at<unsigned int>(rix, gix, bix)++;
        }
    }
    
    
    for(i = 0; i < hsize; i++){
        for(j = 0; j < hsize; j++){
            for(k = 0; k < hsize; k++){
                dst.push_back(hist3d_1.at<int>(i,j,k));
            }
        }
    }

    for(i = 0; i < hsize; i++){
        for(j = 0; j < hsize; j++){
            for(k = 0; k < hsize; k++){
                dst2.push_back(hist3d_2.at<int>(i,j,k));
            }
        }
    }

    return 0;
}

// HSV and RGB histograms Matching - calculation & sort
int ex2HsvRgbMatching(std::vector<char *> &filenames, std::vector<std::vector<float>> &hsvData,  std::vector<std::vector<float>> &rgbData, char *imageFile, int &match_size, std::vector<std::pair<std::string,float>> &result){
    int i, j, k;
    std::vector<float> hsvTarget, rgbTarget;
    getHviMatch(imageFile, hsvTarget, rgbTarget);

    // std::map<std::string, float> compareDist;
    std::vector<std::pair<std::string,float>> compareHist;

    for (i = 0; i < hsvData.size(); i++){
        float value = 0;
        float hsvHist = 0;
        float rgbHist = 0;
        for(j = 0; j < hsvData[i].size(); j++){
            //histogram intersection of two HSV histograms
            hsvHist += hsvTarget[j] - std::min(hsvTarget[j], hsvData[i][j]); 
        }
        for(j = 0; j < rgbData[i].size(); j++){
            //histogram intersection of two RGB histograms
            rgbHist += rgbTarget[j] - std::min(rgbTarget[j], rgbData[i][j]); 
        }
        // weight half and half for histograms
        value = hsvHist * 0.5 + rgbHist * 0.5;
        compareHist.push_back(std::make_pair(filenames[i], value));
    }

    sort(compareHist.begin(), compareHist.end(), cmp); // geeksforgeeks

    // return top 3 matches;
    result = {compareHist.begin() + 1, compareHist.begin() + (match_size + 1)};

    return 0;
}