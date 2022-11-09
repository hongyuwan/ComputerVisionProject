/*
  Course: Computer Vision - 5330 S22
  Project 3: Real-time Object 2-D Recognition
  Name: Sida Zhang and Hongyu Wan
  Febuary 26, 2022

  This file contains all the functions for DB feature extration and
  Camera feature extration.
  Including Threshold, cleanup, connectComponent, FeatureCompute,
  Collect feature from video, collect feature from database, 
  Clear feature from the database, save image, save feature to database,
  Find Euclidean Distance Metric, and find the KNN.
*/


using namespace cv;
// compare two integers in descending order.
bool cmpInt(int &a, int &b){
    return a > b;
}
// compare two float point in pairs.
bool cmpPair(std::pair<std::string, float>& a, std::pair<std::string, float>& b){
    return a.second < b.second;
}

// this function threasholds an image to a binary image by the thresh level
// it also darken a pixel if the pixel has high saturation.
// thresholding for Mat input
int thresholding(cv::Mat &src, int thresh, cv::Mat &dst)
{   
    int i, j;
    dst = src.clone();

    for (i = 0; i < src.rows; i++)
    {
        for (j = 0; j < src.cols; j++)
        {
            // getting the saturation value of each pixel
            double max, min;
            double sat;
            max = src.at<cv::Vec3b>(i, j)[0];
            if (src.at<cv::Vec3b>(i, j)[1] > max)
                max = src.at<cv::Vec3b>(i, j)[1];
            if (src.at<cv::Vec3b>(i, j)[2] > max)
                max = src.at<cv::Vec3b>(i, j)[2];
            min = src.at<cv::Vec3b>(i, j)[0];
            if (src.at<cv::Vec3b>(i, j)[1] < min)
                max = src.at<cv::Vec3b>(i, j)[1];
            if (src.at<cv::Vec3b>(i, j)[2] < min)
                max = src.at<cv::Vec3b>(i, j)[2];

            if (max != 0)
            {
                sat = (max - min) / max;
            }
            else
            {
                sat = 0;
            }
            // darken a pixel by -10 if the pixel's saturation is more than 50%
            if (sat > 0.5)
            {
                if (src.at<cv::Vec3b>(i, j)[0] > 10)
                    src.at<cv::Vec3b>(i, j)[0] -= 10;
                if (src.at<cv::Vec3b>(i, j)[1] > 10)
                    src.at<cv::Vec3b>(i, j)[1] -= 10;
                if (src.at<cv::Vec3b>(i, j)[2] > 10)
                    src.at<cv::Vec3b>(i, j)[2] -= 10;
            }

            // thresholding the source image with threshold of 60 to a binary image
            if (src.at<cv::Vec3b>(i, j)[0] > thresh &&
                src.at<cv::Vec3b>(i, j)[1] > thresh &&
                src.at<cv::Vec3b>(i, j)[2] > thresh)
            {
                dst.at<uchar>(i, j) = 0;
            }
            else
            {
                dst.at<uchar>(i, j) = 255;
            }
        }
    }
    return 0;
}

// thresholding for char input
int thresholding_d(char *imageFile, int thresh, cv::Mat &dst){
    int i, j;
    cv::Mat src = cv::imread(imageFile);
    dst = cv::Mat::zeros(src.rows, src.cols, CV_8UC1);

    for (i = 0; i < src.rows; i++){
        for(j = 0; j < src.cols; j++){
            // getting the saturation value of each pixel
            double max, min;
            double sat;
            max = src.at<cv::Vec3b>(i, j)[0];
            if(src.at<cv::Vec3b>(i, j)[1] > max) max = src.at<cv::Vec3b>(i, j)[1];
            if(src.at<cv::Vec3b>(i, j)[2] > max) max = src.at<cv::Vec3b>(i, j)[2];
            min = src.at<cv::Vec3b>(i, j)[0];
            if(src.at<cv::Vec3b>(i, j)[1] < min) max = src.at<cv::Vec3b>(i, j)[1];
            if(src.at<cv::Vec3b>(i, j)[2] < min) max = src.at<cv::Vec3b>(i, j)[2];

            if(max != 0){
                sat = (max - min) / max;
            } else {
                sat = 0;
            }
            // darken a pixel by -10 if the pixel's saturation is more than 50%
            if(sat > 0.5){
                if(src.at<cv::Vec3b>(i, j)[0] > 10) src.at<cv::Vec3b>(i, j)[0] -= 10;
                if(src.at<cv::Vec3b>(i, j)[1] > 10) src.at<cv::Vec3b>(i, j)[1] -= 10;
                if(src.at<cv::Vec3b>(i, j)[2] > 10) src.at<cv::Vec3b>(i, j)[2] -= 10;
            }

            // thresholding the source image with threshold of 60 to a binary image
            if(src.at<cv::Vec3b>(i, j)[0] > thresh &&
                src.at<cv::Vec3b>(i, j)[1] > thresh &&
                src.at<cv::Vec3b>(i, j)[2] > thresh){
                dst.at<uchar>(i, j) = 0;
            } else {
                dst.at<uchar>(i, j) = 255;
            }
        }
    }
    return 0;
}
// growing helps to remove noises
int growing(cv::Mat &src, cv::Mat &dst){
    dst = cv::Mat::zeros(src.rows, src.cols, CV_8UC1);
    for (int i = 0; i < src.rows; ++i){
        for(int j = 0; j < src.cols; ++j){
            unsigned char max = 0;
            // getting the adjacent pixels
            for (int x = i - 1; x <= i + 1 ; x++){
                for(int y = j - 1; y <= j + 1; y++){
                    if(x < 0 || y < 0 || x >= src.rows || y >= src.cols){
                        continue;
                    }
                    // getting the max pixel value to perform shrinking.
                    // formula from opencv:
                    // citation: https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html#gaeb1e0c1033e3f6b891a25d0511362aeb
                    max = (std::max<uchar>)(max, src.at<uchar>(x, y));
                }
            }
            dst.at<uchar>(i, j) = max;
        }
    }
    return 0;
}

// shrinking helps to filling holes
int shrinking(cv::Mat &src, cv::Mat &dst){
    dst = cv::Mat::zeros(src.rows, src.cols, CV_8UC1);
    for (int i = 0; i < src.rows; ++i){
        for(int j = 0; j < src.cols; ++j){
            unsigned char min = 255;
            // getting the adjacent pixels
            for (int x = i - 1; x <= i + 1 ; x++){
                for(int y = j - 1; y <= j + 1; y++){
                    if(x < 0 || y < 0 || x >= src.rows || y >= src.cols){
                        continue;
                    }
                    // getting the min pixel value to perform shrinking.
                    // formula from opencv:
                    // citation: https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html#ga4ff0f3318642c4f469d0e11f242f3b6c
                    min = (std::min<uchar>)(min, src.at<uchar>(x, y));
                }
            }
            dst.at<uchar>(i, j) = min;
        }
    }

    return 0;
}
// clean up
int cleanup(cv::Mat &src, cv::Mat &dst){
    cv::Mat temp = cv::Mat::zeros(src.rows, src.cols, CV_8UC1);
    growing(src, temp);
    shrinking(temp, dst);

    return 0;
}

// Compute Connected Component with Segment Region Filter
// citation: https://docs.opencv.org/3.4/d3/dc0/group__imgproc__shape.html#ga107a78bf7cd25dec05fb4dfc5c9e765f
int ccAnalysis(cv::Mat &src, cv::Mat &dst, double &num){
    cv::Mat imgLabels, imgStats, imgCentriods;
    dst = cv::Mat::zeros(src.rows, src.cols, CV_8UC1);
    // find the connected components with its stats by 8 connected growing.
    int imgLabelNum = cv::connectedComponentsWithStats(src, imgLabels, imgStats, imgCentriods, 8);
    int maxArea = 0;
    
    // get components' area in order to find the biggest three objects.
    std::vector<int> imgStatsArea;
    for(int i = 0; i < imgStats.rows; i++){
        imgStatsArea.push_back(imgStats.at<int>(i, CC_STAT_AREA));
    }
    sort(imgStatsArea.begin(), imgStatsArea.end(), cmpInt);


    std::vector<uchar> colors(imgLabelNum + 1);
    // set the background remain black
    colors[0] = 0;
    for(int i = 1; i < imgLabelNum; i++){
        // set to random colors and find the biggest 6 areas.
        colors[i] = 255;

        //  Extensions: Multiple Objects: return the biggest 3 objects
        if(imgStats.at<int>(i, CC_STAT_AREA) <= imgStatsArea[4] || imgStatsArea[i] < num){
            colors[i] = 0;
        }
    }

    for( int x = 0; x < src.rows; x++){
        for( int y = 0; y < src.cols; y++){
            if(dst.at<uchar>(x, y) == 0){
                int label = imgLabels.at<int>(x, y);
                dst.at<uchar>(x, y) = colors[label];
            }
        }
    }


    return 0;
}

// Compute Connected Component without Segment Region Filter
int ccAnalysis_d(cv::Mat &src, cv::Mat &dst){
    cv::Mat imgLabels, imgStats, imgCentriods;
    dst = cv::Mat::zeros(src.rows, src.cols, CV_8UC1);
    // find the connected components with its stats by 8 connected growing.
    int imgLabelNum = cv::connectedComponentsWithStats(src, imgLabels, imgStats, imgCentriods, 8);
    int maxArea = 0;
    
    // get components' area in order to find the biggest three objects.
    std::vector<int> imgStatsArea;
    for(int i = 0; i < imgStats.rows; i++){
        imgStatsArea.push_back(imgStats.at<int>(i, CC_STAT_AREA));
    }
    sort(imgStatsArea.begin(), imgStatsArea.end(), cmpInt);


    std::vector<uchar> colors(imgLabelNum + 1);
    // set the background remain black
    colors[0] = 0;
    for(int i = 1; i < imgLabelNum; i++){
        // set to random colors and find the biggest 6 areas.
        colors[i] = 255;

        if(imgStats.at<int>(i, CC_STAT_AREA) <= imgStatsArea[2]){
            colors[i] = 0;
        }
    }

    for( int x = 0; x < src.rows; x++){
        for( int y = 0; y < src.cols; y++){
            if(dst.at<uchar>(x, y) == 0){
                int label = imgLabels.at<int>(x, y);
                dst.at<uchar>(x, y) = colors[label];
            }
        }
    }


    return 0;
}

// feature Compute for video stream
int featureCompute(cv::Mat &src, cv::Mat &dst)
{
    // find the contours from the src
    std::vector<std::vector<Point>> contours;
    findContours(src, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, Point(0, 0));
    // convert to RGB
    dst = cv::Mat::zeros(src.rows, src.cols, CV_8UC3);
    for (int i = 0; i < src.rows; i++)
    {
        for (int j = 0; j < src.cols; j++)
        {
            dst.at<cv::Vec3b>(i, j) = {src.at<uchar>(i, j), src.at<uchar>(i, j), src.at<uchar>(i, j)};
        }
    }

    // declear the bounding box;
    std::vector<cv::RotatedRect> minRect(contours.size());

    // declear the centroid of an object;
    std::vector<cv::Point2f> centriod(contours.size());

    // declear the moment of an object;
    std::vector<cv::Moments> monment(contours.size());
    std::vector<cv::Vec3b> colors(contours.size() + 1);

    colors[0] = cv::Vec3b(0, 0, 0);
    for (int k = 0; k < contours.size(); k++){
        // find the moment of the object;
        monment[k] = moments(contours[k]);
        // find the bounding box of the object;
        minRect[k] = minAreaRect(contours[k]);
        // find the centroid of the object;
        centriod[k] = Point2f(monment[k].m10 / monment[k].m00, monment[k].m01 / monment[k].m00);
        // pick random color to area the regions
        colors[k] = cv::Vec3b(std::rand() % 256, std::rand() % 256, std::rand() % 256);
        // draw centroid of the object
        cv::circle(dst, centriod[k], 3, colors[k]);

        // right bottom, counterwise;
        cv::Point2f rect_points[4];
        minRect[k].points(rect_points);
        for (int j = 0; j < 4; j++)
        {
            cv::line(dst, rect_points[j], rect_points[(j + 1) % 4], colors[k]);
        }
        
        // calculating the angle;
        for (int i = 0; i < src.rows; i++)
        {
            for (int j = 0; j < src.cols; j++)
            {
                if (src.at<uchar>(i, j) != 0)
                {
                    // not the background
                    int xc = i - monment[k].m10 / (monment[k].m00 + 1e-5);
                    int yc = j - monment[k].m01 / (monment[k].m00 + 1e-5);
                    monment[k].mu02 += (i - yc) * (i - yc);
                    monment[k].mu20 += (j - xc) * (j - xc);
                    monment[k].mu11 += (i - yc) * (j - xc);
                }
            }
        }
        monment[k].mu02 /= (monment[k].m00 + 1e-5);
        monment[k].mu20 /= (monment[k].m00 + 1e-5);
        double ang = 0.5 * atan2(2 * monment[k].mu11, (monment[k].mu20 - monment[k].mu02));
        int length = std::max(minRect[k].size.height, minRect[k].size.width);
        int xtop = (monment[k].m10 / (monment[k].m00 + 1e-5)) + length / 2 * cos(ang);
        int ytop = (monment[k].m01 / (monment[k].m00 + 1e-5)) - length / 2 * sin(ang);
        int xbot = (monment[k].m10 / (monment[k].m00 + 1e-5)) - length / 2 * cos(ang);
        int ybot = (monment[k].m01 / (monment[k].m00 + 1e-5)) + length / 2 * sin(ang);

        // draw the major axis of the region.
        Point p1(xtop, ytop), p2(xbot, ybot);
        cv::line(dst, p1, p2, colors[k], 2);
    }
    return 0;
}

// similar as the previous function
int featureCompute_c(cv::Mat &src, cv::Mat &dst, std::vector<std::pair<std::string,std::vector<float>>> &featureData){
    // find the contours from the src
    std::vector<std::vector<Point>> contours;
    findContours(src, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, Point(0, 0));
    // convert to RGB
    dst = cv::Mat::zeros(src.rows, src.cols, CV_8UC3);
    for(int i = 0; i < src.rows; i++){
        for(int j = 0; j < src.cols; j++){
            dst.at<cv::Vec3b>(i, j) = {src.at<uchar>(i,j), src.at<uchar>(i,j), src.at<uchar>(i,j)};
        }
    }

    std::vector<cv::RotatedRect> minRect(contours.size());
    std::vector<cv::Moments> monment(contours.size());
    std::vector<cv::Point2f> centriod(contours.size());
    std::vector<cv::Vec3b> colors(contours.size() + 1);
    colors[0] = cv::Vec3b(0, 0, 0);

    for (int k = 0; k < contours.size(); k++){
        minRect[k] = minAreaRect(contours[k]);
        monment[k] = moments(contours[k]);
        centriod[k] = Point2f(monment[k].m10 / monment[k].m00, monment[k].m01 / monment[k].m00); 
        colors[k] = cv::Vec3b(std::rand() % 256, std::rand() % 256, std::rand() % 256);
        
        cv::circle(dst, centriod[k], 3, colors[k]);
        
        //right bottom, counterwise;
        cv::Point2f rect_points[4];
        minRect[k].points(rect_points);
        for (int j = 0; j < 4; j++){
            cv::line(dst, rect_points[j], rect_points[(j + 1) % 4], colors[k]);
        }

        for (int i = 0; i < src.rows; i++){
            for (int j = 0; j < src.cols; j++){
                if (src.at<uchar>(i, j) != 0){
                    // not the background
                    int xc = i - monment[k].m10 / monment[k].m00;
                    int yc = j - monment[k].m01 / monment[k].m00;
                    monment[k].mu02 += (i-yc)*(i-yc);
                    monment[k].mu20 += (j-xc)*(j-xc);
                    monment[k].mu11 += (i-yc)*(j-xc);
                }
            }
        }

        monment[k].mu02/=monment[k].m00;
        monment[k].mu20/=monment[k].m00;
        double angle = 0.5*atan2(2*monment[k].mu11,(monment[k].mu20-monment[k].mu02));
        int length = std::max(minRect[k].size.height, minRect[k].size.width);
        int xtop = (monment[k].m10 / monment[k].m00) + length/2 * cos(angle);
        int ytop = (monment[k].m01 / monment[k].m00) - length/2 * sin(angle);
        int xbot = (monment[k].m10 / monment[k].m00) - length/2 * cos(angle);
        int ybot = (monment[k].m01 / monment[k].m00) + length/2 * sin(angle);
        
        Point p1(xtop, ytop), p2(xbot, ybot);
        cv::line(dst, p1, p2, colors[k], 2);

        // get features for objects
        // ratio: height/width
        double ratio = std::max(minRect[k].size.height, minRect[k].size.width) / std::min(minRect[k].size.height, minRect[k].size.width);

        // percentage of the area:
        // region/(height*width)
        double percentFill = monment[k].m00 / (minRect[k].size.height * minRect[k].size.width);

        std::string objectLabel;
        std::string iter;
        if(k == 0){
            iter = "1st";
        } else if (k == 1){
            iter = "2nd";
        } else if (k == 2){
            iter = "3rd";
        } else if (k == 3){
            iter = "4th";
        }
        std::cout << "Please label your " << iter << " object from the image." << std::endl;
        std::cin >> objectLabel;
        std::vector<float> feature;
        feature.push_back(ratio);
        feature.push_back(percentFill);
        featureData.push_back(std::make_pair(objectLabel, feature));
    }
    return 0;
}

// similar as the previous two functions
int featureCompute_d(cv::Mat &src, std::string &objectLabel, cv::Mat &dst, std::vector<std::pair<std::string,std::vector<float>>> &featureData){
    // find the contours from the src
    std::vector<std::vector<Point>> contours;
    findContours(src, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, Point(0, 0));
    // convert to RGB
    dst = cv::Mat::zeros(src.rows, src.cols, CV_8UC3);
    for(int i = 0; i < src.rows; i++){
        for(int j = 0; j < src.cols; j++){
            dst.at<cv::Vec3b>(i, j) = {src.at<uchar>(i,j), src.at<uchar>(i,j), src.at<uchar>(i,j)};
        }
    }

    std::vector<cv::RotatedRect> minRect(contours.size());
    std::vector<cv::Moments> monment(contours.size());
    std::vector<cv::Point2f> centriod(contours.size());
    std::vector<cv::Vec3b> colors(contours.size() + 1);
    colors[0] = cv::Vec3b(0, 0, 0);

    for (int k = 0; k < contours.size(); k++){
        minRect[k] = minAreaRect(contours[k]);
        monment[k] = moments(contours[k]);
        centriod[k] = Point2f(monment[k].m10 / monment[k].m00, monment[k].m01 / monment[k].m00); 
        colors[k] = cv::Vec3b(std::rand() % 256, std::rand() % 256, std::rand() % 256);
        
        cv::circle(dst, centriod[k], 3, colors[k]);
        
        //right bottom, counterwise;
        cv::Point2f rect_points[4];
        minRect[k].points(rect_points);
        for (int j = 0; j < 4; j++){
            cv::line(dst, rect_points[j], rect_points[(j + 1) % 4], colors[k]);
        }

        for (int i = 0; i < src.rows; i++){
            for (int j = 0; j < src.cols; j++){
                if (src.at<uchar>(i, j) != 0){
                    // not the background
                    int xc = i - monment[k].m10 / monment[k].m00;
                    int yc = j - monment[k].m01 / monment[k].m00;
                    monment[k].mu02 += (i-yc)*(i-yc);
                    monment[k].mu20 += (j-xc)*(j-xc);
                    monment[k].mu11 += (i-yc)*(j-xc);
                }
            }
        }

        monment[k].mu02/=monment[k].m00;
        monment[k].mu20/=monment[k].m00;
        double angle = 0.5*atan2(2*monment[k].mu11,(monment[k].mu20-monment[k].mu02));
        int length = std::max(minRect[k].size.height, minRect[k].size.width);
        int xtop = (monment[k].m10 / monment[k].m00) + length/2 * cos(angle);
        int ytop = (monment[k].m01 / monment[k].m00) - length/2 * sin(angle);
        int xbot = (monment[k].m10 / monment[k].m00) - length/2 * cos(angle);
        int ybot = (monment[k].m01 / monment[k].m00) + length/2 * sin(angle);
        
        Point p1(xtop, ytop), p2(xbot, ybot);
        cv::line(dst, p1, p2, colors[k], 2);

        // get features for objects
        // ratio: height/width
        // double ratio = minRect[k].size.height / minRect[k].size.width;
        double ratio = std::max(minRect[k].size.height, minRect[k].size.width) / std::min(minRect[k].size.height, minRect[k].size.width);
        // percentage of the area:
        // region/(height*width)
        double percentFill = monment[k].m00 / (minRect[k].size.height * minRect[k].size.width);

        std::vector<float> feature;
        feature.push_back(ratio);
        feature.push_back(percentFill);
        featureData.push_back(std::make_pair(objectLabel, feature));
    }
    return 0;
}

// draw color connected components
int drawfeatures(cv::Mat &src, cv::Mat &dst, double &num)
{
    cv::Mat imgLabels, imgStats, imgCentriods;
    dst = cv::Mat::zeros(src.rows, src.cols, CV_8UC3);
    // find the connected components with its stats by 8 connected growing.
    int imgLabelNum = cv::connectedComponentsWithStats(src, imgLabels, imgStats, imgCentriods, 8);
    int maxArea = 0;

    // get components' area in order to find the biggest three objects.
    std::vector<int> imgStatsArea;
    for (int i = 0; i < imgStats.rows; i++)
    {
        imgStatsArea.push_back(imgStats.at<int>(i, CC_STAT_AREA));
    }
    sort(imgStatsArea.begin(), imgStatsArea.end(), cmpInt);

    std::vector<cv::Vec3b> colors(imgLabelNum + 1);
    // set the background remain black
    colors[0] = cv::Vec3b(0, 0, 0);
    for (int i = 1; i < imgLabelNum; i++)
    {
        // set to random colors and find the biggest 6 areas.
        colors[i] = cv::Vec3b(std::rand() % 256, std::rand() % 256, std::rand() % 256);
        // colors[i] = cv::Vec3b(255, 255, 255);
        if (imgStats.at<int>(i, CC_STAT_AREA) <= imgStatsArea[5]|| imgStatsArea[i] < num)
            colors[i] = cv::Vec3b(0, 0, 0);
    }

    for (int x = 0; x < src.rows; x++)
    {
        for (int y = 0; y < src.cols; y++)
        {
            int label = imgLabels.at<int>(x, y);
            dst.at<cv::Vec3b>(x, y) = colors[label];
        }
    }

    return 0;
}

// store data to database (csv file)
int collectData(std::vector<std::pair<std::string,std::vector<float>>> &featureData){
    for(int i = 0; i < featureData.size(); i++){
        std::string str = "C:\\Users\\sidaz\\Desktop\\5330computerVision\\Projects\\project3\\data\\objectFeatures.csv";
        const char *csv = str.c_str();
        char *csvLoc;
        csvLoc = (char *)csv;

        const char *c = featureData[i].first.c_str();
        char *objectLabel;
        objectLabel = (char *)c;
        append_image_data_csv(csvLoc, objectLabel, featureData[i].second, false);
    }
    return 0;
}

// calculate distance Metric matching
int distanceMetric(cv::Mat &src, std::string &output, double &num){
    // get new image first

    // read features from the database.
    char csv[256];
    std::vector<char *> objectLabels;
    std::vector<std::vector<float>> dbFeatureData;
    strcpy(csv, "C:\\Users\\sidaz\\Desktop\\5330computerVision\\Projects\\project3\\data\\objectFeatures.csv");
    read_image_data_csv(csv, objectLabels, dbFeatureData, false);
    
    // threshold and clean up the new image.
    cv::Mat threshold, morpho, connectedComponent;
    double ratio, percentFill;
    thresholding(src, 60, threshold);
    cleanup(threshold, morpho);
    ccAnalysis(morpho, connectedComponent, num);
    
    // find the contours of the new image.
    std::vector<std::vector<Point>> contours;
    findContours(connectedComponent, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, Point(0, 0));
    std::cout << contours.size() << " obejct(s) found from the camera: "<< std::endl;

    std::vector<cv::RotatedRect> minRect(contours.size());
    std::vector<cv::Moments> monment(contours.size());


    for(int k = 0; k < contours.size(); k++){
        std::vector<std::pair<std::string, float>> compareDist;
        minRect[k] = minAreaRect(contours[k]);
        monment[k] = moments(contours[k]);
        
        ratio = std::max(minRect[k].size.height, minRect[k].size.width) / std::min(minRect[k].size.height, minRect[k].size.width);
        percentFill = monment[k].m00 / (minRect[k].size.height * minRect[k].size.width);

        
        std::vector<float> distanceRatio, standardRatio, distancePercent, standardPercent;
        for(int i = 0; i < dbFeatureData.size(); i++){
            // we tried to use sqrt but it returns intergers only..
            // so we had to push the values into a vector first
            // then read the values from the vector in the next for-loop
            // more explanations on the report.
            distanceRatio.push_back(std::abs(ratio - dbFeatureData[i][0]));
            standardRatio.push_back(std::sqrt(ratio - ((ratio+dbFeatureData[i][0])/2) * (ratio - ((ratio+dbFeatureData[i][0])/2)) +
             (dbFeatureData[i][0] - ((ratio+dbFeatureData[i][0])/2)) * (dbFeatureData[i][0] - ((ratio+dbFeatureData[i][0])/2))));

            distancePercent.push_back(std::abs(percentFill - dbFeatureData[i][1]));
            standardPercent.push_back(std::sqrt(percentFill - ((percentFill+dbFeatureData[i][1])/2) * (percentFill - ((ratio+dbFeatureData[i][1])/2)) +
             (dbFeatureData[i][1] - ((percentFill+dbFeatureData[i][1])/2)) * (dbFeatureData[i][1] - ((percentFill+dbFeatureData[i][1])/2))));
        }
        for(int x = 0; x < distanceRatio.size(); x++){
            double distanceM = 0;  
            distanceM = distanceRatio[x] / standardRatio[x];
            distanceM += distancePercent[x] / standardPercent[x];

            compareDist.push_back(std::make_pair(objectLabels[x], distanceM));
        }
        sort(compareDist.begin(), compareDist.end(), cmpPair);
        std::cout << "The Euclidean's best match for this object is " << compareDist[0].first << std::endl;
        output=compareDist[0].first;
    }
    return 0;
}

// calculate kNearestNeighbor matching by compare its frequency by 3.
// citation: https://www.geeksforgeeks.org/k-nearest-neighbours/
int kNearestNeighbor(cv::Mat &src, std::string &output, double &num){
    // read features from the database.
    char csv[256];
    std::vector<char *> objectLabels;
    std::vector<std::vector<float>> dbFeatureData;
    strcpy(csv, "C:\\Users\\sidaz\\Desktop\\5330computerVision\\Projects\\project3\\data\\objectFeatures.csv");
    read_image_data_csv(csv, objectLabels, dbFeatureData, false);

    // threshold and clean up the new image.
    cv::Mat threshold, morpho, connectedComponent;
    double ratio, percentFill;
    thresholding(src, 60, threshold);
    cleanup(threshold, morpho);
    ccAnalysis(morpho, connectedComponent, num);

    // find the contours of the new image.
    std::vector<std::vector<Point>> contours;
    findContours(connectedComponent, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, Point(0, 0));
    std::cout << contours.size() << " obejct(s) found from the camera: "<< std::endl;

    std::vector<cv::RotatedRect> minRect(contours.size());
    std::vector<cv::Moments> monment(contours.size());

    for (int k = 0; k < contours.size(); k++)
    {
        std::vector<std::pair<std::string, float>> compareDist;
        minRect[k] = minAreaRect(contours[k]);
        monment[k] = moments(contours[k]);

        ratio = std::max(minRect[k].size.height, minRect[k].size.width) / std::min(minRect[k].size.height, minRect[k].size.width);
        percentFill = monment[k].m00 / (minRect[k].size.height * minRect[k].size.width);

        std::vector<float> distanceRatio, standardRatio, distancePercent, standardPercent;
        for (int i = 0; i < dbFeatureData.size(); i++)
        {
            distanceRatio.push_back(std::abs(ratio - dbFeatureData[i][0]));
            standardRatio.push_back(std::sqrt(ratio - ((ratio + dbFeatureData[i][0]) / 2) * (ratio - ((ratio + dbFeatureData[i][0]) / 2)) +
                                              (dbFeatureData[i][0] - ((ratio + dbFeatureData[i][0]) / 2)) * (dbFeatureData[i][0] - ((ratio + dbFeatureData[i][0]) / 2))));

            distancePercent.push_back(std::abs(percentFill - dbFeatureData[i][1]));
            standardPercent.push_back(std::sqrt(percentFill - ((percentFill + dbFeatureData[i][1]) / 2) * (percentFill - ((ratio + dbFeatureData[i][1]) / 2)) +
                                                (dbFeatureData[i][1] - ((percentFill + dbFeatureData[i][1]) / 2)) * (dbFeatureData[i][1] - ((percentFill + dbFeatureData[i][1]) / 2))));
        }
        for (int x = 0; x < distanceRatio.size(); x++)
        {
            double distanceM = 0;
            distanceM = distanceRatio[x] / standardRatio[x];
            distanceM += distancePercent[x] / standardPercent[x];

            compareDist.push_back(std::make_pair(objectLabels[x], distanceM));
        }
        sort(compareDist.begin(), compareDist.end(), cmpPair);

        //Boyer–Moore majority vote algorithm to find most freq object.
        int freq=1;
        int freqMax=0;
        std::string MatchingLabel;
        MatchingLabel = compareDist[0].first;
        int i = 1, count = 1;
		while (i < 3) // knn where k is 3
		{
			if (compareDist[i].first == MatchingLabel) {
				count++;
			}
			else
			{
				count--;
				if (count < 0) {
					MatchingLabel = compareDist[i].first;
					count = 1;
				}
			}
            if(i==2&&count==1)
            {
                MatchingLabel = compareDist[0].first;
            }
			i++;
		}
        std::cout << "The Knn's best match for this object is " << MatchingLabel << std::endl;
        output = MatchingLabel;
    }
    return 0;
}