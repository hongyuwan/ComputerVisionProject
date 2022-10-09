using namespace cv;

int greyscale( cv::Mat &src, cv::Mat &dst ){
    int i, j;
    dst = cv::Mat::zeros(src.rows, src.cols, CV_8UC1);
    for (i = 1; i< src.rows - 1; i++){
        for(j = 1; j < src.cols - 1; j ++){
            dst.at<uchar>(i, j) = (uchar)(src.at<cv::Vec3b>(i, j)[0] * 0.114 + src.at<cv::Vec3b>(i, j)[1] * 0.587 + src.at<cv::Vec3b>(i, j)[2] * 0.299); 
        }
    }

    return 0;
}

int blur5x5( cv::Mat &src, cv::Mat &dst ){
    int i, j, c;
    cv::Mat src2;
    src2.create(src.size(), CV_16SC3);
    dst.create(src.size(), CV_16SC3);

    cv::Vec3s result = {0, 0, 0};
    // [1, 2, 4, 2, 1]
    // [1,
    //  2,
    //  4, * 
    //  2,
    //  1]
    for (i = 2; i< src.rows - 2; i++){
        for(j = 0; j < src.cols; j++){
            for(c = 0; c < 3; c++){
                result[c] = src.at<cv::Vec3b>(i - 2, j)[c] + 
                    src.at<cv::Vec3b>(i - 1, j)[c] * 2 +
                    src.at<cv::Vec3b>(i, j)[c] * 4 + 
                    src.at<cv::Vec3b>(i + 1, j)[c] * 2 +
                    src.at<cv::Vec3b>(i + 2, j)[c];

                result[c] /= 10;
            }
            src2.at<cv::Vec3s>(i, j) = result;
        }
    }    

    for (i = 0; i< src2.rows; i++){
        for(j = 2; j < src2.cols - 2; j++){
            for(c = 0; c < 3; c++){
                result[c] = src2.at<cv::Vec3s>(i, j - 2)[c] + 
                    src2.at<cv::Vec3s>(i, j - 1)[c] * 2 +
                    src2.at<cv::Vec3s>(i, j)[c] * 4 + 
                    src2.at<cv::Vec3s>(i, j + 1)[c] * 2 +
                    src2.at<cv::Vec3s>(i, j + 2)[c];

                result[c] /= 10;
            }
            dst.at<cv::Vec3s>(i, j) = result;
        }
    }

    return 0;
}

int sobelX3x3( cv::Mat &src, cv::Mat &dst ){
    int i, j, c;
    cv::Mat src2;
    src2.create(src.size(), CV_16SC3);
    dst.create(src.size(), CV_16SC3);

    cv::Vec3s result = {0, 0, 0};
    // [1,
    //  2   * [-1, 0, 1]
    //  1]
    for (i = 1; i< src.rows - 1; i++){
        for(j = 1 ; j < src.cols - 1; j++){
            for(c = 0; c < 3; c++){
                result[c] = src.at<cv::Vec3b>(i, j - 1)[c] * - 1 + 
                    src.at<cv::Vec3b>(i, j)[c] * 0 +
                    src.at<cv::Vec3b>(i, j + 1)[c];
            }
            src2.at<cv::Vec3s>(i, j) = result;
        }
    }

    for (i = 1; i< src2.rows - 1; i++){
        for(j = 1; j < src2.cols - 1; j++){
            for(c = 0; c < 3; c++){
                result[c] = src2.at<cv::Vec3s>(i - 1, j)[c] + 
                    src2.at<cv::Vec3s>(i, j)[c] * 2 + 
                    src2.at<cv::Vec3s>(i + 1, j)[c];

                result[c] /= 4;
            }
            dst.at<cv::Vec3s>(i, j) = result;
        }
    }

    return 0;
}

int sobelY3x3( cv::Mat &src, cv::Mat &dst ){
    int i, j, c;
    cv::Mat src2;    
    src2.create(src.size(), CV_16SC3);
    dst.create(src.size(), CV_16SC3);

    cv::Vec3s result = {0, 0, 0};
    // [1,
    //  0   * [1, 2, 1]
    //  -1]
    for (i = 1; i< src.rows - 1; i++){
        for(j = 1; j < src.cols - 1; j++){
            for(c = 0; c < 3; c++){
                result[c] = src.at<cv::Vec3b>(i, j - 1)[c] + 
                    src.at<cv::Vec3b>(i, j)[c] * 2+
                    src.at<cv::Vec3b>(i, j + 1)[c];
                    
                result[c] /= 4;
            }
            src2.at<cv::Vec3s>(i, j) = result;
        }
    }

    for (i = 1; i< src2.rows - 1; i++){
        for(j = 1; j < src2.cols - 1; j++){
            for(c = 0; c < 3; c++){
                result[c] = src2.at<cv::Vec3s>(i - 1, j)[c]  + 
                    src2.at<cv::Vec3s>(i, j)[c] * 0 +
                    src2.at<cv::Vec3s>(i + 1, j)[c] * - 1;
            }
            dst.at<cv::Vec3s>(i, j) = result;
        }
    }

    return 0;
}

int magnitude( cv::Mat &sx, cv::Mat &sy, cv::Mat &src, cv::Mat &dst){
    int i, j, c;
    cv::Mat sx2 = sx;
    cv::Mat sy2 = sy;
    sx.copyTo(sx2);
    sy.copyTo(sy2);
    dst.create(src.size(), CV_16SC3);

    cv::Vec3s result = {0, 0, 0};

    for (i = 1; i< sx.rows - 1; i++){
        for(j = 1; j < sx.cols - 1; j++){
            for(c = 0; c < 3; c++){
                double x = sx2.at<cv::Vec3s>(i, j)[c];
                double y = sy2.at<cv::Vec3s>(i, j)[c];

                result[c] = std::sqrt(x*x + y*y);
            }
            dst.at<cv::Vec3s>(i, j) = result;
        }
    }
    return 0;
}

int blurQuantize( cv::Mat &src, cv::Mat &dst, int levels ){
    int i, j, c;
    cv::Mat src2 = src;
    dst.create(src.size(), CV_16SC3);

    double b = 255/levels;

    cv::Vec3s result = {0, 0, 0};

    for (i = 1; i< src2.rows - 1; i++){
        for(j = 1; j < src2.cols - 1; j++){
            for(c = 0; c < 3; c++){
                result[c] = src2.at<cv::Vec3s>(i, j)[c] / b;
                result[c] = result[c] * b;
            }
            dst.at<cv::Vec3s>(i, j) = result;
        }
    }

    return 0;
}

int cartoon(cv::Mat &src, cv::Mat&dst, int levels, int magThreshold ){
    int i, j, c;
    cv::Mat mag = src;
    cv::convertScaleAbs(mag, mag);
    cv::convertScaleAbs(dst, dst);

    for (i = 1; i< dst.rows - 1; i++){
        for(j = 1; j < dst.cols - 1; j++){
            if(mag.at<cv::Vec3b>(i, j)[0] > magThreshold &&
                mag.at<cv::Vec3b>(i, j)[1] > magThreshold &&
                mag.at<cv::Vec3b>(i, j)[2] > magThreshold ){
                dst.at<cv::Vec3b>(i, j) = {0, 0, 0};
            }
        }
    }

    return 0;
}

int brightnessAdjust(cv::Mat &src, cv::Mat&dst, int upOrDown, int adjVal){
    cv::Mat src2 = src;
    int newVal;
    if(upOrDown == 1){
        newVal = adjVal;
    } else {
        newVal = adjVal * -1;
    }
    src2.convertTo(dst, -1, 1, newVal); //decrease the brightness by the new value (signed).

    return 0;
}


int blueLineChallenge(cv::Mat &src, cv::Mat &src2, cv::Mat&dst){
    int x, y, z;

    for (x = 0; x< dst.rows; x++){
        for(y = 0; y < dst.cols; y++){
            for(z = 0; z < 3; z++){
                if(src.at<Vec3b>(x, y)[0] != 0){
                    dst.at<Vec3b>(x, y)[z] = src.at<Vec3b>(x, y)[z];
                } else {
                    dst.at<Vec3b>(x, y)[z] = src2.at<Vec3b>(x, y)[z];
                }
            }
        }
    }

    return 0;
}

int readMe(){
    std::cout
        << "=======================================  Instruction  ====================================" << std::endl
        << "Hello user, this is a live web cam with different functionalities." << std::endl
        << "Please read below menu CAREFULLY to perform different functionalities.\n" << std::endl
        << "To perform video version of the functions, please:  " << std::endl
        << "       Press the corresponding capitalized letters. \n" << std::endl
        << "Press 'Enter' to repeat this menu in the terminal." << std::endl
        << "Press 'q' or 'ESC 'to terminate the program." << std::endl
        << "Press 's' to save the frame to the current folder." << std::endl
        << "Press 'g/G' to apply greyscale filter." << std::endl
        << "Press 'h/H' to apply customized greyscale filter." << std::endl
        << "Press 'b/B' to apply two Gaussian 1x5 filters." << std::endl
        << "Press 'x/X' to apply 1x3 Sobel X filters." << std::endl
        << "Press 'y/Y' to apply 1x3 Sobel Y filters." << std::endl
        << "Press 'm/M' to apply gradient magnitude from the X/Y Sobel." << std::endl
        << "Press 'i/I' to apply blur and quantize." << std::endl
        << "    Note(*): Press 'w/W' to apply customized level to blur and to quantize." << std::endl
        << "Press 'c/C' to apply cartoonization." << std::endl
        << "    Note(*): Press 'e' to apply customized level/threshold cartoonization." << std::endl
        << "Press 'r/R' to apply customized brightness changes." << std::endl
        << "Press 'o/O' to perform a Blue Line Challenge (Tiktok)." << std::endl
        << "==========================================================================================" << std::endl
        ;

    return 0;
}