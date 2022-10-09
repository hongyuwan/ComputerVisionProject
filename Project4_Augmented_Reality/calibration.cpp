/*
  Course: Computer Vision - 5330 S22
  Project 4: Calibration and Augmented Reality
  Name: Sida Zhang and Hongyu Wan
  March 10, 2022

  This file contains all the functions for calibaration and AR system.
*/

//PART I:
// Detect and Extract Chessboard Corners
int detectDrawChessboardCorners(cv::Mat &src, cv::Size &chessboardPattern,  std::vector<cv::Point2f> &corner_set)
{   
    // !Very Helpful!!!
    // CALIB_CB flags provide much faster speed when searching for chessboard
    // https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html
    bool found = cv::findChessboardCorners(src, chessboardPattern, corner_set,
                cv::CALIB_CB_FAST_CHECK);

    if (found)
    {
        // refine corners
        cv::Mat gray;
        cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
        cv::Size searchArea(11,11);
        cv::Size zeroZone(-1,-1); //unused parameter
        cv::TermCriteria criteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.1);
        cv::cornerSubPix(gray, corner_set, searchArea, zeroZone, criteria);
        cv::drawChessboardCorners(src, chessboardPattern, corner_set, found);
    }
    cv::imshow("Calibration Window", src);
    return 0;
}

// Select Calibration Images
int selectCalibrationImages(cv::Mat &src, cv::Size &chessboardPattern,
                    std::vector<cv::Point2f> &corner_set,
                    std::vector<std::vector<cv::Vec3f> > &point_list,
                    std::vector<std::vector<cv::Point2f> > &corner_list )
{
    
    std::vector<cv::Vec3f> point_set;
    for(int i = 0; i < chessboardPattern.height; i++){
        for (int j = 0; j < chessboardPattern.width; j++){
            point_set.push_back(cv::Vec3f(j, -i, 0));
        }
    }
    point_list.push_back(point_set);
    corner_list.push_back(corner_set);

    return 0;
}

//	Calibrate the Camera
int calibrateFrame(cv::Mat &src, std::vector<std::vector<cv::Vec3f> > &point_list,
                std::vector<std::vector<cv::Point2f> > &corner_list,
                cv::Mat &camera_matrix, cv::Mat &disCoefficients,
                std::vector<cv::Mat> &rotationVec, std::vector<cv::Mat> &translationVec, int &num){

    // check if user has saved enough calibration frames;
    if(corner_list.size() >= 5){
        double pixelError = cv::calibrateCamera(point_list, corner_list, src.size(),
                              camera_matrix, disCoefficients, rotationVec, translationVec,
                              cv::CALIB_FIX_ASPECT_RATIO);
        std::cout << "Camera Matrix: " << std::endl;
        for(int i = 0; i < camera_matrix.rows; i++){
            std::cout << camera_matrix.at<double>(i, 0) << "\t";
            std::cout << camera_matrix.at<double>(i, 1) << "\t";
            std::cout << camera_matrix.at<double>(i, 2) << "\n";
        }

        std::cout << "Distortion Coefficients: " << std::endl;
        int size = disCoefficients.rows;
        for(int i = 0; i < size - 1; i++){
            std::cout << disCoefficients.at<double>(i, 0)  << "\t";
        }
        std::cout << disCoefficients.at<double>(size, 0) << std::endl;
        
        std::cout << pixelError << " Re-Projection Error found" << std::endl;
        
    } else {
        std::cout << "At least 5 calibration frames is required for this task." << std::endl;
        std::cout << "You currently only have " + std::to_string(num) + " images saved." << std::endl; 
    }

    return 0;
}

// write camera matrix and discoefficients to txt file in data folder
int writeIntrinsicParameters(cv::Mat &camera_matrix, cv::Mat &disCoefficients){
    if(camera_matrix.at<double>(0, 0) != 0){
        std::ofstream matrix_data;
        matrix_data.open("C:\\Users\\sidaz\\Desktop\\5330ComputerVision\\Projects\\project4\\data\\camera_matrix.txt");
        
        for(int i = 0; i < 3; i++){
            matrix_data << camera_matrix.at<double>(i, 0) << ", "
                        << camera_matrix.at<double>(i, 1) << ", "
                        << camera_matrix.at<double>(i, 2) << "\n";
        }
        matrix_data.close();
    }

    if(disCoefficients.at<double>(0, 0) != 0){
        std::ofstream distCoeffs_data;
        std::cout << "Writting data to files..." << std::endl;
        distCoeffs_data.open("C:\\Users\\sidaz\\Desktop\\5330ComputerVision\\Projects\\project4\\data\\dist_coeffs.txt");
        
        int size = disCoefficients.rows;

        for(int i = 0; i < size - 1; i++){
            distCoeffs_data << disCoefficients.at<double>(i, 0) << ", ";
        }
        distCoeffs_data << disCoefficients.at<double>(size, 0);
        distCoeffs_data.close();
        
        std::cout << "Finished writing data to data/camera_matrix.txt and data/dist_coeffs.txt" << std::endl;
    }

    return 0;
}

//PART II:
// read camera matrix and discoefficients to txt file from data folder
int readIntrinsicParameters(char* matrix_file, char* dist_file, cv::Mat &camera_matrix, cv::Mat &disCoefficients){
    std::ifstream matrix_data(matrix_file);
    std::ifstream dist_data(dist_file);

    std::string line;
    int i = 0;
    while(std::getline(matrix_data, line)){
        const char *lineStr = line.c_str();
        char *newLine;
        newLine = (char *)lineStr;

        camera_matrix.at<double>(i, 0) = atof( strtok(newLine, ", ") );
        camera_matrix.at<double>(i, 1) = atof( strtok(NULL, ", ") ); //NULL b/c continuing on same line
        camera_matrix.at<double>(i, 2) = atof( strtok(NULL, ", ") );
        i++;
    }

    std::getline(dist_data, line);
    int j = 0;
    const char *lineStr = line.c_str();
    char *newLine;
    newLine = (char *)lineStr;
    char *ptr = strtok(newLine, ", ");
    while(ptr != NULL){
        disCoefficients.at<double>(j, 0) = atof(ptr);
        ptr = strtok(NULL, ", ");
        j++;
    }
    if(!camera_matrix.empty() && !disCoefficients.empty()){
        std::cout << "Finished reading calibration files..." << std::endl;
        std::cout << "Execute livevid_II to work on Augmented Reality" << std::endl;
    } else {
        std::cout << "Reading Calibration files failed, please check your calibration files.";
    }

    return 0;
}

// draw 3D axes on checkerboard when pressing A
int axes3D(cv::Mat &src, cv::Mat &rotationVec, cv::Mat &translationVec, cv::Mat &camera_matrix, cv::Mat &disCoefficients)
{
    std::vector<cv::Point3f> objPoints{
        {0, 0, 0}, 
        {1, 0, 0}, 
        {0, -1, 0}, 
        {0, 0, 1}
    };
    std::vector<cv::Point2f> imgPoints;
    cv::projectPoints(objPoints, rotationVec, translationVec, camera_matrix, disCoefficients, imgPoints);
            
    cv::line(src, imgPoints[0], imgPoints[1], cv::Scalar(255, 0, 0), 2);
    cv::line(src, imgPoints[0], imgPoints[2], cv::Scalar(0, 255, 0), 2);
    cv::line(src, imgPoints[0], imgPoints[3], cv::Scalar(0, 0, 255), 2);

    return 0;
}

// draw dimond on checkerboard when pressing A
// https://people.sc.fsu.edu/~jburkardt/data/obj/octahedron.obj
int createVirtualObj(cv::Mat &src, cv::Mat &rotationVec, cv::Mat &translationVec, cv::Mat &camera_matrix, cv::Mat &disCoefficients)
{
    float x = 4;
    float y = -2.5;

    std::vector<cv::Point3f> objPoints{
        {3 + x, 0 + y, 3}, 
        {0 + x, -3 + y, 3}, 
        {-3 + x, 0 + y, 3}, 
        {0 + x, 3 + y, 3},
        {0 + x, 0 + y, 6},
        {0 + x, 0 + y, 0}
    };
    std::vector<cv::Point2f> imgPoints;
    cv::projectPoints(objPoints, rotationVec, translationVec, camera_matrix, disCoefficients, imgPoints);

    // Diamond color: Tiffiny blue
    cv::line(src, imgPoints[1], imgPoints[0], cv::Scalar(244, 209, 112), 2);
    cv::line(src, imgPoints[0], imgPoints[4], cv::Scalar(244, 209, 112), 2);
    cv::line(src, imgPoints[4], imgPoints[1], cv::Scalar(244, 209, 112), 2);
    
    cv::line(src, imgPoints[2], imgPoints[1], cv::Scalar(244, 209, 112), 2);
    cv::line(src, imgPoints[1], imgPoints[4], cv::Scalar(244, 209, 112), 2);
    cv::line(src, imgPoints[4], imgPoints[2], cv::Scalar(244, 209, 112), 2);

    cv::line(src, imgPoints[3], imgPoints[2], cv::Scalar(244, 209, 112), 2);
    cv::line(src, imgPoints[2], imgPoints[4], cv::Scalar(244, 209, 112), 2);
    cv::line(src, imgPoints[4], imgPoints[3], cv::Scalar(244, 209, 112), 2);
    
    cv::line(src, imgPoints[0], imgPoints[3], cv::Scalar(244, 209, 112), 2);
    cv::line(src, imgPoints[3], imgPoints[4], cv::Scalar(244, 209, 112), 2);
    cv::line(src, imgPoints[4], imgPoints[0], cv::Scalar(244, 209, 112), 2);
    
    cv::line(src, imgPoints[0], imgPoints[1], cv::Scalar(244, 209, 112), 2);
    cv::line(src, imgPoints[1], imgPoints[5], cv::Scalar(244, 209, 112), 2);
    cv::line(src, imgPoints[5], imgPoints[0], cv::Scalar(244, 209, 112), 2);
    
    cv::line(src, imgPoints[1], imgPoints[2], cv::Scalar(244, 209, 112), 2);
    cv::line(src, imgPoints[2], imgPoints[5], cv::Scalar(244, 209, 112), 2);
    cv::line(src, imgPoints[5], imgPoints[1], cv::Scalar(244, 209, 112), 2);
    
    cv::line(src, imgPoints[2], imgPoints[3], cv::Scalar(244, 209, 112), 2);
    cv::line(src, imgPoints[3], imgPoints[5], cv::Scalar(244, 209, 112), 2);
    cv::line(src, imgPoints[5], imgPoints[2], cv::Scalar(244, 209, 112), 2);
    
    cv::line(src, imgPoints[3], imgPoints[0], cv::Scalar(244, 209, 112), 2);
    cv::line(src, imgPoints[0], imgPoints[5], cv::Scalar(244, 209, 112), 2);
    cv::line(src, imgPoints[5], imgPoints[3], cv::Scalar(244, 209, 112), 2);

    return 0;
}

// draw icosahedron on checkerboard when pressing A
// https://people.sc.fsu.edu/~jburkardt/data/obj/icosahedron.obj
int createVirtualObj2(cv::Mat &src, cv::Mat &rotationVec, cv::Mat &translationVec, cv::Mat &camera_matrix, cv::Mat &disCoefficients, int &scale){
    float x = 4;
    float y = -2.5;

    std::vector<cv::Point3f> objPoints{
        {0 * scale + x, -0.525731f * scale + y, 0.850651f * scale},
        {0.850651f * scale + x, 0 * scale + y, 0.525731f * scale},
        {0.850651f * scale + x, 0 * scale + y, -0.525731f * scale},
        {-0.850651f * scale + x, 0 * scale + y, -0.525731f * scale},
        {-0.850651f * scale + x, 0 * scale + y, 0.525731f * scale},
        {-0.525731f * scale + x, 0.850651f * scale + y, 0 * scale},
        {0.525731f * scale + x, 0.850651f * scale + y, 0 * scale},
        {0.525731f * scale + x, -0.850651f * scale + y, 0 * scale},
        {-0.525731f * scale + x, -0.850651f * scale + y, 0 * scale},
        {0 * scale + x, -0.525731f * scale + y, -0.850651f * scale},
        {0 * scale + x, 0.525731f * scale + y, -0.850651f * scale},
        {0 * scale + x, 0.525731f * scale + y, 0.850651f * scale}
    };

    std::vector<cv::Point2f> imgPoints;
    cv::projectPoints(objPoints, rotationVec, translationVec, camera_matrix, disCoefficients, imgPoints);

    // Dice color: greenish
    cv::line(src, imgPoints[1], imgPoints[2], cv::Scalar(17, 83, 58), 2);
    cv::line(src, imgPoints[2], imgPoints[6], cv::Scalar(17, 83, 58), 2);
    cv::line(src, imgPoints[6], imgPoints[1], cv::Scalar(17, 83, 58), 2);
    
    cv::line(src, imgPoints[1], imgPoints[7], cv::Scalar(17, 83, 58), 2);
    cv::line(src, imgPoints[7], imgPoints[2], cv::Scalar(17, 83, 58), 2);
    cv::line(src, imgPoints[2], imgPoints[1], cv::Scalar(17, 83, 58), 2);

    cv::line(src, imgPoints[3], imgPoints[4], cv::Scalar(17, 83, 58), 2);
    cv::line(src, imgPoints[4], imgPoints[5], cv::Scalar(17, 83, 58), 2);
    cv::line(src, imgPoints[5], imgPoints[3], cv::Scalar(17, 83, 58), 2);
    
    cv::line(src, imgPoints[4], imgPoints[3], cv::Scalar(17, 83, 58), 2);
    cv::line(src, imgPoints[3], imgPoints[8], cv::Scalar(17, 83, 58), 2);
    cv::line(src, imgPoints[8], imgPoints[4], cv::Scalar(17, 83, 58), 2);
    
    cv::line(src, imgPoints[6], imgPoints[5], cv::Scalar(17, 83, 58), 2);
    cv::line(src, imgPoints[5], imgPoints[11], cv::Scalar(17, 83, 58), 2);
    cv::line(src, imgPoints[11], imgPoints[6], cv::Scalar(17, 83, 58), 2);

    cv::line(src, imgPoints[5], imgPoints[6], cv::Scalar(17, 83, 58), 2);
    cv::line(src, imgPoints[6], imgPoints[10], cv::Scalar(17, 83, 58), 2);
    cv::line(src, imgPoints[10], imgPoints[5], cv::Scalar(17, 83, 58), 2);
    
    cv::line(src, imgPoints[9], imgPoints[10], cv::Scalar(17, 83, 58), 2);
    cv::line(src, imgPoints[10], imgPoints[2], cv::Scalar(17, 83, 58), 2);
    cv::line(src, imgPoints[2], imgPoints[9], cv::Scalar(17, 83, 58), 2);
    
    cv::line(src, imgPoints[10], imgPoints[9], cv::Scalar(17, 83, 58), 2);
    cv::line(src, imgPoints[9], imgPoints[3], cv::Scalar(17, 83, 58), 2);
    cv::line(src, imgPoints[3], imgPoints[10], cv::Scalar(17, 83, 58), 2);
    
    cv::line(src, imgPoints[7], imgPoints[8], cv::Scalar(17, 83, 58), 2);
    cv::line(src, imgPoints[8], imgPoints[9], cv::Scalar(17, 83, 58), 2);
    cv::line(src, imgPoints[9], imgPoints[7], cv::Scalar(17, 83, 58), 2);
    
    cv::line(src, imgPoints[8], imgPoints[7], cv::Scalar(17, 83, 58), 2);
    cv::line(src, imgPoints[7], imgPoints[0], cv::Scalar(17, 83, 58), 2);
    cv::line(src, imgPoints[0], imgPoints[8], cv::Scalar(17, 83, 58), 2);

    cv::line(src, imgPoints[11], imgPoints[0], cv::Scalar(17, 83, 58), 2);
    cv::line(src, imgPoints[0], imgPoints[1], cv::Scalar(17, 83, 58), 2);
    
    cv::line(src, imgPoints[0], imgPoints[11], cv::Scalar(17, 83, 58), 2);
    cv::line(src, imgPoints[11], imgPoints[4], cv::Scalar(17, 83, 58), 2);
    cv::line(src, imgPoints[4], imgPoints[0], cv::Scalar(17, 83, 58), 2);
    
    cv::line(src, imgPoints[6], imgPoints[2], cv::Scalar(17, 83, 58), 2);
    cv::line(src, imgPoints[2], imgPoints[10], cv::Scalar(17, 83, 58), 2);
    cv::line(src, imgPoints[10], imgPoints[6], cv::Scalar(17, 83, 58), 2);
    
    cv::line(src, imgPoints[1], imgPoints[6], cv::Scalar(17, 83, 58), 2);
    cv::line(src, imgPoints[6], imgPoints[11], cv::Scalar(17, 83, 58), 2);
    cv::line(src, imgPoints[11], imgPoints[1], cv::Scalar(17, 83, 58), 2);
    
    cv::line(src, imgPoints[3], imgPoints[5], cv::Scalar(17, 83, 58), 2);
    cv::line(src, imgPoints[5], imgPoints[10], cv::Scalar(17, 83, 58), 2);
    cv::line(src, imgPoints[10], imgPoints[3], cv::Scalar(17, 83, 58), 2);
    
    cv::line(src, imgPoints[5], imgPoints[4], cv::Scalar(17, 83, 58), 2);
    cv::line(src, imgPoints[4], imgPoints[11], cv::Scalar(17, 83, 58), 2);
    cv::line(src, imgPoints[11], imgPoints[5], cv::Scalar(17, 83, 58), 2);
    
    cv::line(src, imgPoints[2], imgPoints[7], cv::Scalar(17, 83, 58), 2);
    cv::line(src, imgPoints[7], imgPoints[9], cv::Scalar(17, 83, 58), 2);
    cv::line(src, imgPoints[9], imgPoints[2], cv::Scalar(17, 83, 58), 2);

    cv::line(src, imgPoints[7], imgPoints[1], cv::Scalar(17, 83, 58), 2);
    cv::line(src, imgPoints[1], imgPoints[0], cv::Scalar(17, 83, 58), 2);
    cv::line(src, imgPoints[0], imgPoints[7], cv::Scalar(17, 83, 58), 2);
    
    cv::line(src, imgPoints[3], imgPoints[9], cv::Scalar(17, 83, 58), 2);
    cv::line(src, imgPoints[9], imgPoints[8], cv::Scalar(17, 83, 58), 2);
    cv::line(src, imgPoints[8], imgPoints[3], cv::Scalar(17, 83, 58), 2);
    
    cv::line(src, imgPoints[4], imgPoints[8], cv::Scalar(17, 83, 58), 2);
    cv::line(src, imgPoints[8], imgPoints[0], cv::Scalar(17, 83, 58), 2);
    cv::line(src, imgPoints[0], imgPoints[4], cv::Scalar(17, 83, 58), 2);
    return 0;
}


int harrisCorner(cv::Mat &src){
    cv::Mat greyscale, dst;
    cv::cvtColor(src, greyscale, cv::COLOR_RGB2GRAY);

    dst = cv::Mat::zeros(src.size(), CV_32FC1);

    int block = 2;      //	Neighborhood size
    int ksize = 3;      //	Parameter size
    double k = 0.04;    //	Harris detector free parameter
    double thresh = 0.001;

    cv::cornerHarris(greyscale, dst, block, ksize, k);

    for( int i = 0; i < src.rows ; i++){
        for( int j = 0; j < src.cols; j++ ){
            if(dst.at<float>(i,j) > thresh ){
                //draw a marker here
                cv::circle(src, cv::Point(j,i), 3, cv::Scalar(0, 0, 255), 1, 4, 0 );
            }
        }
    }
    return 0;
}

// instead of using glm, we parsed the value to cv:Vec3f.
// please check out this site for parsing data.
// http://www.opengl-tutorial.org/beginners-tutorials/tutorial-7-model-loading/
int loadObj(char *path, std::vector<cv::Vec3f> &vertices, std::vector<int> &faces){
    FILE * file = fopen(path, "r");
	if( file == NULL ){
		printf("Could not open the .obj file.\n");
		getchar();
		return false;
	}

    while(1){
        char lineHeader[128];
		// read the first word of the line
		int res = fscanf(file, "%s", lineHeader);
		if (res == EOF)
			break; // EOF = End Of File. Quit the loop.
            
        if ( strcmp( lineHeader, "v" ) == 0 ){
			cv::Vec3f vertex;
			fscanf(file, "%f %f %f/n", &vertex[0], &vertex[1], &vertex[2]);
			vertices.push_back(vertex);
		} else if ( strcmp( lineHeader, "f" ) == 0 ){
			unsigned int vertexIndex[3];
			int matches = fscanf(file, "%d %d %d/n", &vertexIndex[0], &vertexIndex[1], &vertexIndex[2]);
			if (matches != 3){
				printf("File can't be parsered.\n");
				fclose(file);
				return false;
			}
            faces.push_back(vertexIndex[0]);
            faces.push_back(vertexIndex[1]);
            faces.push_back(vertexIndex[2]);
		}
    }
    return 0;
}

// https://people.sc.fsu.edu/~jburkardt/data/obj/cessna.obj
// cv::line connects all the vertices to make faces to the object.
int createVirtualObjExtension(cv::Mat &src, cv::Mat &rotationVec, cv::Mat &translationVec, cv::Mat &camera_matrix, cv::Mat &disCoefficients, int &arg)
{
    // arg 0 is video and arg 1 is static image
    std::vector<cv::Vec3f> vertices;
    std::vector<int> faces;

    // https://people.sc.fsu.edu/~jburkardt/data/obj/cessna.obj
    loadObj("C:\\Users\\sidaz\\Desktop\\5330ComputerVision\\Projects\\project4\\data\\aircraft.obj", vertices, faces);

    std::vector<cv::Point2f> imgPoints;
    cv::projectPoints(vertices, rotationVec, translationVec, camera_matrix, disCoefficients, imgPoints);

    // aircraft camouflage colours: reddish, yellowish, greenish
    for(int i = 0; i < faces.size(); i+=3){
        cv::line(src, imgPoints[faces[i] - 1], imgPoints[faces[i + 1] - 1], cv::Scalar(57, 68, 96), 2);
        cv::line(src, imgPoints[faces[i + 1] - 1], imgPoints[faces[i + 2] - 1], cv::Scalar(117, 154, 158), 2);
        cv::line(src, imgPoints[faces[i + 2] - 1], imgPoints[faces[i] - 1], cv::Scalar(59, 83, 65), 2);
    }

    if(arg == 1){
        cv::imshow("Static Image", src);
    }
    return 0;
}

// extensions: reading static image
int readStaticImageARsystem(char* staticImageme, cv::Mat &camera_matrix, cv::Mat &disCoefficients){
    cv::Mat src = cv::imread(staticImageme);

    cv::Size chessboardPattern(9,6);

    std::vector<cv::Point2f> corner_set;
    std::vector<cv::Point3f> point_set;
    for(int i = 0; i < chessboardPattern.height; i++){
        for (int j = 0; j < chessboardPattern.width; j++){
            point_set.push_back(cv::Vec3f(j, -i, 0));
        }
    }
    
    cv::Mat rotationVec, translationVec;

    bool found = cv::findChessboardCorners(src, chessboardPattern, corner_set);

    if (found)
    {
        cv::solvePnP(point_set, corner_set, camera_matrix, disCoefficients, rotationVec, translationVec);
        
        // 1 as image 0 as video;
        int arg = 1;
        createVirtualObjExtension(src, rotationVec, translationVec, camera_matrix, disCoefficients, arg);
    }
    return 0;
}