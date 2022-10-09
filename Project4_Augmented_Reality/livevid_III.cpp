#include <opencv2\highgui.hpp>
#include <opencv2\videoio.hpp>
#include <opencv2\aruco.hpp>
#include <opencv2\aruco\dictionary.hpp>
#include <opencv2\aruco\charuco.hpp>
#include <opencv2\core.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <opencv2\opencv.hpp>
#include "calibration.h"

/*
  Course: Computer Vision - 5330 S22
  Project 4: Calibration and Augmented Reality
  Name: Sida Zhang and Hongyu Wan
  March 10, 2022
  
  This file using aruco to calibrate camera, detect Markers
     for Charucoboard and create virtual object on Charucoboard.
*/

using namespace std;
using namespace cv;
using namespace cv::aruco;

#define WINDOW_NAME "ARDemo"


Mat srcImage;
Mat camMatrix(3, 3, CV_64FC1);
cv::Mat distCoeffs = cv::Mat::zeros(8, 1, CV_64F);
float markerLength = 100;
int show = 3;
cv::Vec3d rvecc, tvecc;

//read CameraParameters
int readCameraParameters(Mat &camMatrix, Mat &distCoeffs)
{
	char camera_matrix_data[256];
	char dist_coeffs_data[256];
	int intervals = 0; // used for showing results in frames
	int flag = 0;	   // used for draw 3d or objects.
	DIR *dir_path;
	struct dirent *dp;
	strcpy(camera_matrix_data, "C:\\Users\\hongy\\opencv\\launch\\project4\\data\\camera_matrix.txt");
	strcpy(dist_coeffs_data, "C:\\Users\\hongy\\opencv\\launch\\project4\\data\\dist_coeffs.txt");
	readIntrinsicParameters(camera_matrix_data, dist_coeffs_data, camMatrix, distCoeffs);
}

void draw_image(Mat image, Mat &frame, Vec3d &rvec, Vec3d &tvec)
{
	vector<Point3f> points;
	vector<Point2f> imagePoints;

	float l = markerLength * 0.5f;

	points.emplace_back(-l, l, 0); // Determine the position of the four vertices
	points.emplace_back(l, l, 0);
	points.emplace_back(l, -l, 0);
	points.emplace_back(-l, -l, 0);

	// Map 3d coordinates to the image coordinate system through camera internal and external parameters
	cv::projectPoints(points, rvec, tvec, camMatrix, distCoeffs, imagePoints);

	unsigned int x = (unsigned int)imagePoints[0].x;
	unsigned int y = (unsigned int)imagePoints[0].y;

	vector<Point2f> corners;
	corners.emplace_back(0, 0);
	corners.emplace_back(image.cols, 0);
	corners.emplace_back(image.cols, image.rows);
	corners.emplace_back(0, image.rows);

	// Compute perspective transformation from four pairs of corresponding points
	cv::Mat T = getPerspectiveTransform(corners, imagePoints);

	// apply perspective transform to image
	Mat warpedImg;
	cv::warpPerspective(image, warpedImg, T, frame.size());

	vector<Point2i> pts;

	for (auto i : imagePoints)
	{
		pts.emplace_back((int)i.x, (int)i.y);
	}

	// Fill a convex polygon with all 0s
	cv::fillConvexPoly(frame, pts, cv::Scalar::all(0), cv::LINE_AA);
	// OR operation
	cv::bitwise_or(warpedImg, frame, frame);
}

int main(int argc, char *argv[])
{
	VideoCapture inputVideo(0);

	Ptr<aruco::Dictionary> dictionary =
		aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_50);
	readCameraParameters(camMatrix, distCoeffs);

	cout << camMatrix << endl;
	cout << distCoeffs << endl;

	srcImage = imread("C:\\Users\\hongy\\opencv\\launch\\project4\\peng.jpg");
	resize(srcImage, srcImage, Size(100, 100));

	namedWindow(WINDOW_NAME);

	Mat frame;

	while (inputVideo.grab())
	{
		inputVideo >> frame;
		cv::Mat image, imageCopy;
		inputVideo.retrieve(image);
		image.copyTo(imageCopy);
		vector<int> ids;
		vector<vector<Point2f>> corners;
		aruco::detectMarkers(frame, dictionary, corners, ids);
		 cv::Ptr<cv::aruco::CharucoBoard> board = cv::aruco::CharucoBoard::create(5, 5, 100, 50, dictionary);
		std::vector<int> markerIds;
		std::vector<std::vector<cv::Point2f>> markerCorners;
			//if marker more than one
		if (ids.size() > 0)
		{
			cv::aruco::drawDetectedMarkers(imageCopy, corners, ids);
			vector<Vec3d> rvecs, tvecs;
			cv::Vec3d rvec, tvec; // get rotation vector and translation vector
			//estimate board pose
			 cv::aruco::estimatePoseBoard(corners, ids, board, camMatrix, distCoeffs, rvec, tvec);
			 //estimate marker pose
			cv::aruco::estimatePoseSingleMarkers(corners, 10, camMatrix, distCoeffs, rvecs, tvecs);
			rvecc=rvecs[0];
			tvecc=tvecs[0];
			for (unsigned int i = 0; i < ids.size(); i++)
			{
				if (show == 1)
				 cv::aruco::drawAxis(imageCopy, camMatrix, distCoeffs, rvecc, tvecc, 10);
				if (show == 2)
					draw_image(srcImage, imageCopy, rvecs[i], tvecs[i]);

			}
			if (show == 3)
			{	//draw vitual object
				std::vector<cv::Vec3f> vertices;
				std::vector<int> faces;

				loadObj("C:\\Users\\hongy\\opencv\\launch\\project4\\aircraft.obj", vertices, faces);

				std::vector<cv::Point2f> imgPoints;
				cv::projectPoints(vertices, rvecc, tvecc, camMatrix, distCoeffs, imgPoints);

				// aircraft camourflag colours: redish, yellow, greenish
				for (int i = 0; i < faces.size(); i += 3)
				{
					cv::line(imageCopy, imgPoints[faces[i] - 1], imgPoints[faces[i + 1] - 1], cv::Scalar(0, 255, 0), 2);
					cv::line(imageCopy, imgPoints[faces[i + 1] - 1], imgPoints[faces[i + 2] - 1], cv::Scalar(0, 255, 0), 2);
					cv::line(imageCopy, imgPoints[faces[i + 2] - 1], imgPoints[faces[i] - 1], cv::Scalar(0, 255, 0), 2);
				}
			}
			// }
		}

		cv::imshow(WINDOW_NAME, imageCopy);

		char key = (char)waitKey(10);
		if (key == 'b')
			break;
		if (key == 'q')
			show = 1;
		if (key == 'w')
			show = 2;
		if (key == 'e')
			show = 3;
	}

	return 0;
}