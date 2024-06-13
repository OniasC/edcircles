#include "EDLib.h"

#include <iostream>
#include <filesystem>
#include "opencv2/ximgproc.hpp"
#include "opencv2/imgcodecs.hpp"

//using namespace cv;
//using namespace std;
//using namespace cv::ximgproc;

std::vector<cv::Vec3f> houghCirclesGet(const cv::Mat& img)
{
    cv::Mat gray = img;
    std::vector<cv::Vec3f> circles;
    //std::cout << "num of channels: " << img.channels() << std::endl;
    //cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    cv::medianBlur(gray, gray, 5);

    cv::HoughCircles(gray, circles, cv::HOUGH_GRADIENT, 1,
                gray.rows/16, // change this value to detect circles with different distances to each other
                100, 30, 1, 0 // change the last two parameters
                // (min_radius & max_radius) to detect larger circles
                );
    return circles;
}

int main(int argc, char** argv)
{

    std::string inputDir = "datasets/bases_pics";
    std::string outputDir = "datasets/output_pics";
    std::string outputDirOpenCv = "datasets/output_opencv";
    std::string outputVideo = "datasets/src_edc_hough.avi";

    uint32_t frameCount = 0UL;
    std::vector<double> frameTimes;
    for (const auto & entry : std::filesystem::directory_iterator(inputDir))
    {
        cv::Mat testImg = cv::imread(entry.path(), 0);
        cv::TickMeter tm;

        std::cout << "frame: " << std::to_string(frameCount) << std::endl;
        //cv::Ptr<cv::ximgproc::EdgeDrawing> ed = cv::ximgproc::createEdgeDrawing();
        //ed->params.EdgeDetectionOperator = cv::ximgproc::EdgeDrawing::SOBEL;
        //ed->params.GradientThresholdValue = 36;
        //ed->params.AnchorThresholdValue = 8;


        std::vector<cv::Vec6d> ellipsesOpenCv;
        std::vector<cv::Vec4f> linesOpenCv;

        //Detection of edge segments from an input image
        tm.start();
        //Call ED constructor
        ED testED = ED(testImg, SOBEL_OPERATOR, 36, 8, 1, 10, 1.0, true);

        //std::cout << "\ntestED.getEdgeImage()  (Original)  : " << tm.getTimeMilli() << endl;

        /*
        cv::Mat edgeImg0 = testED.getEdgeImage();
        cv::Mat anchImg0 = testED.getAnchorImage();
        cv::Mat gradImg0 = testED.getGradImage();

        cv::imwrite("gradImg0.png", gradImg0);
        cv::imwrite("anchImg0.png", anchImg0);
        cv::imwrite("edgeImg0.png", edgeImg0);
        */
        cv::Mat gradImg0 = testED.getGradImage();
        
        //***************************** EDLINES Line Segment Detection *****************************
        //Detection of lines segments from edge segments instead of input image
        //Therefore, redundant detection of edge segmens can be avoided
        EDLines testEDLines = EDLines(testED);
        //ed->detectLines(linesOpenCv);

        cv::Mat lineImg0 = testEDLines.getLineImage();    //draws on an empty image


        //***************************** EDCIRCLES Circle Segment Detection *****************************
        //Detection of circles from already available EDPF or ED image

        EDCircles testEDCircles = EDCircles(testEDLines);
        //ed->detectEllipses(ellipsesOpenCv);
        //std::cout << "OONIAS" << std::endl;
        std::vector<cv::Vec3f> houghCircles = houghCirclesGet(testImg);

        std::vector<mCircle> found_circles = testEDCircles.getCircles();
        std::vector<mEllipse> found_ellipses = testEDCircles.getEllipses();
        cv::Mat ellipsImg0 = cv::Mat(lineImg0.rows, lineImg0.cols, CV_8UC3, cv::Scalar::all(0));
        cv::Mat ellipsImg1 = cv::Mat(lineImg0.rows, lineImg0.cols, CV_8UC3, cv::Scalar::all(0));
        tm.stop();
        frameTimes.push_back(tm.getTimeMilli());

        for (int j = 0; j < found_circles.size(); j++)
        {
            cv::Point center((int)found_circles[j].center.x, (int)found_circles[j].center.y);
            cv::Size axes((int)found_circles[j].r, (int)found_circles[j].r);
            double angle(0.0);
            cv::Scalar color = cv::Scalar(0, 255, 0);

            cv::ellipse(ellipsImg0, center, axes, angle, 0, 360, color, 1, cv::LINE_AA);
        }

        for (int j = 0; j < found_ellipses.size(); j++)
        {
            cv::Point center((int)found_ellipses[j].center.x, (int)found_ellipses[j].center.y);
            cv::Size axes((int)found_ellipses[j].axes.width, (int)found_ellipses[j].axes.height);
            double angle = found_ellipses[j].theta * 180 / CV_PI;
            cv::Scalar color = cv::Scalar(255, 255, 0);

            cv::ellipse(ellipsImg0, center, axes, angle, 0, 360, color, 1, cv::LINE_AA);
        }

        for (size_t i = 0; i < houghCircles.size(); i++)
        {
            cv::Vec3i c = houghCircles[i];
            cv::Point center =  cv::Point(c[0], c[1]);
            // circle outline
            int radius = c[2];
            cv::circle(ellipsImg1, center, radius,  cv::Scalar(255,0,255), 3,  cv::LINE_AA);
        }

        std::string outputPath = outputDir + "/" + entry.path().filename().string();
        imwrite(outputPath, ellipsImg0);

        std::string outputPath2= outputDirOpenCv + "/" + entry.path().filename().string();
        imwrite(outputPath2, ellipsImg1);

        frameCount++;

    }

    cv::VideoWriter video;
    int frame_width = 0;
    int frame_height = 0;
    int fps = 10; // You can adjust this value according to your needs

    for (uint32_t frame = 0UL; frame < frameCount; frame++)
    {
    //for (const auto & entry : std::filesystem::directory_iterator(inputDir)) {
        std::string baseName = "test_";
        cv::Mat img1 = cv::imread(inputDir + "/" + baseName + std::to_string(frame) + ".jpg");
        cv::Mat img2 = cv::imread(outputDir + "/" + baseName + std::to_string(frame) + ".jpg");
        cv::Mat img3 = cv::imread(outputDirOpenCv + "/" + baseName + std::to_string(frame) + ".jpg");

        if (img1.empty() || img2.empty()) {
            std::cout << "Could not open or find the image" << std::endl;
            return -1;
        }

        // Concatenate the images
        cv::Mat img;
        cv::hconcat(img1, img2, img);
        cv::hconcat(img, img3, img);

        // Initialize the video writer
        if (!video.isOpened()) {
            frame_width = img.cols;
            frame_height = img.rows;
            video.open(outputVideo, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, cv::Size(frame_width, frame_height));
        }

        // Write the concatenated image to the video
        video.write(img);
    }

    video.release();
    return 0;
}

#if 0
cv::Sobel(srcImg, dstInfo->dxImg, CV_16SC1, 1, 0, 3, 1, 0, cv::BORDER_REPLICATE);
cv::Sobel(srcImg, dstInfo->dyImg, CV_16SC1, 0, 1, 3, 1, 0, cv::BORDER_REPLICATE);
#endif