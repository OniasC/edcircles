#include "EDLib.h"

#include <iostream>
#include <filesystem>
#include <numeric>
#include "opencv2/ximgproc.hpp"

#include "opencv2/imgcodecs.hpp"

//using namespace cv;
//using namespace std;
//using namespace cv::ximgproc;

#define VIDEO 1

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

cv::Vec3f averageLineGet(const cv::Mat& img, const cv::Point& pt1, const cv::Point& pt2)
{
    cv::LineIterator it(img, pt1, pt2, 8);

    // Initialize the sum of BGR values
    cv::Vec3f sum_BGR(0, 0, 0);

    // Iterate over the line
    for(int i = 0; i < it.count; i++, ++it) {
        sum_BGR += img.at<cv::Vec3b>(it.pos());
    }

    // Calculate the average BGR values
    cv::Vec3f avg_BGR = sum_BGR / it.count;

    return avg_BGR;
}

/**
 * pt1 is at top left corner
 * pt2 is bottom right corner
*/
cv::Vec3f rectangleAverageGet(const cv::Mat& img, const cv::Point& pt1, const cv::Point& pt2)
{
    int width  = (pt1.x > pt2.x) ? pt1.x - pt2.x : pt2.x - pt1.x;
    int height = (pt1.y > pt2.y) ? pt1.y - pt2.y : pt2.y - pt1.y;
    cv::Rect rect(pt1.x, pt1.y, width, height); // Example: rectangle with top-left corner at (50, 50) and size 100x100
    cv::Vec3d meanColor(0, 0, 0);
    // Ensure the rectangle is within the image boundaries
    if ((rect.x + rect.width <= img.cols) && (rect.y + rect.height <= img.rows)) {
        // Get the ROI
        cv::Mat roi = img(rect);

        // Variables to accumulate the sum of pixel values and count of non-black pixels
        cv::Vec3d sumPixel(0, 0, 0); // Double precision to avoid overflow
        int nonBlackPixelCount = 0;

        // Iterate over each pixel in the ROI
        for (int y = 0; y < roi.rows; ++y) {
            for (int x = 0; x < roi.cols; ++x) {
                cv::Vec3b pixel = roi.at<cv::Vec3b>(y, x);
                if (pixel != cv::Vec3b(0, 0, 0)) {
                    sumPixel += cv::Vec3d(pixel);
                    ++nonBlackPixelCount;
                }
            }
        }

        // Calculate the mean color, avoiding division by zero

        if (nonBlackPixelCount > 0) {
            meanColor = sumPixel / nonBlackPixelCount;
        }
    }

    return meanColor;
}


bool isWhite(const cv::Vec3f& pt)
{
    static constexpr float WHITE_THRESHOLD = 600.f;
    bool isWhitish = false;
    if ((pt[0]+pt[1]+pt[2]) > WHITE_THRESHOLD)
    {
        isWhitish = true;
    }
    //return false;
    return isWhitish;
}

bool isBlue(const cv::Vec3f& pt)
{
    bool isBlueish = false;
    if ((pt[0] > pt[1])  && (pt[1] > pt[2]))
    {
        isBlueish = true;
    }
    //return false;
    return isBlueish;
}

bool isYellow(const cv::Vec3f& pt)
{
    static constexpr float GREEN_RED_DISTANCE = 20.0f;
    bool isYellowish = false;
    if ((pt[1] > pt[0])  && (pt[2] > pt[0]))
    {
        if (std::abs(pt[1]-pt[2]) < GREEN_RED_DISTANCE)
        {
            isYellowish = true;
        }
    }
    //return false;
    return isYellowish;
}

double maxRadiusGet(const int x, const int y)
{
    //return std::min(x, y);
    return sqrt(x*x + y*y);
}

bool isInterestingCircle(const cv::Mat& polarImg, const int adjustedRadius, const float TOLERANCE, const int startLine, const int endLine)
{
    const int toleranceRadius = (int)((float)adjustedRadius*TOLERANCE);
    bool isTarget = false;
    int leftOfRadius = adjustedRadius - toleranceRadius;
    int rightOfRadius = adjustedRadius + toleranceRadius;

    // left side
    //cv::Point pt1l(leftOfRadius, startLine);  // Starting point of the line
    //cv::Point pt2l(leftOfRadius, endLine);  // Ending point of the line

    cv::Point pt1l(leftOfRadius, startLine);  // Starting point of the line
    cv::Point pt2l(adjustedRadius, endLine);  // Ending point of the line
    auto avgLeft = rectangleAverageGet(polarImg, pt1l, pt2l);
    //auto avgLeft = averageLineGet(polarImg, pt1l, pt2l);
    //std::cout << "Average BGR values Left: " << avgLeft << std::endl;

    // left side
    //cv::Point pt1r(rightOfRadius, startLine);  // Starting point of the line
    //cv::Point pt2r(rightOfRadius, endLine);  // Ending point of the line
    //auto avgRight = averageLineGet(polarImg, pt1r, pt2r);
    cv::Point pt1r(adjustedRadius, startLine);  // Starting point of the line
    cv::Point pt2r(rightOfRadius, endLine);  // Ending point of the line
    auto avgRight = rectangleAverageGet(polarImg, pt1r, pt2r);
    //std::cout << "Average BGR values Right: " << avgRight << std::endl;

    if (isWhite(avgRight) || isWhite(avgLeft))
    {
        //std::cout << "eh branco" <<std::endl;
        isTarget = false;
    }
    else if (isBlue(avgRight) && isYellow(avgLeft))
    {
        isTarget = true;
        //std::cout << "Average BGR values Left: " << avgLeft << std::endl;
        //std::cout << "Average BGR values Right: " << avgRight << std::endl;
    }
    else if (isBlue(avgLeft) && isYellow(avgRight))
    {
        isTarget = true;
        //std::cout << "Average BGR values Left: " << avgLeft << std::endl;
        //std::cout << "Average BGR values Right: " << avgRight << std::endl;
    }

    return isTarget;
}

void detectionCheck(const cv::Point2d& center,
                    const cv::Size axes,
                    const double angle,
                    const cv::Mat& originalImg,
                    const bool isCircle,
                    const std::string outputPath,
                    cv::Mat& ellipsImg)
{
    static constexpr float TOLERANCE = 0.12f;

    double maxRadius = maxRadiusGet(center.x, center.y);
    int outputWidth = static_cast<int>(maxRadius * 2 * CV_PI); // Width corresponds to angle (0 to 2PI)
    int outputHeight = static_cast<int>(maxRadius); // Height corresponds to radius (0 to maxRadius)

    int adjustedRadius = 0;
    cv::Scalar color;
    if (isCircle == true)
    {
        //adjustedRadius = (int)axes.height*originalImg.cols/maxRadius;
        adjustedRadius = (int)axes.height*outputWidth/maxRadius;
        color = cv::Scalar(0, 255, 0);
    }
    else
    {
        color = cv::Scalar(255, 255, 0);
        //adjustedRadius = (int)((axes.width + axes.height)/2)*originalImg.cols/maxRadius;
        adjustedRadius = (int)((axes.width + axes.height)/2)*outputWidth/maxRadius;

    }

    cv::Mat polarImage;


    // Convert the image to polar coordinates
    cv::Point pt1(adjustedRadius, outputHeight-10);
    cv::Point pt2(adjustedRadius, 10);

    //cv::warpPolar(originalImg, polarImage, cv::Size(originalImg.cols, originalImg.rows), center, maxRadius, cv::WARP_FILL_OUTLIERS);
    cv::warpPolar(originalImg, polarImage, cv::Size(outputWidth, outputHeight), center, maxRadius, cv::WARP_FILL_OUTLIERS);
    //std::cout << originalImg.cols << "  " << originalImg.rows << std::endl;

    bool isValid = isInterestingCircle(polarImage, adjustedRadius, TOLERANCE, 10, outputHeight-10);
    if (isValid == true)
    {
        cv::ellipse(ellipsImg, center, axes, angle, 0, 360, color, 1, cv::LINE_AA);
        cv::line(polarImage, pt1, pt2, cv::Scalar(255, 0, 0), 3, cv::LINE_8);
        imwrite(outputPath, polarImage);
    }
}

int main(int argc, char** argv)
{

    std::string inputDir = "datasets/bases_pics";
    std::string outputDir = "datasets/output_pics";
    std::string outputDirOpenCv = "datasets/output_opencv";
    std::string outputGradient = "datasets/output_gradient";
    std::string outputVideo = "grad.avi";

    uint32_t frameCount = 0UL;
    std::vector<double> frameTimes;
    std::vector<double> frameTimes1;
    uint64_t countHoughCircles = 0;
    uint64_t countEDCircles = 0;

    uint32_t MAX_FRAMES = 700UL;
    for (const auto & entry : std::filesystem::directory_iterator(inputDir))
    {
        std::cout << entry.path() << std::endl;
        cv::Mat originalImg = cv::imread(entry.path());
        //cv::cvtColor(originalImg, originalImg, cv::COLOR_BGR2HSV);
        cv::Mat testImg;
        cv::cvtColor(originalImg, testImg, cv::COLOR_BGR2GRAY);
        cv::TickMeter tm;
        cv::TickMeter tm1;

        std::cout << "frame: " << std::to_string(frameCount) << std::endl;
        //cv::Ptr<cv::ximgproc::EdgeDrawing> ed = cv::ximgproc::createEdgeDrawing();
        //ed->params.EdgeDetectionOperator = cv::ximgproc::EdgeDrawing::SOBEL;
        //ed->params.GradientThresholdValue = 36;
        //ed->params.AnchorThresholdValue = 8;
        tm1.start();
        //std::vector<cv::Vec3f> houghCircles = houghCirclesGet(testImg);
        //countHoughCircles += houghCircles.size();
        tm1.stop();
        frameTimes1.push_back(tm1.getTimeMilli());

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

        std::vector<mCircle> found_circles = testEDCircles.getCircles();
        std::vector<mEllipse> found_ellipses = testEDCircles.getEllipses();
        cv::Mat ellipsImg0 = cv::Mat(lineImg0.rows, lineImg0.cols, CV_8UC3, cv::Scalar::all(0));

        tm.stop();
        frameTimes.push_back(tm.getTimeMilli());

        static constexpr float TOLERANCE = 0.1f;
        for (int j = 0; j < found_circles.size(); j++)
        {
            cv::Point center((int)found_circles[j].center.x, (int)found_circles[j].center.y);
            cv::Size axes((int)found_circles[j].r, (int)found_circles[j].r);
            double angle(0.0);
            std::string outputPath2 = outputGradient + "/"+ entry.path().filename().string() + "circle_" + std::to_string(j) +  ".jpg" ;
            detectionCheck(center, axes, angle, originalImg, true, outputPath2, ellipsImg0);

        }

        for (int j = 0; j < found_ellipses.size(); j++)
        {
            cv::Point center((int)found_ellipses[j].center.x, (int)found_ellipses[j].center.y);
            cv::Size axes((int)found_ellipses[j].axes.width, (int)found_ellipses[j].axes.height);
            double angle = found_ellipses[j].theta * 180 / CV_PI;
            std::string outputPath2 = outputGradient + "/"+ entry.path().filename().string() + "circle_" + std::to_string(j+found_circles.size()) +  ".jpg" ;

            detectionCheck(center, axes, angle, originalImg, false, outputPath2, ellipsImg0);
        }

        countEDCircles += (found_ellipses.size() + found_circles.size());

        std::string outputPath = outputDir + "/" + entry.path().filename().string();
        imwrite(outputPath, ellipsImg0);

        std::string outputPath2= outputGradient + "/" + entry.path().filename().string();
        imwrite(outputPath2, gradImg0);

        /*if (frameCount >= MAX_FRAMES )
        {
            break;
        }
        else
        {
            frameCount++;
        }*/
        frameCount++;
        MAX_FRAMES = frameCount;

    }

    std::cout << "tempo medio hough circles: " << std::accumulate(frameTimes1.begin(), frameTimes1.end(), 0.0)/frameTimes1.size() << std::endl;
    std::cout << "tempo medio edcircles: " << std::accumulate(frameTimes.begin(), frameTimes.end(), 0.0)/frameTimes.size() << std::endl;

    std::cout << "quantidade de circulos encontrado hough circles: " << std::to_string(countHoughCircles) << std::endl;
    std::cout << "quantidade de circulos encontrado EDCircles: " << std::to_string(countEDCircles) << std::endl;


    cv::VideoWriter video;
    int frame_width = 0;
    int frame_height = 0;
    int fps = 10; // You can adjust this value according to your needs
#if VIDEO
    for (uint32_t frame = 0UL; frame < MAX_FRAMES; frame++)
    {
    //for (const auto & entry : std::filesystem::directory_iterator(inputDir)) {
        std::string baseName = "test_";
        cv::Mat img1 = cv::imread(inputDir + "/" + baseName + std::to_string(frame) + ".jpg");
        cv::Mat img2 = cv::imread(outputDir + "/" + baseName + std::to_string(frame) + ".jpg");
        //cv::Mat img3 = cv::imread(outputGradient + "/" + baseName + std::to_string(frame) + ".jpg");

        if (img1.empty() || img2.empty()) {
            std::cout << "Could not open or find the image" << std::endl;
            return -1;
        }

        // Concatenate the images
        cv::Mat img;
        cv::hconcat(img1, img2, img);
        //cv::hconcat(img, img3, img);

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
#endif
    return 0;
}

#if 0
cv::Sobel(srcImg, dstInfo->dxImg, CV_16SC1, 1, 0, 3, 1, 0, cv::BORDER_REPLICATE);
cv::Sobel(srcImg, dstInfo->dyImg, CV_16SC1, 0, 1, 3, 1, 0, cv::BORDER_REPLICATE);
#endif