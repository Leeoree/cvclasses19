/* Demo application for Computer Vision Library.
 * @file
 * @date 2018-09-05
 * @author Anonymous
 */

#include <cvlib.hpp>
#include <opencv2/opencv.hpp>

#include "utils.hpp"

int demo_corner_detector(int argc, char* argv[])
{
    cv::VideoCapture cap(0);
    if (!cap.isOpened())
        return -1;

    const auto main_wnd = "orig";
    const auto demo_wnd = "demo";

    cv::namedWindow(main_wnd);
    cv::namedWindow(demo_wnd);

    cv::Mat frame;
    auto detector = cvlib::corner_detector_fast::create();
    std::vector<cv::KeyPoint> corners;

    utils::fps_counter fps;
    while (cv::waitKey(30) != 27) // ESC
    {
        cap >> frame;
        cv::imshow(main_wnd, frame);

        detector->detect(frame, corners);
        cv::drawKeypoints(frame, corners, frame, cv::Scalar(0, 0, 255));
        utils::put_fps_text(frame, fps);

        std::stringstream ss;
        ss << "Detected corners: " << std::fixed << corners.size();

        cv::putText(frame, ss.str(), {frame.rows / 8, frame.cols / 8}, cv::FONT_HERSHEY_PLAIN, 0.7, cv::Scalar(0, 255, 0), 1, 8, false);

        cv::imshow(demo_wnd, frame);
    }

    cv::destroyWindow(main_wnd);
    cv::destroyWindow(demo_wnd);

    return 0;
}
