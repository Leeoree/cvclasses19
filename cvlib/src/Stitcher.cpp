
#include "cvlib.hpp"
#include <cmath>

namespace cvlib
{
Stitcher::Stitcher()
{
}

void Stitcher::init(cv::Mat init_image, int ratio)
{
    init_image.copyTo(_resulting_image);

    _corn_detector = cvlib::corner_detector_fast::create();
    _descr_matcher = cvlib::descriptor_matcher(ratio);

    _corn_detector->detectAndCompute(_resulting_image, cv::Mat(), _resulting_image_corners, _resulting_image_descriptors);
}

void Stitcher::stitch(cv::Mat input_image)
{
    _corn_detector->detectAndCompute(_resulting_image, cv::Mat(), _resulting_image_corners, _resulting_image_descriptors);

    std::vector<cv::KeyPoint> input_img_corners;
    cv::Mat input_img_descriptors;

    _corn_detector->detectAndCompute(input_image, cv::Mat(), input_img_corners, input_img_descriptors);

    std::vector<std::vector<cv::DMatch>> matches;
    _descr_matcher.radiusMatch(input_img_descriptors, _resulting_image_descriptors, matches, 100.0f);

    std::vector<cv::Point2f> key_points_result, key_points_input;

    for (size_t i = 0; i < matches.size(); i++)
    {
        if (!matches[i].empty())
        {
            key_points_input.push_back(input_img_corners[matches[i][0].queryIdx].pt);
            key_points_result.push_back(_resulting_image_corners[matches[i][0].trainIdx].pt);
        }
    }
    if ((key_points_input.size() < 4) || (key_points_result.size() < 4))
    {
        return;
    }
    cv::Mat homog = cv::findHomography(cv::Mat(key_points_input), cv::Mat(key_points_result), cv::RANSAC);

    float f_x_offset = homog.at<double>(2);
    int i_x_offset_abs = (int)ceil(abs(f_x_offset));

    int new_result_img_width = _resulting_image.cols + i_x_offset_abs;
    int new_result_img_height = _resulting_image.rows;
    cv::Size new_result_size = cv::Size(new_result_img_width, new_result_img_height);
    cv::Mat new_result = cv::Mat(new_result_size, CV_8U);
    cv::warpPerspective(input_image, new_result, homog, new_result_size, cv::INTER_CUBIC);
    cv::Mat roi;

    roi = cv::Mat(new_result, cv::Rect(0, 0, _resulting_image.cols, _resulting_image.rows));
    _resulting_image.copyTo(roi);

    new_result.copyTo(_resulting_image);
}

cv::Mat Stitcher::get_result_image(void)
{
    return (_resulting_image);
}
} // namespace cvlib
