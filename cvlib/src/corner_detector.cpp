/* FAST corner detector algorithm implementation.
 * @file
 * @date 2018-10-16
 * @author Anonymous
 */

#include "cvlib.hpp"

#include <ctime>

namespace cvlib
{
// static
cv::Ptr<corner_detector_fast> corner_detector_fast::create()
{
    return cv::makePtr<corner_detector_fast>();
}

void corner_detector_fast::detect(cv::InputArray image, CV_OUT std::vector<cv::KeyPoint>& keypoints, cv::InputArray /*mask = cv::noArray()*/)
{
    keypoints.clear();

    int t = 15;
    cv::Mat _image = image.getMat();
    image.getMat().copyTo(_image);
    cv::cvtColor(_image, _image, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(_image, _image, cv::Size(5, 5), 0, 0);
    cv::copyMakeBorder(_image, _image, radius, radius, radius, radius, cv::BORDER_REPLICATE);

    for (int i = 0; i < _image.rows; i++)
    {
        for (int j = 0; j < _image.cols; j++)
        {
            if (is_corner(_image, i, j, t))
                keypoints.emplace_back(cv::Point(j, i), radius + 3);
        }
    }
}

bool corner_detector_fast::is_corner(cv::Mat& image, int i, int j, int t)
{
    std::vector<int> circle_points = std::vector<int>(16, -255);

    int count_black = 0;
    int count_white = 0;

    for (auto point_idx : first_stage_verify)
    {
        if (image.at<uchar>(cv::Point(j, i) + verify_pixels[point_idx - 1]) > image.at<uchar>(i, j) + t)
        {
            count_white++;
            circle_points[point_idx - 1] = 255;
        }
        else if (image.at<uchar>(cv::Point(j, i) + verify_pixels[point_idx - 1]) < image.at<uchar>(i, j) - t)
        {
            count_black++;
            circle_points[point_idx - 1] = 0;
        }
    }

    if ((count_black >= first_verify_threshold) || (count_white >= first_verify_threshold))
    {
        for (auto point_idx : second_stage_verify)
        {
            if (image.at<uchar>(cv::Point(j, i) + verify_pixels[point_idx - 1]) > image.at<uchar>(i, j) + t)
            {
                circle_points[point_idx - 1] = 255;
            }
            else if (image.at<uchar>(cv::Point(j, i) + verify_pixels[point_idx - 1]) < image.at<uchar>(i, j) - t)
            {
                circle_points[point_idx - 1] = 0;
            }
        }

        if (has_continuous_sequence(circle_points))
            return true;
    }

    return false;
}

bool corner_detector_fast::has_continuous_sequence(std::vector<int> seq)
{
    int countMax = 1;
    int count = 1;

    for (size_t seq_idx = 1; seq_idx < 2 * seq.size() - 2; seq_idx++)
    {
        if (seq[seq_idx % seq.size()] == seq[(seq_idx - 1) % seq.size()])
            count++;
        else
            count = 1;

        countMax = count > countMax ? count : countMax;
    }

    if (countMax >= second_verify_threshold)
        return true;
    else
        return false;
}

void corner_detector_fast::compute(cv::InputArray, std::vector<cv::KeyPoint>& keypoints, cv::OutputArray descriptors)
{
    std::srand(unsigned(std::time(0))); // \todo remove me
    // \todo implement any binary descriptor
    const int desc_length = 2;
    descriptors.create(static_cast<int>(keypoints.size()), desc_length, CV_32S);
    auto desc_mat = descriptors.getMat();
    desc_mat.setTo(0);

    int* ptr = reinterpret_cast<int*>(desc_mat.ptr());
    for (const auto& pt : keypoints)
    {
        for (int i = 0; i < desc_length; ++i)
        {
            *ptr = std::rand();
            ++ptr;
        }
    }
}

void corner_detector_fast::detectAndCompute(cv::InputArray, cv::InputArray, std::vector<cv::KeyPoint>&, cv::OutputArray descriptors, bool /*= false*/)
{
    // \todo implement me
}
} // namespace cvlib
