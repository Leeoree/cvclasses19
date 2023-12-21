/* FAST corner detector algorithm implementation.
 * @file
 * @date 2018-10-16
 * @author Anonymous
 */

#include "cvlib.hpp"

#include <ctime>
#include <random>

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

    for (int i = radius; i < _image.rows - radius; i++)
    {
        for (int j = radius; j < _image.cols - radius; j++)
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

void corner_detector_fast::compute(cv::InputArray image, std::vector<cv::KeyPoint>& keypoints, cv::OutputArray descriptors)
{
    cv::Mat _image;
    image.getMat().copyTo(_image);
    cv::cvtColor(_image, _image, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(_image, _image, cv::Size(5, 5), 0, 0);

    const int diam = 25;
    const int desc_length = 16;

    if (pixel_pairs.empty())
    {
        gen_pixel_pairs(diam, desc_length * 16);
    }

    descriptors.create(static_cast<int>(keypoints.size()), desc_length, CV_16U);
    auto desc_mat = descriptors.getMat();
    desc_mat.setTo(0);

    int rad = diam / 2 + 1;
    cv::copyMakeBorder(_image, _image, rad, rad, rad, rad, cv::BORDER_REPLICATE);
    uint16_t* ptr = reinterpret_cast<uint16_t*>(desc_mat.ptr());

    uint8_t pixel_1_buf;
    uint8_t pixel_2_buf;

    for (auto featPoint : keypoints)
    {
        featPoint.pt.x += rad;
        featPoint.pt.y += rad;

        int pair_idx = 0;
        for (int desc_idx = 0; desc_idx < desc_length; desc_idx++)
        {
            uint16_t descrpt = 0;
            for (int bit_idx = 0; bit_idx < 2 * 8; bit_idx++)
            {
                pixel_1_buf = _image.at<uint8_t>(featPoint.pt + pixel_pairs[pair_idx]);
                pixel_2_buf = _image.at<uint8_t>(featPoint.pt + pixel_pairs[pair_idx + 1]);

                descrpt |= (pixel_1_buf < pixel_2_buf) << (15 - bit_idx);
                pair_idx += 2;
            }
            *ptr = descrpt;
            ++ptr;
        }
    }
}

void corner_detector_fast::gen_pixel_pairs(int diam, int pair_num)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distribution(-diam / 2, diam / 2);
    int x1, y1, x2, y2;

    for (int i = 0; i < pair_num; i++)
    {
        x1 = distribution(gen);
        y1 = distribution(gen);
        x2 = distribution(gen);
        y2 = distribution(gen);

        pixel_pairs.push_back(cv::Point(x1, y1));
        pixel_pairs.push_back(cv::Point(x2, y2));
    }
}

void corner_detector_fast::detectAndCompute(cv::InputArray image, cv::InputArray mask, std::vector<cv::KeyPoint>& keypoints,
                                            cv::OutputArray descriptors, bool useProvidedKeypoints /*= false*/)
{
    detect(image, keypoints);
    compute(image, keypoints, descriptors);
}
} // namespace cvlib
