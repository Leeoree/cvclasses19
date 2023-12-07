/* Split and merge segmentation algorithm implementation.
 * @file
 * @date 2018-09-18
 * @author Anonymous
 */

#include "cvlib.hpp"

#include <iostream>

namespace cvlib
{

int N;
int frame_cntr;
std::vector<cv::Mat> prev_frames;

double alpha;
int _threshold;

motion_segmentation::motion_segmentation()
{
    N = 20;
    frame_cntr = 0;
    prev_frames.reserve(N);

    alpha = 0.05;
    _threshold = 0;
}

void motion_segmentation::setVarThreshold(int th)
{
    _threshold = th;
}

void motion_segmentation::apply(cv::InputArray _image, cv::OutputArray _fgmask, double)
{
    // \todo implement your own algorithm:
    //       * MinMax
    //       * Mean
    //       * 1G
    //       * GMM

    // \todo implement bg model updates

    frame_cntr = 0;
    prev_frames.clear();

    cv::Mat input;
    cv::cvtColor(_image, input, cv::COLOR_BGR2GRAY);
    input.convertTo(input, CV_32FC1);

    if (prev_frames.size() < N)
    {
        prev_frames.push_back(input);
        if (frame_cntr == 0)
            input.copyTo(bg_model_);
        else
            bg_model_ += input.mul(1 / N);
        frame_cntr++;
    }
    else
    {
        cv::Mat absdiff;
        cv::absdiff(bg_model_, input, absdiff);
        cv::threshold(absdiff, _fgmask, _threshold, 255, CV_8U);
        updateBackgroundModel(input);
    }
}

void motion_segmentation::updateBackgroundModel(cv::Mat image, cv::Mat mask)
{
    bg_model_ = (1 - alpha) * bg_model_ + alpha * image;

    bg_model_ += image.mul(1 / N);

    frame_cntr++;
    frame_cntr %= N;
    image.copyTo(prev_frames[frame_cntr]);
}
} // namespace cvlib
