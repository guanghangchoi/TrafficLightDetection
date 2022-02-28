#include "preprocess.h"
#include <vector>
#include <cstdio>
#include <cmath>

bool cvMat2CnnInput(
  const std::vector<cv::Mat> & in_imgs, const int num_rois, std::vector<float> & data, 
  float & dw, float & dh, int width_, int height_, int channel_)
{
  int width_i, height_i;
  for (int i = 0; i < num_rois; ++i) {
    // cv::Mat rgb;
    // cv::cvtColor(in_imgs.at(i), rgb, CV_BGR2RGB);
    cv::Mat resized;

    // padding image to shape (width_, height_)
    // calculate ratio = new / old
    width_i = in_imgs.at(i).cols;
    height_i = in_imgs.at(i).rows;
    // enlargement will reduce the performance of the model, so a min(1, x) is added
    double ratio = std::min(1.0, std::min((double)width_ / width_i, (double)height_ / height_i));
    int unpad_w = width_i * ratio, unpad_h = height_i * ratio;
    float pad_w = width_ - unpad_w, pad_h = height_ - unpad_h;
    pad_w /= 2;
    pad_h /= 2;
    dw = pad_w;
    dh = pad_h;
    // resize image if zoom-in is needed
    if (unpad_w != width_i || unpad_h != height_i){
      cv::resize(in_imgs.at(i), resized, cv::Size(unpad_w, unpad_h));
    }
    else resized = in_imgs.at(i);
    // make border
    int top = round(pad_h - 0.1), bottom = round(pad_h + 0.1);
    int left = round(pad_w - 0.1), right = round(pad_w + 0.1);
    cv::Mat padding;
    cv::copyMakeBorder(resized, padding, top, bottom, left, right, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));

    cv::Mat pixels;
    padding.convertTo(pixels, CV_32FC3, 1.0 / 255, 0);
    std::vector<float> img;
    if (pixels.isContinuous()) {
      img.assign((float *)pixels.datastart, (float *)pixels.dataend);
    } else {
      cv::Mat pixels_copy = pixels.clone();
      if (!pixels_copy.isContinuous()){
        printf("Cannot convert image to a continuous format.");
        return false;
      }
      else {
        img.assign((float *)pixels_copy.datastart, (float *)pixels_copy.dataend);
      }
    }

    for (int c = 0; c < channel_; ++c) {
      for (int j = 0, hw = width_ * height_; j < hw; ++j) {
        data[i * channel_ * width_ * height_ + c * hw + j] =
          img[channel_ * j + 2 - c];
      }
    }
  }
  return true;
}
