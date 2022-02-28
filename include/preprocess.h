#ifndef _PREPROCESS_H
#define _PREPROCESS_H

#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

bool cvMat2CnnInput(
  const std::vector<cv::Mat> & in_imgs, const int num_rois, std::vector<float> & data, 
  float & dw, float & dh, int width_, int height_, int channel_);

static inline int display_image(cv::Mat &image) {
  cv::namedWindow("test", cv::WINDOW_AUTOSIZE);
  cv::imshow("test", image);
  cv::waitKey(0);
  cv::destroyAllWindows();
  return 0;
}
#endif // def _PREPROCESS_H