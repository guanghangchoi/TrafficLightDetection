/*
 * Copyright 2022 Pix Moving, Inc. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "trt.h"
#include "util.h"
#include <chrono>
#include <fstream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui.hpp>
#include <cuda_runtime_api.h>

#define VERBOSE true

bool cvMat2CnnInput(
  const cv::Mat &in_img, std::vector<float> &data, float &dw, float &dh, int width_, int height_,
  int channel_);

int main()
{
  const std::string engine_name = "../data/yolov5s.engine"; 
  std::string dir_name = "../data";
  std::ifstream file(engine_name, std::ios::binary);
  if (!file.good()){
    std::cerr << "Read " << engine_name << " failed!" << std::endl;
    return -1;
  }
  char *trtModelStream = nullptr;
  size_t size = 0;
  file.seekg(0, file.end);
  size = file.tellg();
  std::cout << "The engine has size of " << size << " . " << std::endl;
  file.seekg(0, file.beg);
  trtModelStream = new char[size];
  assert(trtModelStream);
  file.read(trtModelStream, size);
  file.close();
  
  std::vector<std::string> file_names;
  if (read_image(dir_name.c_str(), file_names) < 0){
    std::cerr << "Read image failed ! " << std::endl;
    return -1;
  }
  
  std::vector<cv::Mat> imgs;
  std::cout << "file_names : " << std::endl;
  for (int i=0; i<file_names.size();i++) {
    std::cout << file_names[i] << std::endl;
    cv::Mat img = cv::imread(file_names[i], cv::IMREAD_COLOR);
    imgs.push_back(img);
    // cv::namedWindow("test", cv::WINDOW_AUTOSIZE);
    // cv::imshow("test", img);
    // cv::waitKey(0);
    // cv::destroyAllWindows();
  }
  int width_ = 640;
  int height_ = 640;
  int channel_ = 3;
  int prob_ = 85;
  cv::Mat img_t = imgs[0];

  std::vector<float> data(width_ * height_ * channel_);
  float dw, dh;
  if (!cvMat2CnnInput(img_t, data, dw, dh, width_, height_, channel_)){
    std::cout << "Cannot preprocess image..." << std::endl;
    return -1;
  }

  // create TRT engine
  Net net(trtModelStream, size, VERBOSE);
  delete[] trtModelStream;

  // create cuda stream
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));
  
  // create cuda buffer for device
  std::vector<void*> buffer = {nullptr, nullptr, nullptr, nullptr, nullptr};
  CUDA_CHECK(cudaMalloc(&buffer[0], (channel_ * width_ * height_) * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&buffer[1], (channel_ * 80 * 80 * prob_) * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&buffer[2], (channel_ * 40 * 40 * prob_) * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&buffer[3], (channel_ * 20 * 20 * prob_) * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&buffer[4], 25200 * prob_ * sizeof(float)));
  std::cout << "Distribute cuda memory." << std::endl;
  std::cout << "Data size = " << data.size() << std::endl;
  std::cout << "Buffer[0] contains " << channel_ * width_ * height_ << " floats. " << std::endl;

  CUDA_CHECK(cudaMemcpyAsync(buffer[0], data.data(), 
                             data.size() * sizeof(float), cudaMemcpyHostToDevice, stream));
  std::cout << "Transfered data to buffer." << std::endl;
  cudaStreamSynchronize(stream);

  std::vector<float> out_prob(25200 * prob_);
  auto t_start = std::chrono::system_clock::now();
  net.infer(buffer, stream);
  CUDA_CHECK(cudaMemcpyAsync((void*)out_prob.data(), buffer[4],
                             2142000 * sizeof(float), cudaMemcpyDeviceToHost, stream));
  cudaStreamSynchronize(stream);
  auto t_end = std::chrono::system_clock::now();

  std::cout << "Time elapse : " << std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count(); 
  std::cout << " ms. " << std::endl;

  for (int i=0;i<25200*prob_;i++)
  {
    std::cout << out_prob[i] << " " << std::endl;
  }

  // release cuda buffer and stream
  cudaStreamDestroy(stream);
  for (int i=0;i<5;i++) CUDA_CHECK(cudaFree(buffer[i]));


  return 0;
}

bool cvMat2CnnInput(
  const cv::Mat &in_img, std::vector<float> &data, float &dw, float &dh, int width_, int height_,
  int channel_)
{
  int width_i, height_i;
  
  // cv::Mat rgb;
  // cv::cvtColor(in_imgs.at(i), rgb, CV_BGR2RGB);
  cv::Mat resized;

  // padding image to shape (width_, height_)
  // calculate ratio = new / old
  width_i = in_img.cols;
  height_i = in_img.rows;
  std::cout << "Get image with width = " << width_i << " and height = " << height_i << " ." << std::endl;
  // enlargement will reduce the performance of the model, so a min(1, x) is added
  double ratio = std::min(1.0, std::min((double)width_ / width_i, (double)height_ / height_i));
  int unpad_w = width_i * ratio, unpad_h = height_i * ratio;
  float pad_w = width_ - unpad_w, pad_h = height_ - unpad_h;
  pad_w /= 2;
  pad_h /= 2;
  dw = pad_w;
  dh = pad_h;
  // resize image if zoom-in is needed
  std::cout << "unpad_w = " << unpad_w << "  unpad_h = " << unpad_h << std::endl;
  if (unpad_w != width_i || unpad_h != height_i){
    cv::resize(in_img, resized, cv::Size(unpad_w, unpad_h));
    std::cout << "Resize image done." << std::endl;
  }
  else resized = in_img;
  // make border
  int top = round(pad_h - 0.1), bottom = round(pad_h + 0.1);
  int left = round(pad_w - 0.1), right = round(pad_w + 0.1);
  cv::Mat padding;
  cv::copyMakeBorder(resized, padding, top, bottom, left, right, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
  std::cout << "Get image with width (" << padding.cols << ") height (" << padding.rows << ") " << std::endl;
  cv::Mat pixels;
  padding.convertTo(pixels, CV_32FC3, 1.0 / 255, 0);
  std::vector<float> img;
  if (pixels.isContinuous()) {
    std::cout << "pixels is continue, assign to img." << std::endl;
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

  std::cout << "image size : " << img.size() << std::endl;
  // cv::namedWindow("test");
  // cv::imshow("test", pixels);
  // cv::waitKey(0);
  // cv::destroyAllWindows();

  for (int c = 0; c < channel_; ++c) {
    for (int j = 0, hw = width_ * height_; j < hw; ++j) {
      data[c * hw + j] =
        img[channel_ * j + 2 - c];
    }
  }
  
  return true;
}