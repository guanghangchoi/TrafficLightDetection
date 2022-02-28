#ifndef _TRT_H
#define _TRT_H

#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <iostream>
#include <vector>
#include <cuda_runtime_api.h>
#include <stdlib.h>
#include <stdio.h>
#include <cmath>
#include <assert.h>

#ifndef CUDA_CHECK
#define CUDA_CHECK(callstr)\
    {\
        cudaError_t error_code = callstr;\
        if (error_code != cudaSuccess) {\
            std::cerr << "CUDA error " << error_code << " at " << __FILE__ << ":" << __LINE__;\
            assert(0);\
        }\
    }
#endif  // CUDA_CHECK

class Logger : public nvinfer1::ILogger
{
public:
  Logger(bool verbose) : verbose_(verbose) {}

  void log(Severity severity, const char * msg) override
  {
    if (verbose_ || ((severity != Severity::kINFO) && (severity != Severity::kVERBOSE)))
      std::cout << msg << std::endl;
  }

private:
  bool verbose_{false};
};

class Net
{
    public:
    // create engine from engine stream
    Net(const char* engine_stream, size_t size, bool verbose);

    ~Net();

    // Infer model
    void infer(std::vector<void*> & buffer, cudaStream_t stream);

    private:
    nvinfer1::IRuntime* runtime;
    nvinfer1::ICudaEngine* engine;
    nvinfer1::IExecutionContext* context;
};
#endif // def _TRT_H