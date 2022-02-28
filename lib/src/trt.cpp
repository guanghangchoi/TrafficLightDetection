#include "trt.h"
#include <cuda_runtime.h>
#include <assert.h>

#define BATCH_SIZE 1

const char* INPUT_BLOB_NAME = "images";
const char* OUTPUT_BLOB_NAME = "output";
Net::Net(const char* engine_stream, size_t size, bool verbose)
{
    Logger logger(verbose);
    runtime = nvinfer1::createInferRuntime(logger);
    assert(runtime != nullptr);
    engine = runtime->deserializeCudaEngine(engine_stream, size);
    assert(engine != nullptr);
    context = engine->createExecutionContext();
    assert(context != nullptr);
    // std::cout << "Number of Bindings : " << engine->getNbBindings() << std::endl;
    // for (int i = 0; i < engine->getNbBindings(); i++)
    // {
    //     std::cout << "Name : " << engine->getBindingName(i) << std::endl;
    //     nvinfer1::Dims dims = engine->getBindingDimensions(i);
    //     for (int j=0; j<dims.nbDims; j++){
    //         std::cout << " Dimension " << j << " :  " << dims.d[j] << std::endl;
    //     }
    // }
}

Net::~Net()
{
    context->destroy();
    engine->destroy();
    runtime->destroy();
}

void Net::infer(std::vector<void*> & buffer, cudaStream_t stream)
{
    try {
        context->enqueue(BATCH_SIZE, buffer.data(), stream, nullptr);
        cudaStreamSynchronize(stream);
    } catch (std::exception & e){
        std::cerr << "Error : " << e.what() << std::endl;
    }
}