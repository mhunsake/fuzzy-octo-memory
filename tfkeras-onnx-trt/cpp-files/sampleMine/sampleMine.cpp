#include "BatchStream.h"
#include "EntropyCalibrator.h"
#include "argsParser.h"
#include "buffers.h"
#include "common.h"
#include "logger.h"
#include "parserOnnxConfig.h"

#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include "NvInfer.h"
#include <cuda_runtime_api.h>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>

// Given a serialized engine plan for inception_v3 model (channels_first),
// deserialize engine and run inference on an image

const std::string gSampleName = "TensorRT.sample_mine";

//
// !! https://forums.developer.nvidia.com/t/custom-trained-ssd-inception-model-in-tensorrt-c-version/143048/14
//
void readImage(const std::string& filename, cv::Mat &image)
{
    image = cv::imread(filename, cv::IMREAD_COLOR);
    if( image.empty() )
    {
        std::cout << "Cannot open image " << filename << std::endl;
        exit(0);
    }
    gLogInfo << filename <<   " " << image.channels() <<  "x" << image.rows<<  "x" << image.cols<< "HWC original" <<std::endl;
    cv::resize(image, image, cv::Size(299,299));
    gLogInfo << filename <<   " " << image.channels() <<  "x" << image.rows<<  "x" << image.cols<< "HWC resized" <<std::endl;
}


class SampleMine
{
    template <typename T>
    using SampleUniquePtr = std::unique_ptr<T, samplesCommon::InferDeleter>;

public:
    SampleMine(const samplesCommon::OnnxSampleParams& params)
        : mParams(params)
        , mEngine(nullptr)
    {
    }

    //!
    //! \brief Builds the network engine
    //!
    bool build();

    //!
    //! \brief Runs the TensorRT inference engine for this sample
    //!
    bool infer();

private:
    samplesCommon::OnnxSampleParams mParams;

    nvinfer1::Dims mInputDims;  //!< The dimensions of the input to the network.
    nvinfer1::Dims mOutputDims; //!< The dimensions of the output to the network.

    cv::Mat image;  // the test IMAGE

    std::shared_ptr<nvinfer1::ICudaEngine> mEngine; //!< The TensorRT engine used to run the network

    bool processInput(const samplesCommon::BufferManager& buffers);

    bool verifyOutput(const samplesCommon::BufferManager& buffers);
};

//!
//! \brief BUILD - read serialized engine
//!
bool SampleMine::build()
{

    gLogInfo << "... Importing TensorRT engine "<<mParams.onnxFileName << locateFile(mParams.onnxFileName, mParams.dataDirs).c_str() << std::endl;
    std::ifstream ifs(locateFile(mParams.onnxFileName, mParams.dataDirs).c_str(), std::ios::binary | std::ios::ate);
        
    std::ifstream::pos_type len = ifs.tellg();
    std::vector<char> blob(len);
    ifs.seekg(0, std::ios::beg);
    ifs.read(&blob[0], len);
    IRuntime* runtime = createInferRuntime(gLogger);
    mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(
        runtime->deserializeCudaEngine(&blob[0], blob.size(), nullptr),
	samplesCommon::InferDeleter());

    runtime->destroy();
    if (!mEngine)
    {
        gLogInfo << "COULD NOT LOAD ENGINE?"<<std::endl;
        return false;
    }

    //---
    int nbindings = mEngine.get()->getNbBindings();
    assert(nbindings == 2);
    for (int b = 0; b < nbindings; ++b)
    {
        nvinfer1::Dims dims = mEngine.get()->getBindingDimensions(b);
        if (mEngine.get()->bindingIsInput(b))
        {
            mInputDims = dims;
            if (true) //mParams.verbose)
            {
                gLogInfo << "Found input: " << mEngine.get()->getBindingName(b) << " shape=" << dims
                         << " dtype=" << (int) mEngine.get()->getBindingDataType(b) << std::endl;
            }
        }
        else
        {
            mOutputDims = dims;
            if (true) //mParams.verbose)
            {
                gLogInfo << "Found output: " << mEngine.get()->getBindingName(b) << " shape=" << dims
                         << " dtype=" << (int) mEngine.get()->getBindingDataType(b) << std::endl;
            }
        }
    }
    //---

    return true;
}



//! \brief INFER
//!
bool SampleMine::infer()
{
    // Create RAII buffer manager object
    samplesCommon::BufferManager buffers(mEngine, mParams.batchSize);
    auto context = SampleUniquePtr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
    if (!context)
    {
        return false;
    }

    // Read the input data into the managed buffers
    assert(mParams.inputTensorNames.size() == 1);
    if (!processInput(buffers))
    {
        return false;
    }

    // Memcpy from host input buffers to device input buffers
    buffers.copyInputToDevice();

    bool status = context->executeV2(buffers.getDeviceBindings().data());
    if (!status)
    {
        return false;
    }

    // Memcpy from device output buffers to host output buffers
    buffers.copyOutputToHost();

    // Verify results
    if (!verifyOutput(buffers))
    {
        return false;
    }

    return true;
}

bool SampleMine::processInput(const samplesCommon::BufferManager& buffers)
{
    const int inputC = mInputDims.d[0];
    const int inputH = mInputDims.d[1];
    const int inputW = mInputDims.d[2];
    gLogInfo << "... inputC " << inputC <<std::endl;
    gLogInfo << "... inputH " << inputH <<std::endl;
    gLogInfo << "... inputW " << inputW <<std::endl;

    const int batchSize = mParams.batchSize;

    // Available images
    std::vector<std::string> imageList = {"dog.0.jpg"};
    for (int i = 0; i < batchSize; ++i)
    {
        readImage(locateFile(imageList[i], mParams.dataDirs), image);
    }

    float* hostDataBuffer = static_cast<float*>(buffers.getHostBuffer(mParams.inputTensorNames[0]));
    for (int i = 0, volImg = inputH * inputW; i < mParams.batchSize; ++i)
    {
        for (unsigned j = 0, volChl = inputH * inputW; j < inputH; ++j)
        {
            for( unsigned k = 0; k < inputW; ++ k)
            {
		// THIS INCLUDES BGR 012 -> 210 RGB
                cv::Vec3b bgr = image.at<cv::Vec3b>(j,k);
                hostDataBuffer[i * volImg + 0 * volChl + j * inputW + k] = (1.0 / 255.0) * float(bgr[2]);
                hostDataBuffer[i * volImg + 1 * volChl + j * inputW + k] = (1.0 / 255.0) * float(bgr[1]);
                hostDataBuffer[i * volImg + 2 * volChl + j * inputW + k] = (1.0 / 255.0) * float(bgr[0]);
            }
        }
    }
    
    return true;
}


//!
//! \brief Classifies digits and verify result
//!
bool SampleMine::verifyOutput(const samplesCommon::BufferManager& buffers)
{
    const int outputSize = mOutputDims.d[1];
    float* output = static_cast<float*>(buffers.getHostBuffer(mParams.outputTensorNames[0]));
    float val{0.0f};
    //int idx{0};

    // Calculate Softmax
    float sum{0.0f};
    for (int i = 0; i < outputSize; i++)
    {
        output[i] = exp(output[i]);
        sum += output[i];
    }

    gLogInfo << "Output:" << std::endl;
    for (int i = 0; i < outputSize; i++)
    {
        output[i] /= sum;
        val = std::max(val, output[i]);
        //if (val == output[i])
        //{
        //    idx = i;
        //}

        gLogInfo << " Prob " << i << "  " << std::fixed << std::setw(5) << std::setprecision(4) << output[i] << " "
                 << "Class " << i << ": " << std::string(int(std::floor(output[i] * 10 + 0.5f)), '*') << std::endl;
    }
    gLogInfo << std::endl;

    // return idx == mNumber && val > 0.9f;
    return true;
}



//!
//! \brief Initializes members of the params struct using the command line args
//!
samplesCommon::OnnxSampleParams initializeSampleParams(const samplesCommon::Args& args)
{
    samplesCommon::OnnxSampleParams params;
    if (args.dataDirs.empty()) //!< Use default directories if user hasn't provided directory paths
    {
        params.dataDirs.push_back("data/mine/");
        params.dataDirs.push_back("data/samples/mine/");
    }
    else //!< Use the data directory provided by the user
    {
        params.dataDirs = args.dataDirs;
    }
    //params.onnxFileName = "mnist.onnx";
    //params.inputTensorNames.push_back("Input3");
    params.onnxFileName = "dogs_vs_cats_model.trt";
    params.inputTensorNames.push_back("inception_v3_input:0");
    params.batchSize = 1;
    //params.outputTensorNames.push_back("Plus214_Output_0");
    params.outputTensorNames.push_back("dense_1");
    params.dlaCore = args.useDLACore;
    params.int8 = args.runInInt8;
    params.fp16 = args.runInFp16;

    return params;
}


//!
//! \brief Prints the help information for running this sample
//!
void printHelpInfo()
{
    std::cout
        << "Usage: ./sample_melinda [-h or --help] [-d or --datadir=<path to data directory>] [--useDLACore=<int>]\n";
    std::cout << "--help          Display help information\n";
    std::cout << "--datadir       Specify path to a data directory, overriding the default. This option can be used "
                 "multiple times to add multiple directories. If no data directories are given, the default is to use "
                 "data/samples/mine/ and data/mine/"
              << std::endl;
    std::cout << "--useDLACore=N  Specify a DLA engine for layers that support DLA. Value can range from 0 to n-1, "
                 "where n is the number of DLA engines on the platform."
              << std::endl;
}


//!
//! \brief MAIN
//!
int main(int argc, char** argv)
{
    samplesCommon::Args args;
    bool argsOK = samplesCommon::parseArgs(args, argc, argv);
    if (!argsOK)
    {
        gLogError << "Invalid arguments" << std::endl;
        printHelpInfo();
        return EXIT_FAILURE;
    }

    if (args.help)
    {
        printHelpInfo();
        return EXIT_SUCCESS;
    }

    auto sampleTest = gLogger.defineTest(gSampleName, argc, argv);

    gLogger.reportTestStart(sampleTest);

    SampleMine sample(initializeSampleParams(args));

    gLogInfo << "Building and running a GPU inference engine for DOGS.VS.CATS" << std::endl;

    if (!sample.build())
    {
        return gLogger.reportFail(sampleTest);
    }

    if (!sample.infer())
    {
        return gLogger.reportFail(sampleTest);
    }

    gLogInfo << "Ran " << argv[0] << " with: " << std::endl;

    return gLogger.reportPass(sampleTest);
}
