/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

//!
//! sampleOnnxMNIST.cpp
//! 本文件实现 ONNX MNIST 手写数字识别示例: 从 ONNX 模型构建 TensorRT 引擎并执行推理。
//!
//! 运行方式:
//!   ./sample_onnx_mnist [-h|--help] [-d=/path/to/data/dir|--datadir=...] [--useDLACore=<int>]
//!   [-t|--timingCacheFile=...]
//!
//! ========== 在 DeepStream 插件中的使用思路 ==========
//! 1) 引擎构建(可放在插件 init/one-time) :
//!    - 用 IBuilder + ONNX Parser 解析 .onnx, 得到 INetworkDefinition；
//!    - 用 IBuilderConfig 配置精度、DLA 等, 再 buildSerializedNetwork 得到 plan；
//!    - 用 IRuntime::deserializeCudaEngine 反序列化得到 ICudaEngine(即本示例的 build() 流程) 。
//! 2) 推理(每帧/每个 batch) :
//!    - 从 engine 创建 IExecutionContext；
//!    - 用 setTensorAddress 绑定输入/输出 GPU 缓冲区；
//!    - 把预处理好的输入拷到 device buffer, 再 executeV2, 最后把输出拷回 host(即本示例的 infer() 流程) 。
//! 3) 与 DeepStream 对接:
//!    - 输入: 从 GstBuffer/NvDsBatchMeta 取图像, 按模型要求做预处理(缩放、归一化等) 写入 engine 的输入 tensor；
//!    - 输出: 从 engine 的输出 tensor 读结果, 再写回 NvDsFrameMeta / 自定义元数据供下游使用。
//!

// 供 common 中 TRT 符号解析使用
#define DEFINE_TRT_ENTRYPOINTS 1

#include "argsParser.h" // 命令行解析、OnnxSampleParams/Args
#include "buffers.h"    // BufferManager: host/device 缓冲区分配与拷贝
#include "common.h"     // ASSERT、locateFile、makeCudaStream、enableDLA 等
#include "logger.h"     // sample::gLogger、gLogInfo、gLogError
#include "parserOnnxConfig.h"

#include "NvInfer.h"      // IBuilder、INetworkDefinition、IRuntime、ICudaEngine、IExecutionContext 等
#include "NvOnnxParser.h" // createParser、IParser、parseFromFile
#include <cuda_runtime_api.h>

#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>

using namespace nvinfer1;

// 本 sample 的名称, 用于日志与测试报告
const std::string gSampleName = "TensorRT.sample_onnx_mnist";

//! \brief  The SampleOnnxMNIST class implements the ONNX MNIST sample
//!
//! \brief SampleOnnxMNIST: ONNX MNIST 手写数字识别示例类
//!
//! \details 封装「从 ONNX 构建 TensorRT 引擎」与「单次推理」的完整流程,
//!          对应 DeepStream 中「加载自定义 ONNX 模型并做推理」的典型用法。
//!
class SampleOnnxMNIST
{
public:
    //!
    //! \brief 构造函数, 仅保存参数, 不创建引擎
    //! \param params 包含 onnx 路径、数据目录、输入/输出 tensor 名称、DLA、timing cache 等
    //!
    SampleOnnxMNIST(const samplesCommon::OnnxSampleParams& params)
        : mParams(params)
        , mRuntime(nullptr)
        , mEngine(nullptr)
    {
    }

    //!
    //! \brief 构建 TensorRT 引擎(解析 ONNX -> 构建 plan -> 反序列化得到 engine)
    //! \return 成功返回 true, 失败返回 false
    //!
    bool build();

    //!
    //! \brief 执行一次推理: 准备输入 -> 拷贝到设备 -> executeV2 -> 拷贝输出回 host -> 校验
    //! \return 推理且校验通过返回 true, 否则 false
    //!
    bool infer();

private:
    samplesCommon::OnnxSampleParams mParams; //!< 样本参数: onnx 文件名、数据目录、输入/输出名、DLA、timing cache 等

    nvinfer1::Dims mInputDims;  //!< 网络输入维度(MNIST 一般为 [N,1,28,28])
    nvinfer1::Dims mOutputDims; //!< 网络输出维度(MNIST 一般为 [N,10])
    int mNumber{0};             //!< 当前用于测试的标签(0~9) , 用于 verifyOutput 校验

    std::shared_ptr<nvinfer1::IRuntime> mRuntime;   //!< TensorRT 运行时, 用于反序列化 plan 得到 engine
    std::shared_ptr<nvinfer1::ICudaEngine> mEngine; //!< 反序列化后的引擎, 推理时由 context 使用

    //!
    //! \brief 使用 ONNX 解析器填充网络定义(解析 mnist.onnx 到 network)
    //! \param builder   引擎构建器
    //! \param network   待填充的网络定义(解析后包含 MNIST 的层)
    //! \param config    构建配置(精度、DLA、profile 等)
    //! \param parser    ONNX 解析器, 绑定到 network
    //! \param timingCache 可选的时间缓存, 用于加速重复构建
    //! \return 解析成功返回 true
    //!
    bool constructNetwork(std::unique_ptr<nvinfer1::IBuilder>& builder,
        std::unique_ptr<nvinfer1::INetworkDefinition>& network, std::unique_ptr<nvinfer1::IBuilderConfig>& config,
        std::unique_ptr<nvonnxparser::IParser>& parser, std::unique_ptr<nvinfer1::ITimingCache>& timingCache);

    //!
    //! \brief 读取一张 MNIST 图像到 host 缓冲区并做简单预处理(本示例用 PGM 文件)
    //! \param buffers 缓冲区管理器, 提供 getHostBuffer 等
    //! \return 成功返回 true
    //!
    bool processInput(const samplesCommon::BufferManager& buffers);

    //!
    //! \brief 对输出做 softmax 并校验预测类别是否与 mNumber 一致且置信度 > 0.9
    //! \param buffers 缓冲区管理器, 用于取输出 host 指针
    //! \return 预测正确且置信度足够返回 true
    //!
    bool verifyOutput(const samplesCommon::BufferManager& buffers);
};

//!
//! \brief 构建 TensorRT 引擎: 创建 builder/network/config/parser -> 解析 ONNX -> 构建序列化 plan -> 反序列化得到 engine
//!
//! \details 流程对应 DeepStream 中「加载自定义 ONNX 并生成 engine」的一次性初始化；
//!          若在 DeepStream 插件里, 可将 plan 保存为 .engine 文件, 后续直接 deserialize 以加快启动。
//! \return 成功返回 true, 任一步失败返回 false
//!
bool SampleOnnxMNIST::build()
{
    // 1) 创建 builder(入口, 用于创建 network、config 等)
    auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(sample::gLogger.getTRTLogger()));
    if (!builder)
    {
        return false;
    }

    // 2) 创建强类型网络定义(kSTRONGLY_TYPED 便于 ONNX 解析)
    auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(
        builder->createNetworkV2(1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kSTRONGLY_TYPED)));
    if (!network)
    {
        return false;
    }

    // 3) 创建构建配置(精度、workspace、DLA、timing cache 等)
    auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config)
    {
        return false;
    }

    // 4) 创建 ONNX 解析器: createParser 需要 INetworkDefinition&, 故传 *network(对 unique_ptr 解引用得到引用)
    // XXX 参数 network
    auto parser
        = std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, sample::gLogger.getTRTLogger()));
    if (!parser)
    {
        return false;
    }

    auto timingCache = std::unique_ptr<nvinfer1::ITimingCache>();

    // 5) 解析 ONNX 文件并写入 network, 可选加载/保存 timing cache、启用 DLA
    auto constructed = constructNetwork(builder, network, config, parser, timingCache);
    if (!constructed)
    {
        return false;
    }

    // 6) 设置 profile 用的 CUDA stream(构建时 profiling 需要)
    auto profileStream = samplesCommon::makeCudaStream();
    if (!profileStream)
    {
        return false;
    }
    config->setProfileStream(*profileStream);

    // 7) 构建序列化引擎(plan) : 可保存到文件, DeepStream 中可只构建一次或离线生成 .engine
    std::unique_ptr<IHostMemory> plan{builder->buildSerializedNetwork(*network, *config)};
    if (!plan)
    {
        return false;
    }

    if (timingCache != nullptr && !mParams.timingCacheFile.empty())
    {
        samplesCommon::updateTimingCacheFile(
            sample::gLogger.getTRTLogger(), mParams.timingCacheFile, timingCache.get(), *builder);
    }

    // 8) 创建 runtime 并用 plan 反序列化得到可执行的 engine
    mRuntime = std::shared_ptr<nvinfer1::IRuntime>(createInferRuntime(sample::gLogger.getTRTLogger()));
    if (!mRuntime)
    {
        return false;
    }

    mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(mRuntime->deserializeCudaEngine(plan->data(), plan->size()));
    if (!mEngine)
    {
        return false;
    }

    // 9) 记录输入/输出维度, 供后续 processInput / verifyOutput 使用
    ASSERT(network->getNbInputs() == 1);
    mInputDims = network->getInput(0)->getDimensions();
    ASSERT(mInputDims.nbDims == 4);

    ASSERT(network->getNbOutputs() == 1);
    mOutputDims = network->getOutput(0)->getDimensions();
    ASSERT(mOutputDims.nbDims == 2);

    return true;
}

//!
//! \brief 用 ONNX 解析器把 mnist.onnx 解析进 network, 并可选配置 timing cache、DLA
//!
//! \param builder      引擎构建器(用于 enableDLA 等)
//! \param network      将被填充的网络定义；parseFromFile 会把 ONNX 层写入此 network
//! \param config       构建配置(用于 timing cache、DLA)
//! \param parser       已绑定到 network 的 ONNX 解析器
//! \param timingCache  若指定了 timingCacheFile 则从文件加载, 用于加速重复构建
//! \return 解析成功且配置成功返回 true
//!
bool SampleOnnxMNIST::constructNetwork(std::unique_ptr<nvinfer1::IBuilder>& builder,
    std::unique_ptr<nvinfer1::INetworkDefinition>& network, std::unique_ptr<nvinfer1::IBuilderConfig>& config,
    std::unique_ptr<nvonnxparser::IParser>& parser, std::unique_ptr<nvinfer1::ITimingCache>& timingCache)
{
    // 从 dataDirs 中定位 onnx 文件并解析到 network
    auto parsed = parser->parseFromFile(samplesCommon::locateFile(mParams.onnxFileName, mParams.dataDirs).c_str(),
        static_cast<int>(sample::gLogger.getReportableSeverity()));
    if (!parsed)
    {
        return false;
    }

    if (mParams.timingCacheFile.size())
    {
        timingCache
            = samplesCommon::buildTimingCacheFromFile(sample::gLogger.getTRTLogger(), *config, mParams.timingCacheFile);
    }

    samplesCommon::enableDLA(builder.get(), config.get(), mParams.dlaCore);

    return true;
}

//!
//! \brief 执行一次完整推理: 分配缓冲区 -> 绑定 tensor 地址 -> 填输入 -> 拷贝到设备 -> executeV2 -> 拷回 host -> 校验
//!
//! \details 对应 DeepStream 中「每帧/每个 batch」的推理路径: 用 context 绑定输入/输出 GPU 指针,
//!          把预处理好的数据拷到 device, 执行 executeV2, 再把输出拷回并写回 meta。
//! \return 推理执行且校验通过返回 true
//!
bool SampleOnnxMNIST::infer()
{
    // 根据 engine 的 I/O 信息分配 host/device 缓冲区(RAII 管理)
    samplesCommon::BufferManager buffers(mEngine);

    auto context = std::unique_ptr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
    if (!context)
    {
        return false;
    }

    // 把所有 I/O tensor 的 device 地址绑定到 context(推理时从这些地址读输入、写输出)
    for (int32_t i = 0, e = mEngine->getNbIOTensors(); i < e; i++)
    {
        auto const name = mEngine->getIOTensorName(i);
        context->setTensorAddress(name, buffers.getDeviceBuffer(name));
    }

    // 把本帧输入写入 host 缓冲区(本示例为读 PGM 并归一化；DeepStream 中改为从 GstBuffer/NvDsBatchMeta 预处理)
    ASSERT(mParams.inputTensorNames.size() == 1);
    if (!processInput(buffers))
    {
        return false;
    }

    // Host -> Device: 把输入拷贝到 GPU
    buffers.copyInputToDevice();

    // 同步执行推理(DeepStream 中可用 CUDA stream 与 pipeline 对齐)
    bool status = context->executeV2(buffers.getDeviceBindings().data());
    if (!status)
    {
        return false;
    }

    // Device -> Host: 把输出拷回 CPU
    buffers.copyOutputToHost();

    // Verify results
    if (!verifyOutput(buffers))
    {
        return false;
    }

    return true;
}

//!
//! \brief 读取一张 MNIST 图像到 host 缓冲区并做预处理(本示例用随机 0~9 的 PGM 文件)
//!
//! \param buffers 缓冲区管理器；通过 getHostBuffer(输入 tensor 名) 得到 host 侧输入指针
//! \return 成功返回 true
//!
//! 预处理说明: MNIST 输入通常为 [N,C,H,W]=[1,1,28,28], 灰度 [0,255] 转为 float 并做 1 - x/255(白底黑字) 。
//! 在 DeepStream 中可替换为: 从 NvDsFrameMeta 取 ROI/整帧, resize 到 28x28, 归一化后写入此处得到的 host 缓冲区。
//!
bool SampleOnnxMNIST::processInput(const samplesCommon::BufferManager& buffers)
{
    const int inputH = mInputDims.d[2]; // 输入高(MNIST 为 28)
    const int inputW = mInputDims.d[3]; // 输入宽(MNIST 为 28)

    srand(unsigned(time(nullptr)));
    std::vector<uint8_t> fileData(inputH * inputW);
    mNumber = rand() % 10; // 随机选一个数字作为本帧「真值」, 用于后面 verifyOutput 校验
    samplesCommon::readPGMFile(
        samplesCommon::locateFile(std::to_string(mNumber) + ".pgm", mParams.dataDirs), fileData.data(), inputH, inputW);

    // Print an ascii representation
    sample::gLogInfo << "Input:" << std::endl;
    for (int i = 0; i < inputH * inputW; i++)
    {
        sample::gLogInfo << (" .:-=+*#%@"[fileData[i] / 26]) << (((i + 1) % inputW) ? "" : "\n");
    }
    sample::gLogInfo << std::endl;

    // 写入模型输入: host 侧 float 缓冲区, 格式与模型一致(此处 NCHW、归一化)
    float* hostDataBuffer = static_cast<float*>(buffers.getHostBuffer(mParams.inputTensorNames[0]));
    for (int i = 0; i < inputH * inputW; i++)
    {
        hostDataBuffer[i] = 1.0 - float(fileData[i] / 255.0); // 白底黑字 -> [0,1], 与训练时一致
    }

    return true;
}

//!
//! \brief 对输出做 softmax, 取 argmax 为预测类别, 并校验是否与 mNumber 一致且最大概率 > 0.9
//!
//! \param buffers 缓冲区管理器；通过 getHostBuffer(输出 tensor 名) 得到 host 侧输出指针
//! \return 预测类别 == mNumber 且最大概率 > 0.9 返回 true, 否则 false
//!
//! 说明: MNIST 输出通常为 10 维 logits, 此处先 exp 再归一化得到概率；DeepStream 中可把 idx 与 val 写回 NvDsFrameMeta
//! 等。
//!
bool SampleOnnxMNIST::verifyOutput(const samplesCommon::BufferManager& buffers)
{
    const int outputSize = mOutputDims.d[1]; // 类别数(MNIST 为 10)
    float* output = static_cast<float*>(buffers.getHostBuffer(mParams.outputTensorNames[0]));
    float val{0.0F}; // 最大概率
    int idx{0};      // argmax 类别

    // 就地计算 softmax: 先 exp 再除以 sum
    float sum{0.0F};
    for (int i = 0; i < outputSize; i++)
    {
        output[i] = exp(output[i]);
        sum += output[i];
    }

    sample::gLogInfo << "Output:" << std::endl;
    for (int i = 0; i < outputSize; i++)
    {
        output[i] /= sum;
        val = std::max(val, output[i]);
        if (val == output[i])
        {
            idx = i;
        }

        sample::gLogInfo << " Prob " << i << "  " << std::fixed << std::setw(5) << std::setprecision(4) << output[i]
                         << " "
                         << "Class " << i << ": " << std::string(int(std::floor(output[i] * 10 + 0.5F)), '*')
                         << std::endl;
    }
    sample::gLogInfo << std::endl;

    return idx == mNumber && val > 0.9F;
}

//!
//! \brief 根据命令行参数初始化 OnnxSampleParams(数据目录、onnx 文件名、输入/输出 tensor 名、DLA、timing cache)
//!
//! \param args 解析后的命令行参数(dataDirs、useDLACore、timingCacheFile 等)
//! \return 填充好的 OnnxSampleParams；inputTensorNames/outputTensorNames 需与 mnist.onnx 内名称一致
//!
samplesCommon::OnnxSampleParams initializeSampleParams(const samplesCommon::Args& args)
{
    samplesCommon::OnnxSampleParams params;
    if (args.dataDirs.empty()) // Use default directories if user hasn't provided directory paths
    {
        params.dataDirs.push_back("data/mnist/");
        params.dataDirs.push_back("data/samples/mnist/");
    }
    else // Use the data directory provided by the user
    {
        params.dataDirs = args.dataDirs;
    }
    params.onnxFileName = "mnist.onnx";
    // 以下名称必须与 ONNX 模型中输入/输出 tensor 的 name 一致(可用 netron 等工具查看)
    params.inputTensorNames.push_back("Input3");
    params.outputTensorNames.push_back("Plus214_Output_0");
    params.dlaCore = args.useDLACore;
    params.timingCacheFile = args.timingCacheFile;

    return params;
}

//!
//! \brief 打印命令行用法说明(--help 时调用)
//!
void printHelpInfo()
{
    std::cout
        << "Usage: ./sample_onnx_mnist [-h or --help] [-d or --datadir=<path to data directory>] [--useDLACore=<int>] "
           "[-t or --timingCacheFile=<path to timing cache file]\n"
        << std::endl;
    std::cout << "--help             Display help information\n" << std::endl;
    std::cout << "--datadir          Specify path to a data directory, overriding the default. This option can be used "
                 "multiple times to add multiple directories. If no data directories are given, the default is to use "
                 "(data/samples/mnist/, data/mnist/)\n"
              << std::endl;
    std::cout << "--useDLACore=N     Specify a DLA engine for layers that support DLA. Value can range from 0 to n-1, "
                 "where n is the number of DLA engines on the platform.\n"
              << std::endl;
    std::cout << "--timingCacheFile  Specify path to a timing cache file. If it does not already exist, it will be "
                 "created.\n"
              << std::endl;
}

//!
//! \brief 入口: 解析参数 -> 初始化 SampleOnnxMNIST 参数 -> build() 构建引擎 -> infer() 执行一次推理并校验
//!
//! \param argc 命令行参数个数
//! \param argv 命令行参数列表
//! \return EXIT_SUCCESS 成功, EXIT_FAILURE 参数错误或 build/infer 失败
//!
int main(int argc, char** argv)
{
    samplesCommon::Args args;
    bool argsOK = samplesCommon::parseArgs(args, argc, argv);
    if (!argsOK)
    {
        sample::gLogError << "Invalid arguments" << std::endl;
        printHelpInfo();
        return EXIT_FAILURE;
    }
    if (args.help)
    {
        printHelpInfo();
        return EXIT_SUCCESS;
    }

    auto sampleTest = sample::gLogger.defineTest(gSampleName, argc, argv);

    sample::gLogger.reportTestStart(sampleTest);

    SampleOnnxMNIST onnx_mnist(initializeSampleParams(args));

    sample::gLogInfo << "Building and running a GPU inference engine for Onnx MNIST" << std::endl;

    if (!onnx_mnist.build())
    {
        return sample::gLogger.reportFail(sampleTest);
    }
    if (!onnx_mnist.infer())
    {
        return sample::gLogger.reportFail(sampleTest);
    }

    return sample::gLogger.reportPass(sampleTest);
}
