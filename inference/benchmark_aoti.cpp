// benchmark_aoti.cpp

#include <torch/torch.h>
#include <torch/csrc/inductor/aoti_package/model_package_loader.h>

#include <algorithm>
#include <chrono>
#include <iostream>
#include <string>
#include <vector>


static double median(std::vector<double> values)
{
    std::sort(values.begin(), values.end());

    const size_t n = values.size();

    if (n % 2 == 0)
        return 0.5 * (values[n / 2 - 1] + values[n / 2]);

    return values[n / 2];
}


int main(int argc, const char* argv[])
{
    if (argc < 5)
    {
        std::cerr
            << "Usage:\n"
            << "  " << argv[0]
            << " model.pt2 callback_samples state_size audio_channels "
            << "[warmup] [reps] [trials]\n\n"
            << "Example:\n"
            << "  " << argv[0]
            << " AE_64_1f_aoti.pt2 64 123456 1 20 200 3\n";

        return 1;
    }

    const std::string modelPath = argv[1];

    const int callbackSamples = std::stoi(argv[2]);
    const int stateSize       = std::stoi(argv[3]);
    const int audioChannels   = std::stoi(argv[4]);

    const int warmup =
        argc > 5 ? std::stoi(argv[5]) : 20;

    const int reps =
        argc > 6 ? std::stoi(argv[6]) : 200;

    const int trials =
        argc > 7 ? std::stoi(argv[7]) : 3;

    const double sampleRate = 44100.0;

    //
    // Disable autograd bookkeeping.
    //
    c10::InferenceMode inferenceMode;
    at::set_num_threads(1);
    at::set_num_interop_threads(1);

    std::cout << "Loading " << modelPath << "...\n";

    //
    // Load AOTInductor .pt2 package.
    //
    torch::inductor::AOTIModelPackageLoader loader(
        modelPath,
        "model",
        true
    );

    //
    // IMPORTANT:
    //
    // Shapes/dtype/strides must match what you exported.
    //
    // Adapt the state shape here if your state tensor is not flat.
    //

    auto options =
        torch::TensorOptions()
            .dtype(torch::kFloat32)
            .device(torch::kCPU);

    torch::Tensor audio =
        torch::randn(
            {1, audioChannels, callbackSamples},
            options
        );

    torch::Tensor state =
        torch::zeros(
            {1, stateSize},
            options
        );

    //
    // Warmup
    //
    for (int i = 0; i < warmup; ++i)
    {
        std::vector<torch::Tensor> outputs =
            loader.run({
                audio,
                state
            });

        state = outputs[1];
    }

    //
    // Benchmark
    //
    std::vector<double> trialCallbackMs;

    torch::Tensor outputAudio;

    for (int trial = 0; trial < trials; ++trial)
    {
        auto start =
            std::chrono::steady_clock::now();

        for (int i = 0; i < reps; ++i)
        {
            std::vector<torch::Tensor> outputs =
                loader.run({
                    audio,
                    state
                });

            outputAudio = outputs[0];
            state = outputs[1];
        }

        auto stop =
            std::chrono::steady_clock::now();

        const double elapsedMs =
            std::chrono::duration<double, std::milli>(
                stop - start
            ).count();

        trialCallbackMs.push_back(
            elapsedMs / static_cast<double>(reps)
        );
    }

    const double callbackMs =
        median(trialCallbackMs);

    const double audioMs =
        1000.0
        * static_cast<double>(callbackSamples)
        / sampleRate;

    const double rtf =
        callbackMs / audioMs;

    std::cout << "\n";
    std::cout << "callback_samples : "
              << callbackSamples << "\n";

    std::cout << "audio_ms         : "
              << audioMs << "\n";

    std::cout << "callback_ms      : "
              << callbackMs << "\n";

    std::cout << "RTF              : "
              << rtf << "\n";

    std::cout << "output shape     : "
              << outputAudio.sizes() << "\n";

    return 0;
}
