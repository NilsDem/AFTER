// benchmark_coreml.mm

#import <Foundation/Foundation.h>
#import <CoreML/CoreML.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <numeric>
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
    @autoreleasepool
    {
        if (argc < 2)
        {
            std::cerr
                << "Usage:\n"
                << "  " << argv[0]
                << " model.mlmodelc [warmup] [reps] [trials]\n";

            return 1;
        }

        std::string modelPath = argv[1];

        const int warmup = argc > 2 ? std::stoi(argv[2]) : 20;
        const int reps   = argc > 3 ? std::stoi(argv[3]) : 200;
        const int trials = argc > 4 ? std::stoi(argv[4]) : 3;

        NSString* nsPath =
            [NSString stringWithUTF8String:modelPath.c_str()];

        NSURL* modelURL =
            [NSURL fileURLWithPath:nsPath];

        NSError* error = nil;

        //
        // Load compiled Core ML model
        //

        MLModelConfiguration* config =
            [[MLModelConfiguration alloc] init];

        // config.computeUnits = MLComputeUnitsCPUOnly;

        MLModel* model =
            [MLModel modelWithContentsOfURL:modelURL
                             configuration:config
                                     error:&error];

        if (!model)
        {
            std::cerr
                << "Could not load Core ML model:\n"
                << [[error localizedDescription] UTF8String]
                << "\n";

            return 1;
        }

        std::cout << "Loaded: "
                  << modelPath
                  << "\n";

        //
        // Find audio input description.
        //

        MLFeatureDescription* audioDescription =
            model.modelDescription.inputDescriptionsByName[@"audio"];

        if (!audioDescription)
        {
            std::cerr
                << "Model has no input called 'audio'.\n";

            return 1;
        }

        MLMultiArrayConstraint* constraint =
            audioDescription.multiArrayConstraint;

        if (!constraint)
        {
            std::cerr
                << "'audio' is not an MLMultiArray input.\n";

            return 1;
        }

        NSArray<NSNumber*>* shape = constraint.shape;

        std::cout << "audio shape = [";

        long long elementCount = 1;

        for (NSUInteger i = 0; i < shape.count; ++i)
        {
            long long dim = shape[i].longLongValue;

            elementCount *= dim;

            std::cout << dim;

            if (i + 1 != shape.count)
                std::cout << ", ";
        }

        std::cout << "]\n";

        //
        // Allocate audio input.
        //

        MLMultiArray* audio =
            [[MLMultiArray alloc]
                initWithShape:shape
                dataType:MLMultiArrayDataTypeFloat32
                error:&error];

        if (!audio)
        {
            std::cerr
                << "Could not allocate input:\n"
                << [[error localizedDescription] UTF8String]
                << "\n";

            return 1;
        }

        //
        // Fill it once.
        //
        // For benchmarking there's no need to regenerate random audio
        // every callback.
        //

        float* audioData =
            static_cast<float*>(audio.dataPointer);

        for (long long i = 0; i < elementCount; ++i)
        {
            audioData[i] =
                0.01f * std::sin(0.01f * static_cast<float>(i));
        }

        //
        // Stateless streaming cache.
        //

        MLFeatureValue* audioValue =
            [MLFeatureValue featureValueWithMultiArray:audio];

        MLFeatureDescription* cacheDescription =
            model.modelDescription.inputDescriptionsByName[@"cache"];

        if (!cacheDescription || !cacheDescription.multiArrayConstraint)
        {
            std::cerr
                << "Model has no MLMultiArray input called 'cache'. "
                << "Use the stateless Core ML artifact.\n";

            return 1;
        }

        MLMultiArrayConstraint* cacheConstraint =
            cacheDescription.multiArrayConstraint;

        MLMultiArray* cache =
            [[MLMultiArray alloc]
                initWithShape:cacheConstraint.shape
                dataType:cacheConstraint.dataType
                error:&error];

        if (!cache)
        {
            std::cerr
                << "Could not allocate cache:\n"
                << [[error localizedDescription] UTF8String]
                << "\n";

            return 1;
        }

        for (NSInteger i = 0; i < cache.count; ++i)
            cache[i] = @0;

        std::cout << "cache shape = [";
        for (NSUInteger i = 0; i < cache.shape.count; ++i)
        {
            std::cout << cache.shape[i].longLongValue;
            if (i + 1 != cache.shape.count)
                std::cout << ", ";
        }
        std::cout << "]\n";

        MLPredictionOptions* options =
            [[MLPredictionOptions alloc] init];

        auto predict = [&]() -> id<MLFeatureProvider>
        {
            NSDictionary<NSString*, MLFeatureValue*>* inputs =
                @{
                    @"audio": audioValue,
                    @"cache": [MLFeatureValue featureValueWithMultiArray:cache]
                };

            MLDictionaryFeatureProvider* provider =
                [[MLDictionaryFeatureProvider alloc]
                    initWithDictionary:inputs
                    error:&error];

            if (!provider)
                return nil;

            id<MLFeatureProvider> output =
                [model predictionFromFeatures:provider
                                      options:options
                                        error:&error];

            MLFeatureValue* nextState =
                [output featureValueForName:@"state_out"];

            if (!nextState || !nextState.multiArrayValue)
                return nil;

            cache = nextState.multiArrayValue;
            return output;
        };

        //
        // Warmup
        //

        for (int i = 0; i < warmup; ++i)
        {
            id<MLFeatureProvider> output = predict();

            if (!output)
            {
                std::cerr
                    << "Warmup prediction failed:\n"
                    << [[error localizedDescription] UTF8String]
                    << "\n";

                return 1;
            }
        }

        //
        // Benchmark
        //

        std::vector<double> trialMs;

        for (int trial = 0; trial < trials; ++trial)
        {
            const auto start =
                std::chrono::steady_clock::now();

            id<MLFeatureProvider> output = nil;

            for (int i = 0; i < reps; ++i)
            {
                output = predict();

                if (!output)
                {
                    std::cerr
                        << "Prediction failed:\n"
                        << [[error localizedDescription] UTF8String]
                        << "\n";

                    return 1;
                }
            }

            const auto stop =
                std::chrono::steady_clock::now();

            const double elapsedMs =
                std::chrono::duration<double, std::milli>(
                    stop - start
                ).count();

            trialMs.push_back(
                elapsedMs / static_cast<double>(reps)
            );

            //
            // Make sure we really obtain the model output.
            //

            MLFeatureValue* outputValue =
                [output featureValueForName:@"audio_out"];

            if (!outputValue)
            {
                std::cerr
                    << "No 'audio_out' output.\n";

                return 1;
            }
        }

        const double callbackMs =
            median(trialMs);

        //
        // Derive represented audio duration from last dimension.
        //

        const double sampleRate = 44100.0;
        const long long samples =
            shape.lastObject.longLongValue;

        const double audioMs =
            1000.0
            * static_cast<double>(samples)
            / sampleRate;

        const double rtf =
            callbackMs / audioMs;

        std::cout << "\n";
        std::cout << "samples       : " << samples << "\n";
        std::cout << "audio_ms      : " << audioMs << "\n";
        std::cout << "callback_ms   : " << callbackMs << "\n";
        std::cout << "RTF           : " << rtf << "\n";

        return 0;
    }
}
