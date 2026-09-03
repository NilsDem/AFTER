// inspect_coreml.mm
//
// Usage:
//   ./inspect_coreml model.mlmodelc [cpu|gpu|ane|all]
//
// Example:
//   ./inspect_coreml exports/simpleae/model.mlmodelc all
//
// Compile:
//   clang++ \
//       -std=c++17 \
//       -O2 \
//       -fobjc-arc \
//       -framework Foundation \
//       -framework CoreML \
//       -framework Metal \
//       inspect_coreml.mm \
//       -o inspect_coreml

#import <Foundation/Foundation.h>
#import <CoreML/CoreML.h>
#import <Metal/Metal.h>

#include <dispatch/dispatch.h>

#include <iomanip>
#include <iostream>
#include <string>


static NSString* deviceName(id<MLComputeDeviceProtocol> device)
{
    if ([device isKindOfClass:[MLCPUComputeDevice class]])
    {
        return @"CPU";
    }

    if ([device isKindOfClass:[MLGPUComputeDevice class]])
    {
        MLGPUComputeDevice* gpu =
            (MLGPUComputeDevice*)device;

        if (gpu.metalDevice != nil)
        {
            return [NSString stringWithFormat:
                @"GPU (%@)",
                gpu.metalDevice.name
            ];
        }

        return @"GPU";
    }

    if ([device isKindOfClass:[MLNeuralEngineComputeDevice class]])
    {
        return @"ANE";
    }

    return @"Unknown";
}


static NSString* supportedDevicesString(
    NSArray<id<MLComputeDeviceProtocol>>* devices)
{
    NSMutableArray<NSString*>* names =
        [NSMutableArray array];

    for (id<MLComputeDeviceProtocol> device in devices)
    {
        [names addObject:deviceName(device)];
    }

    return [names componentsJoinedByString:@", "];
}


static void printOperation(
    MLComputePlan* plan,
    MLModelStructureProgramOperation* operation,
    int depth,
    int& index)
{
    MLComputePlanDeviceUsage* usage =
        [plan computeDeviceUsageForMLProgramOperation:operation];

    MLComputePlanCost* cost =
        [plan estimatedCostOfMLProgramOperation:operation];

    NSString* preferred = @"?";
    NSString* supported = @"?";

    if (usage != nil)
    {
        preferred =
            deviceName(usage.preferredComputeDevice);

        supported =
            supportedDevicesString(
                usage.supportedComputeDevices
            );
    }

    double weight = -1.0;

    if (cost != nil)
    {
        weight = cost.weight;
    }

    std::string indent(depth * 2, ' ');

    std::cout
        << std::left
        << std::setw(6)
        << index

        << std::setw(4)
        << indent

        << std::setw(28)
        << [operation.operatorName UTF8String]

        << std::setw(24)
        << [preferred UTF8String]

        << std::setw(42)
        << [supported UTF8String];

    if (weight >= 0.0)
    {
        std::cout
            << std::fixed
            << std::setprecision(4)
            << weight;
    }
    else
    {
        std::cout << "-";
    }

    std::cout << "\n";

    index++;

    //
    // Some ML Program operations contain nested blocks:
    // loops, conditionals, etc.
    //
    for (MLModelStructureProgramBlock* block
         in operation.blocks)
    {
        for (MLModelStructureProgramOperation* nested
             in block.operations)
        {
            printOperation(
                plan,
                nested,
                depth + 1,
                index
            );
        }
    }
}


static MLComputeUnits parseComputeUnits(
    const std::string& name)
{
    if (name == "cpu")
    {
        return MLComputeUnitsCPUOnly;
    }

    if (name == "gpu")
    {
        return MLComputeUnitsCPUAndGPU;
    }

    if (name == "ane")
    {
        return MLComputeUnitsCPUAndNeuralEngine;
    }

    if (name == "all")
    {
        return MLComputeUnitsAll;
    }

    std::cerr
        << "Unknown compute mode: "
        << name
        << "\n";

    std::cerr
        << "Expected one of: "
        << "cpu, gpu, ane, all\n";

    std::exit(1);
}


int main(int argc, const char* argv[])
{
    @autoreleasepool
    {
        if (argc < 2)
        {
            std::cerr
                << "Usage:\n"
                << "  "
                << argv[0]
                << " model.mlmodelc "
                << "[cpu|gpu|ane|all]\n";

            return 1;
        }

        const std::string path = argv[1];

        const std::string mode =
            argc >= 3
            ? argv[2]
            : "all";

        NSURL* modelURL =
            [NSURL fileURLWithPath:
                [NSString stringWithUTF8String:
                    path.c_str()
                ]
            ];

        MLModelConfiguration* configuration =
            [[MLModelConfiguration alloc] init];

        configuration.computeUnits =
            parseComputeUnits(mode);

        std::cout
            << "Model: "
            << path
            << "\n";

        std::cout
            << "Compute units: "
            << mode
            << "\n\n";

        //
        // MLComputePlan loading is asynchronous.
        // For a command-line utility we simply wait for it.
        //

        dispatch_semaphore_t semaphore =
            dispatch_semaphore_create(0);

        __block MLComputePlan* plan = nil;
        __block NSError* loadError = nil;

        [MLComputePlan
            loadContentsOfURL:modelURL
            configuration:configuration
            completionHandler:^(
                MLComputePlan* computePlan,
                NSError* error)
            {
                plan = computePlan;
                loadError = error;

                dispatch_semaphore_signal(
                    semaphore
                );
            }
        ];

        dispatch_semaphore_wait(
            semaphore,
            DISPATCH_TIME_FOREVER
        );

        if (plan == nil)
        {
            std::cerr
                << "Failed to create compute plan.\n";

            if (loadError != nil)
            {
                std::cerr
                    << [[loadError localizedDescription]
                        UTF8String]
                    << "\n";
            }

            return 1;
        }

        //
        // Your CoreMLTools conversion uses:
        //
        //     convert_to="mlprogram"
        //
        // so this should be an ML Program.
        //

        MLModelStructureProgram* program =
            plan.modelStructure.program;

        if (program == nil)
        {
            std::cerr
                << "This model is not an ML Program.\n";

            return 1;
        }

        std::cout
            << std::left
            << std::setw(6)
            << "#"

            << std::setw(4)
            << ""

            << std::setw(28)
            << "operation"

            << std::setw(24)
            << "preferred"

            << std::setw(42)
            << "supported"

            << "cost"
            << "\n";

        std::cout
            << std::string(110, '-')
            << "\n";

        int index = 0;

        //
        // Usually there is a single "main" function,
        // but iterate all functions to be safe.
        //

        for (NSString* functionName
             in program.functions)
        {
            MLModelStructureProgramFunction* function =
                program.functions[functionName];

            std::cout
                << "\nFunction: "
                << [functionName UTF8String]
                << "\n";

            for (
                MLModelStructureProgramOperation* operation
                in function.block.operations
            )
            {
                printOperation(
                    plan,
                    operation,
                    0,
                    index
                );
            }
        }

        std::cout << "\n";

        //
        // Also show all hardware visible to Core ML.
        //

        std::cout
            << "Available Core ML devices:\n";

        for (
            id<MLComputeDeviceProtocol> device
            in MLAllComputeDevices()
        )
        {
            std::cout
                << "  - "
                << [deviceName(device) UTF8String]
                << "\n";
        }

        return 0;
    }
}