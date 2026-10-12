#import <Foundation/Foundation.h>
#import <CoreML/CoreML.h>
#import <CoreAudio/CoreAudio.h>
#import <AudioToolbox/AudioToolbox.h>
#include "BlockRing.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <csignal>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <pthread.h>

static volatile sig_atomic_t stopped = 0;
static void signalHandler(int) { stopped = 1; }
static void check(OSStatus s, const char* what) {
    if (s) throw std::runtime_error(std::string(what) + ": OSStatus " + std::to_string(s));
}
static AudioObjectPropertyAddress address(AudioObjectPropertySelector selector,
    AudioObjectPropertyScope scope = kAudioObjectPropertyScopeGlobal) {
    return {selector, scope, kAudioObjectPropertyElementMain};
}
template<class T> static T get(AudioObjectID id, AudioObjectPropertySelector selector,
    AudioObjectPropertyScope scope = kAudioObjectPropertyScopeGlobal) {
    T result{}; UInt32 bytes = sizeof(result); auto a = address(selector, scope);
    check(AudioObjectGetPropertyData(id, &a, 0, nullptr, &bytes, &result), "read device property");
    return result;
}
template<class T> static void set(AudioObjectID id, AudioObjectPropertySelector selector, T value) {
    auto a = address(selector);
    check(AudioObjectSetPropertyData(id, &a, 0, nullptr, sizeof(value), &value), "set device property");
}
static unsigned channels(AudioDeviceID id, bool input) {
    auto a = address(kAudioDevicePropertyStreamConfiguration,
                     input ? kAudioDevicePropertyScopeInput : kAudioDevicePropertyScopeOutput);
    UInt32 bytes = 0;
    check(AudioObjectGetPropertyDataSize(id, &a, 0, nullptr, &bytes), "stream configuration size");
    std::vector<char> data(bytes);
    check(AudioObjectGetPropertyData(id, &a, 0, nullptr, &bytes, data.data()), "stream configuration");
    auto* list = reinterpret_cast<AudioBufferList*>(data.data());
    unsigned count = 0;
    for (unsigned i=0; i<list->mNumberBuffers; ++i) count += list->mBuffers[i].mNumberChannels;
    return count;
}
static void listDevices() {
    auto a = address(kAudioHardwarePropertyDevices); UInt32 bytes = 0;
    check(AudioObjectGetPropertyDataSize(kAudioObjectSystemObject, &a, 0, nullptr, &bytes), "device list");
    std::vector<AudioDeviceID> devices(bytes / sizeof(AudioDeviceID));
    check(AudioObjectGetPropertyData(kAudioObjectSystemObject, &a, 0, nullptr, &bytes, devices.data()), "device list");
    for (auto id : devices) {
        CFStringRef name = get<CFStringRef>(id, kAudioObjectPropertyName);
        std::cout << id << "  " << [(__bridge NSString*)name UTF8String]
                  << "  in=" << channels(id, true) << " out=" << channels(id, false) << '\n';
        CFRelease(name);
    }
}
struct Engine {
    BlockRing ring;
    unsigned size = 256, inputChannel = 0, outputChannel = 0;
    double rate = 44100;
    std::vector<float> file;
    size_t filePosition = 0;
    bool useFile = false, loop = false;
    std::atomic<bool> running{true}, failed{false}, eof{false};
    std::atomic<uint64_t> calls{0}, nanos{0}, callbackErrors{0};
    std::atomic<uint64_t> inputCallbacks{0}, outputCallbacks{0};
    std::atomic<double> lastRTF{0};
    double delayMs = 0;

    // AudioDeviceIOProc uses native interleaved/noninterleaved Float32 buffers.
    static bool readChannel(const AudioBufferList* list, unsigned channel, float* dst, unsigned n) {
        if (!list) return false;
        for (unsigned b=0; b<list->mNumberBuffers; ++b) {
            const auto& buf = list->mBuffers[b];
            if (channel >= buf.mNumberChannels) { channel -= buf.mNumberChannels; continue; }
            if (!buf.mData || buf.mDataByteSize != n * buf.mNumberChannels * sizeof(float)) return false;
            auto* src = static_cast<const float*>(buf.mData);
            for (unsigned i=0; i<n; ++i) dst[i] = src[i * buf.mNumberChannels + channel];
            return true;
        }
        return false;
    }
    static bool writeChannel(AudioBufferList* list, unsigned channel, const float* src, unsigned n) {
        for (unsigned b=0; b<list->mNumberBuffers; ++b) {
            auto& buf = list->mBuffers[b];
            if (channel >= buf.mNumberChannels) { channel -= buf.mNumberChannels; continue; }
            if (!buf.mData || buf.mDataByteSize != n * buf.mNumberChannels * sizeof(float)) return false;
            auto* dst = static_cast<float*>(buf.mData);
            for (unsigned i=0; i<n; ++i) dst[i * buf.mNumberChannels + channel] = src[i];
            return true;
        }
        return false;
    }
    void input(const AudioBufferList* buffers) {
        ++inputCallbacks;
        float block[512]{};
        if (!readChannel(buffers, inputChannel, block, size)) { ++callbackErrors; return; }
        ring.push(block, size);
    }
    void output(AudioBufferList* buffers) {
        if (!buffers) return;
        ++outputCallbacks;
        for (unsigned b=0; b<buffers->mNumberBuffers; ++b)
            if (buffers->mBuffers[b].mData)
                memset(buffers->mBuffers[b].mData, 0, buffers->mBuffers[b].mDataByteSize);
        float block[512]{};
        ring.pop(block, size);
        if (!writeChannel(buffers, outputChannel, block, size)) ++callbackErrors;
        if (useFile) {
            bool any = false;
            for (unsigned i=0; i<size; ++i) {
                if (filePosition == file.size() && loop) filePosition = 0;
                if (filePosition < file.size()) { block[i] = file[filePosition++]; any = true; }
                else block[i] = 0;
            }
            // Run a short zero tail through the model to drain its STFT state.
            if (!any) { if (++tailBlocks > 16) { eof.store(true); return; } }
            ring.push(block, size);
        }
    }
    unsigned tailBlocks = 0;
};
struct IOContext { Engine* engine; bool input, output; };
static OSStatus io(AudioObjectID, const AudioTimeStamp*, const AudioBufferList* input,
    const AudioTimeStamp*, AudioBufferList* output, const AudioTimeStamp*, void* context) {
    auto& c = *static_cast<IOContext*>(context);
    if (c.output) c.engine->output(output);
    if (c.input) c.engine->input(input);
    return noErr;
}

struct Device {
    AudioDeviceID id; UInt32 oldSize; Float64 oldRate; AudioDeviceIOProcID proc = nullptr;
    bool started = false, configured = false;
    explicit Device(AudioDeviceID value) : id(value),
        oldSize(get<UInt32>(id, kAudioDevicePropertyBufferFrameSize)),
        oldRate(get<Float64>(id, kAudioDevicePropertyNominalSampleRate)) {}
    void configure(unsigned size, double rate) {
        configured = true;
        if (oldRate != rate) set(id, kAudioDevicePropertyNominalSampleRate, Float64(rate));
        if (oldSize != size) set(id, kAudioDevicePropertyBufferFrameSize, UInt32(size));
        // Hardware changes may be asynchronous.
        for (int i=0; i<50; ++i) {
            if (get<UInt32>(id, kAudioDevicePropertyBufferFrameSize) == size &&
                std::abs(get<Float64>(id, kAudioDevicePropertyNominalSampleRate) - rate) < .1) return;
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        throw std::runtime_error("Device did not accept the requested sample rate / buffer size");
    }
    void validate(bool input) {
        auto f = get<AudioStreamBasicDescription>(id, kAudioDevicePropertyStreamFormat,
                input ? kAudioDevicePropertyScopeInput : kAudioDevicePropertyScopeOutput);
        if (f.mFormatID != kAudioFormatLinearPCM || !(f.mFormatFlags & kAudioFormatFlagIsFloat) || f.mBitsPerChannel != 32)
            throw std::runtime_error("Device must expose native Float32 PCM");
    }
    void start(IOContext* context) {
        check(AudioDeviceCreateIOProcID(id, io, context, &proc), "create audio callback");
        check(AudioDeviceStart(id, proc), "start audio (check macOS microphone permission)");
        started = true;
    }
    void stop() { if (started) { AudioDeviceStop(id, proc); started = false; } }
    ~Device() {
        stop();
        if (proc) AudioDeviceDestroyIOProcID(id, proc);
        if (configured) {
            try { set(id, kAudioDevicePropertyBufferFrameSize, oldSize); set(id, kAudioDevicePropertyNominalSampleRate, oldRate); }
            catch (const std::exception& e) { std::cerr << "Restore device: " << e.what() << '\n'; }
        }
    }
};

static std::vector<float> loadFile(NSString* path, double rate) {
    ExtAudioFileRef file = nullptr;
    check(ExtAudioFileOpenURL((__bridge CFURLRef)[NSURL fileURLWithPath:path], &file), "open audio file");
    try {
        AudioStreamBasicDescription source{}; UInt32 bytes = sizeof(source);
        check(ExtAudioFileGetProperty(file, kExtAudioFileProperty_FileDataFormat, &bytes, &source), "file format");
        AudioStreamBasicDescription format{};
        format.mSampleRate = rate; format.mFormatID = kAudioFormatLinearPCM;
        format.mFormatFlags = kAudioFormatFlagIsFloat | kAudioFormatFlagIsPacked;
        format.mBytesPerPacket = format.mBytesPerFrame = 4 * source.mChannelsPerFrame;
        format.mFramesPerPacket = 1; format.mChannelsPerFrame = source.mChannelsPerFrame; format.mBitsPerChannel = 32;
        check(ExtAudioFileSetProperty(file, kExtAudioFileProperty_ClientDataFormat, sizeof(format), &format), "file decode format");
        std::vector<float> result, scratch(4096 * format.mChannelsPerFrame);
        while (true) {
            UInt32 frames = 4096;
            AudioBufferList buffers{1, {{format.mChannelsPerFrame, UInt32(scratch.size()*4), scratch.data()}}};
            check(ExtAudioFileRead(file, &frames, &buffers), "decode audio file");
            if (!frames) break;
            for (unsigned i=0; i<frames; ++i) {
                float sum = 0;
                for (unsigned c=0; c<format.mChannelsPerFrame; ++c) sum += scratch[i*format.mChannelsPerFrame+c];
                result.push_back(sum / format.mChannelsPerFrame);
            }
        }
        ExtAudioFileDispose(file);
        file = nullptr;
        if (result.empty()) throw std::runtime_error("Audio file is empty");
        return result;
    } catch (...) { if (file) ExtAudioFileDispose(file); throw; }
}

struct Predictor {
    MLModel* model; MLMultiArray* audio; MLMultiArray* cache; MLMultiArray* zero;
    MLPredictionOptions* options = [MLPredictionOptions new];
    unsigned size;
    Predictor(NSString* path, unsigned n, MLComputeUnits units) : size(n) {
        NSError* error = nil;
        auto config = [MLModelConfiguration new]; config.computeUnits = units;
        model = [MLModel modelWithContentsOfURL:[NSURL fileURLWithPath:path] configuration:config error:&error];
        if (!model) throw std::runtime_error(error.localizedDescription.UTF8String);
        auto inputs = model.modelDescription.inputDescriptionsByName;
        if (!inputs[@"audio"] || !inputs[@"cache"] || !model.modelDescription.outputDescriptionsByName[@"audio_out"] ||
            !model.modelDescription.outputDescriptionsByName[@"state_out"])
            throw std::runtime_error("Expected stateless audio/cache -> audio_out/state_out model");
        auto shape = inputs[@"audio"].multiArrayConstraint.shape;
        if (shape.count != 3 || shape[0].intValue != 1 || shape[1].intValue != 1)
            throw std::runtime_error("This player requires a batch-one mono model");
        audio = [[MLMultiArray alloc] initWithShape:@[@1, @1, @(n)] dataType:MLMultiArrayDataTypeFloat32 error:&error];
        auto constraint = inputs[@"cache"].multiArrayConstraint;
        zero = [[MLMultiArray alloc] initWithShape:constraint.shape dataType:constraint.dataType error:&error];
        if (!audio || !zero) throw std::runtime_error("Cannot allocate Core ML inputs");
        for (NSInteger i=0; i<zero.count; ++i) zero[i] = @0;
        reset();
    }
    void reset() { cache = zero; }
    void run(const float* input, float* output) {
        memcpy(audio.dataPointer, input, size*sizeof(float));
        NSError* error = nil;
        auto provider = [[MLDictionaryFeatureProvider alloc] initWithDictionary:@{
            @"audio": [MLFeatureValue featureValueWithMultiArray:audio],
            @"cache": [MLFeatureValue featureValueWithMultiArray:cache]} error:&error];
        auto result = [model predictionFromFeatures:provider options:options error:&error];
        if (!result) throw std::runtime_error(error.localizedDescription.UTF8String);
        auto y = [result featureValueForName:@"audio_out"].multiArrayValue;
        auto next = [result featureValueForName:@"state_out"].multiArrayValue;
        if (!y || y.count != size || !next) throw std::runtime_error("Invalid inference output shape");
        // MLMultiArray strides may differ from contiguous tensor layout.
        auto stride = y.strides.lastObject.unsignedIntegerValue;
        if (y.dataType == MLMultiArrayDataTypeFloat32) {
            auto* data = static_cast<float*>(y.dataPointer);
            for (unsigned i=0; i<size; ++i) output[i] = data[i*stride];
        } else {
            for (unsigned i=0; i<size; ++i) output[i] = [y[@[@0,@0,@(i)]] floatValue];
        }
        for (unsigned i=0; i<size; ++i)
            if (!std::isfinite(output[i])) throw std::runtime_error("Non-finite model output");
        cache = next;
    }
};

static void worker(Engine& engine, Predictor& predictor) {
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
    uint64_t previous = 0;
    bool reset = true;
    while (engine.running.load()) {
        auto* slot = engine.ring.take();
        if (!slot) { std::this_thread::sleep_for(std::chrono::microseconds(40)); continue; }
        @autoreleasepool {
            try {
                if (reset || slot->sequence != previous + 1) predictor.reset();
                const auto start = std::chrono::steady_clock::now();
                predictor.run(slot->input.data(), slot->output.data());
                if (engine.delayMs > 0) std::this_thread::sleep_for(std::chrono::duration<double, std::milli>(engine.delayMs));
                const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-start).count();
                engine.nanos.fetch_add(ns); ++engine.calls;
                engine.lastRTF.store(ns * 1e-9 * engine.rate / engine.size);
                previous = slot->sequence;
                reset = !engine.ring.finish(*slot);
            } catch (const std::exception& e) {
                std::cerr << "Inference failed: " << e.what() << '\n';
                engine.failed.store(true); engine.running.store(false);
            }
        }
    }
}
static void usage() {
    std::cout << "Usage: realtime_coreml [options]\n"
        "  --list-devices\n  --models DIR         export directory (default exports/guitar_distill_coreml)\n"
        "  --buffer-size N      64, 128, 256 (default), 512 samples\n"
        "  --input-device ID    default system input\n  --output-device ID   default system output\n"
        "  --input-channel N    zero-based mono source (default 0)\n  --output-channel N   zero-based mono destination (default 0)\n"
        "  --file PATH          decode/resample a file before playback, replacing live input\n"
        "  --loop               loop file\n  --seconds N          stop after N seconds\n"
        "  --compute-units cpu|all|cpu-gpu|cpu-ne (default cpu)\n"
        "  --benchmark N        run N inferences without opening audio devices\n"
        "  --simulate           clocked ring test without audio hardware (uses file or silence)\n"
        "  --inject-delay-ms N  test missed deadlines by delaying inference\n";
}
int main(int argc, const char** argv) {
    @autoreleasepool {
        try {
            std::cout << std::unitbuf;
            Engine engine; std::string directory = "exports/guitar_distill_coreml", file;
            AudioDeviceID input = 0, output = 0;
            double seconds = 0; int benchmark = 0; bool simulate = false;
            MLComputeUnits units = MLComputeUnitsCPUOnly;
            for (int i=1; i<argc; ++i) {
                std::string arg = argv[i];
                auto value = [&]() -> std::string { if (++i >= argc) throw std::runtime_error("Missing value for " + arg); return argv[i]; };
                if (arg == "--help") { usage(); return 0; }
                else if (arg == "--list-devices") { listDevices(); return 0; }
                else if (arg == "--models") directory = value();
                else if (arg == "--buffer-size") engine.size = std::stoul(value());
                else if (arg == "--input-device") input = std::stoul(value());
                else if (arg == "--output-device") output = std::stoul(value());
                else if (arg == "--input-channel") engine.inputChannel = std::stoul(value());
                else if (arg == "--output-channel") engine.outputChannel = std::stoul(value());
                else if (arg == "--file") { file = value(); engine.useFile = true; }
                else if (arg == "--loop") engine.loop = true;
                else if (arg == "--seconds") seconds = std::stod(value());
                else if (arg == "--benchmark") benchmark = std::stoi(value());
                else if (arg == "--simulate") simulate = true;
                else if (arg == "--inject-delay-ms") engine.delayMs = std::stod(value());
                else if (arg == "--compute-units") {
                    auto name = value();
                    if (name == "cpu") units = MLComputeUnitsCPUOnly;
                    else if (name == "all") units = MLComputeUnitsAll;
                    else if (name == "cpu-gpu") units = MLComputeUnitsCPUAndGPU;
                    else if (name == "cpu-ne") units = MLComputeUnitsCPUAndNeuralEngine;
                    else throw std::runtime_error("Unknown compute units " + name);
                } else throw std::runtime_error("Unknown option " + arg);
            }
            if ((engine.size != 64 && engine.size != 128 && engine.size != 256 && engine.size != 512) ||
                seconds < 0 || benchmark < 0 || engine.delayMs < 0 || !std::isfinite(seconds) || !std::isfinite(engine.delayMs))
                throw std::runtime_error("Invalid buffer size, duration, repetitions or delay");
            NSString* dir = [NSString stringWithUTF8String:directory.c_str()];
            NSData* data = [NSData dataWithContentsOfFile:[dir stringByAppendingPathComponent:@"manifest.json"]];
            NSDictionary* manifest = data ? [NSJSONSerialization JSONObjectWithData:data options:0 error:nil] : nil;
            if (!manifest) throw std::runtime_error("Missing/invalid manifest.json; run the exporter first");
            engine.rate = [manifest[@"sample_rate"] doubleValue];
            if (engine.rate <= 0 || [manifest[@"channels"] intValue] != 1) throw std::runtime_error("Invalid sample rate or non-mono manifest");
            NSString* modelName = manifest[@"models"][[NSString stringWithFormat:@"%u",engine.size]][@"path"];
            if (!modelName) throw std::runtime_error("Buffer size has not been exported");
            Predictor predictor([dir stringByAppendingPathComponent:modelName], engine.size, units);
            float silence[512]{}, result[512]{};
            for (int i=0; i<20; ++i) { @autoreleasepool { predictor.run(silence, result); } }
            predictor.reset();
            if (benchmark) {
                auto start = std::chrono::steady_clock::now();
                for (int i=0; i<benchmark; ++i) { @autoreleasepool { predictor.run(silence, result); } }
                auto duration = std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()/benchmark;
                std::cout << "samples=" << engine.size << " inference_ms=" << duration*1000 << " RTF=" << duration*engine.rate/engine.size << '\n';
                return 0;
            }
            if (engine.useFile) engine.file = loadFile([NSString stringWithUTF8String:file.c_str()], engine.rate);
            std::cout << "sample_rate=" << engine.rate << " block=" << engine.size
                      << " block_ms=" << 1000*engine.size/engine.rate << " (plus device and model latency)\n";
            std::signal(SIGINT, signalHandler); std::signal(SIGTERM, signalHandler);
            std::unique_ptr<Device> inDevice, outDevice;
            IOContext inContext{&engine,true,false}, outContext{&engine,false,true};
            if (!simulate) {
                if (!output) output = get<AudioDeviceID>(kAudioObjectSystemObject,kAudioHardwarePropertyDefaultOutputDevice);
                if (!engine.useFile && !input) input = get<AudioDeviceID>(kAudioObjectSystemObject,kAudioHardwarePropertyDefaultInputDevice);
                if (engine.outputChannel >= channels(output,false)) throw std::runtime_error("Output channel unavailable");
                if (!engine.useFile && engine.inputChannel >= channels(input,true)) throw std::runtime_error("Input channel unavailable");
                outDevice = std::make_unique<Device>(output); outDevice->configure(engine.size,engine.rate); outDevice->validate(false);
                if (!engine.useFile) {
                    if (input == output) { outContext.input = true; outDevice->validate(true); }
                    else {
                        engine.ring.deadlineLead = 2;
                        inDevice = std::make_unique<Device>(input);
                        inDevice->configure(engine.size,engine.rate); inDevice->validate(true);
                    }
                }
                std::cout << "input_device=" << (engine.useFile ? 0 : input) << " output_device=" << output << '\n';
            }
            std::cout << "pipeline_blocks=" << engine.ring.deadlineLead << '\n';
            std::thread inference(worker, std::ref(engine), std::ref(predictor));
            try {
                if (inDevice) inDevice->start(&inContext);
                if (outDevice) outDevice->start(&outContext);
                auto start = std::chrono::steady_clock::now(), report = start, next = start;
                uint64_t lastCalls = 0, lastNs = 0;
                while (!stopped && !engine.failed.load() && !engine.eof.load()) {
                    auto now = std::chrono::steady_clock::now();
                    if (seconds > 0 && std::chrono::duration<double>(now-start).count() >= seconds) break;
                    if (simulate) {
                        AudioBufferList buffers{1, {{1, engine.size*4, result}}};
                        engine.output(&buffers);
                        if (!engine.useFile) engine.ring.push(silence, engine.size);
                        next += std::chrono::duration_cast<std::chrono::steady_clock::duration>(std::chrono::duration<double>(engine.size/engine.rate));
                        std::this_thread::sleep_until(next);
                    } else std::this_thread::sleep_for(std::chrono::milliseconds(20));
                    if (now-report >= std::chrono::seconds(1)) {
                        auto calls = engine.calls.load(), ns = engine.nanos.load();
                        double rtf = calls == lastCalls ? 0 : (ns-lastNs)*1e-9/(calls-lastCalls)*engine.rate/engine.size;
                        std::cout << "RTF=" << rtf << " latest_RTF=" << engine.lastRTF.load()
                                  << " dropped=" << engine.ring.dropped.load() << " silent_callbacks=" << engine.ring.silent.load()
                                  << " callback_errors=" << engine.callbackErrors.load() << '\n';
                        lastCalls = calls; lastNs = ns; report = now;
                    }
                }
            } catch (...) {
                if (inDevice) inDevice->stop(); if (outDevice) outDevice->stop();
                engine.running.store(false); inference.join(); throw;
            }
            if (inDevice) inDevice->stop(); if (outDevice) outDevice->stop();
            engine.running.store(false); inference.join();
            std::cout << "Final: processed=" << engine.calls.load() << " dropped=" << engine.ring.dropped.load()
                      << " silent_callbacks=" << engine.ring.silent.load() << " callback_errors=" << engine.callbackErrors.load() << '\n';
            if (!engine.outputCallbacks.load() || (!simulate && !engine.useFile && !engine.inputCallbacks.load())) {
                std::cerr << "No audio callbacks received; check device availability and macOS microphone permission.\n";
                return 1;
            }
            return engine.failed.load() || engine.callbackErrors.load() ? 1 : 0;
        } catch (const std::exception& e) { std::cerr << "Error: " << e.what() << '\n'; return 1; }
    }
}
