#include <iostream>
#include <vector>
#include <sstream>
#include <cstring>
#include <limits>
#include <chrono>
#include <memory>
#include <string>
#include <stdexcept>
#include <iostream>
#include <string>
#include "wav.h"
#include <cstdio>
#include <cstdarg>
#if __cplusplus < 201703L
#include <memory>
#endif

#include <fstream>

#if defined(ONNX)
#include "onnxruntime_cxx_api.h"
#else
#include <nncase/runtime/interpreter.h>
#include <nncase/runtime/runtime_tensor.h>
#include <nncase/runtime/simple_types.h>
#include <nncase/runtime/util.h>
#include <nncase/runtime/runtime_op_utility.h>
#endif

//#define __DEBUG_SPEECH_PROB___

class timestamp_t
{
public:
    int start;
    int end;

    // default + parameterized constructor
    timestamp_t(int start = -1, int end = -1)
        : start(start), end(end)
    {
    };

    // assignment operator modifies object, therefore non-const
    timestamp_t& operator=(const timestamp_t& a)
    {
        start = a.start;
        end = a.end;
        return *this;
    };

    // equality comparison. doesn't modify object. therefore const.
    bool operator==(const timestamp_t& a) const
    {
        return (start == a.start && end == a.end);
    };
    std::string c_str()
    {
        //return std::format("timestamp {:08d}, {:08d}", start, end);
        return format("{start:%08d,end:%08d}", start, end);
    };
private:

    std::string format(const char* fmt, ...)
    {
        char buf[256];

        va_list args;
        va_start(args, fmt);
        const auto r = std::vsnprintf(buf, sizeof buf, fmt, args);
        va_end(args);

        if (r < 0)
            // conversion failed
            return {};

        const size_t len = r;
        if (len < sizeof buf)
            // we fit in the buffer
            return { buf, len };

#if __cplusplus >= 201703L
        // C++17: Create a string and write to its underlying array
        std::string s(len, '\0');
        va_start(args, fmt);
        std::vsnprintf(s.data(), len + 1, fmt, args);
        va_end(args);

        return s;
#else
        // C++11 or C++14: We need to allocate scratch memory
        auto vbuf = std::unique_ptr<char[]>(new char[len + 1]);
        va_start(args, fmt);
        std::vsnprintf(vbuf.get(), len + 1, fmt, args);
        va_end(args);

        return { vbuf.get(), len };
#endif
    };
};


class VadIterator
{
private:
    virtual void predict(const std::vector<float> &data) {}

    void reset_states()
    {
        // Call reset before each audio start
        std::memset(_state.data(), 0.0f, _state.size() * sizeof(float));
        triggered = false;
        temp_end = 0;
        current_sample = 0;

        prev_end = next_start = 0;

        speeches.clear();
        current_speech = timestamp_t();
    };

public:
    void process(const std::vector<float>& input_wav)
    {
        reset_states();

        audio_length_samples = input_wav.size();
        std::cout << "window_size_samples = " << window_size_samples << ", audio_length_samples = " << audio_length_samples << std::endl;

        for (int j = 0; j < audio_length_samples; j += window_size_samples)
        {
            if (j + window_size_samples > audio_length_samples)
                break;
            std::vector<float> r{ &input_wav[0] + j, &input_wav[0] + j + window_size_samples };
            predict(r);
        }

        if (current_speech.start >= 0) {
            current_speech.end = audio_length_samples;
            speeches.push_back(current_speech);
            current_speech = timestamp_t();
            prev_end = 0;
            next_start = 0;
            temp_end = 0;
            triggered = false;
        }
    };

    void process(const std::vector<float>& input_wav, std::vector<float>& output_wav)
    {
        process(input_wav);
        collect_chunks(input_wav, output_wav);
    }

    void collect_chunks(const std::vector<float>& input_wav, std::vector<float>& output_wav)
    {
        output_wav.clear();
        for (int i = 0; i < speeches.size(); i++) {
#ifdef __DEBUG_SPEECH_PROB___
            std::cout << speeches[i].c_str() << std::endl;
#endif //#ifdef __DEBUG_SPEECH_PROB___
            std::vector<float> slice(&input_wav[speeches[i].start], &input_wav[speeches[i].end]);
            output_wav.insert(output_wav.end(),slice.begin(),slice.end());
        }
    };

    const std::vector<timestamp_t> get_speech_timestamps() const
    {
        return speeches;
    }

    void drop_chunks(const std::vector<float>& input_wav, std::vector<float>& output_wav)
    {
        output_wav.clear();
        int current_start = 0;
        for (int i = 0; i < speeches.size(); i++) {

            std::vector<float> slice(&input_wav[current_start],&input_wav[speeches[i].start]);
            output_wav.insert(output_wav.end(), slice.begin(), slice.end());
            current_start = speeches[i].end;
        }

        std::vector<float> slice(&input_wav[current_start], &input_wav[input_wav.size()]);
        output_wav.insert(output_wav.end(), slice.begin(), slice.end());
    };

protected:
    // model config
    int64_t window_size_samples;  // Assign when init, support 256 512 768 for 8k; 512 1024 1536 for 16k.
    int sample_rate;  //Assign when init support 16000 or 8000
    int sr_per_ms;   // Assign when init, support 8 or 16
    float threshold;
    int min_silence_samples; // sr_per_ms * #ms
    int min_silence_samples_at_max_speech; // sr_per_ms * #98
    int min_speech_samples; // sr_per_ms * #ms
    float max_speech_samples;
    int speech_pad_samples; // usually a
    int audio_length_samples;

    // model states
    bool triggered = false;
    unsigned int temp_end = 0;
    unsigned int current_sample = 0;
    // MAX 4294967295 samples / 8sample per ms / 1000 / 60 = 8947 minutes
    int prev_end;
    int next_start = 0;

    //Output timestamp
    std::vector<timestamp_t> speeches;
    timestamp_t current_speech;

    std::vector<const char *> input_node_names = {"input", "state", "sr"};
    std::vector<float> input;
    unsigned int size_state = 2 * 1 * 128; // It's FIXED.
    std::vector<float> _state;
    std::vector<int64_t> sr;

    int64_t input_node_dims[2] = {};
    const int64_t state_node_dims[3] = {2, 1, 128};
    const int64_t sr_node_dims[1] = {1};

    // Outputs
    std::vector<const char *> output_node_names = {"output", "stateN"};

public:
    // Construction
    VadIterator(int Sample_rate = 16000, int windows_frame_size = 32,
        float Threshold = 0.5, int min_silence_duration_ms = 0,
        int speech_pad_ms = 32, int min_speech_duration_ms = 32,
        float max_speech_duration_s = std::numeric_limits<float>::infinity())
    {
        threshold = Threshold;
        sample_rate = Sample_rate;
        sr_per_ms = sample_rate / 1000;

        window_size_samples = windows_frame_size * sr_per_ms;

        min_speech_samples = sr_per_ms * min_speech_duration_ms;
        speech_pad_samples = sr_per_ms * speech_pad_ms;

        max_speech_samples = (
            sample_rate * max_speech_duration_s
            - window_size_samples
            - 2 * speech_pad_samples
            );

        min_silence_samples = sr_per_ms * min_silence_duration_ms;
        min_silence_samples_at_max_speech = sr_per_ms * 98;

        input.resize(window_size_samples);
        input_node_dims[0] = 1;
        input_node_dims[1] = window_size_samples;

        _state.resize(size_state);
        sr.resize(1);
        sr[0] = sample_rate;
    };
};

#if defined ONNX

class OnnxVadIterator: public VadIterator
{
private:
    // OnnxRuntime resources
    Ort::Env env;
    Ort::SessionOptions session_options;
    std::shared_ptr<Ort::Session> session = nullptr;
    Ort::AllocatorWithDefaultOptions allocator;
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeCPU);

private:
    void init_engine_threads(int inter_threads, int intra_threads)
    {
        // The method should be called in each thread/proc in multi-thread/proc work
        session_options.SetIntraOpNumThreads(intra_threads);
        session_options.SetInterOpNumThreads(inter_threads);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    };

    void init_model(const std::string& model_path)
    {
        // Init threads = 1 for
        init_engine_threads(1, 1);
        // Load model
        session = std::make_shared<Ort::Session>(env, model_path.c_str(), session_options);
    };

    void predict(const std::vector<float> &data)
    {
        // Infer
        // Create ort tensors
        input.assign(data.begin(), data.end());
        Ort::Value input_ort = Ort::Value::CreateTensor<float>(
            memory_info, input.data(), input.size(), input_node_dims, 2);
        Ort::Value state_ort = Ort::Value::CreateTensor<float>(
            memory_info, _state.data(), _state.size(), state_node_dims, 3);
        Ort::Value sr_ort = Ort::Value::CreateTensor<int64_t>(
            memory_info, sr.data(), sr.size(), sr_node_dims, 1);

        // Clear and add inputs
        ort_inputs.clear();
        ort_inputs.emplace_back(std::move(input_ort));
        ort_inputs.emplace_back(std::move(state_ort));
        ort_inputs.emplace_back(std::move(sr_ort));

        // Infer
        ort_outputs = session->Run(
            Ort::RunOptions{nullptr},
            input_node_names.data(), ort_inputs.data(), ort_inputs.size(),
            output_node_names.data(), output_node_names.size());

        // Output probability & update h,c recursively
        float speech_prob = ort_outputs[0].GetTensorMutableData<float>()[0];
        // std::cout << "speech_prob = " << speech_prob << std::endl;
        float *stateN = ort_outputs[1].GetTensorMutableData<float>();
        std::memcpy(_state.data(), stateN, size_state * sizeof(float));

        // Push forward sample index
        current_sample += window_size_samples;

        // Reset temp_end when > threshold
        if ((speech_prob >= threshold))
        {
#ifdef __DEBUG_SPEECH_PROB___
            float speech = current_sample - window_size_samples; // minus window_size_samples to get precise start time point.
            printf("{    start: %.3f s (%.3f) %08d}\n", 1.0 * speech / sample_rate, speech_prob, current_sample- window_size_samples);
#endif //__DEBUG_SPEECH_PROB___
            if (temp_end != 0)
            {
                temp_end = 0;
                if (next_start < prev_end)
                    next_start = current_sample - window_size_samples;
            }
            if (triggered == false)
            {
                triggered = true;

                current_speech.start = current_sample - window_size_samples;
            }
            return;
        }

        if (
            (triggered == true)
            && ((current_sample - current_speech.start) > max_speech_samples)
            ) {
            if (prev_end > 0) {
                current_speech.end = prev_end;
                speeches.push_back(current_speech);
                current_speech = timestamp_t();

                // previously reached silence(< neg_thres) and is still not speech(< thres)
                if (next_start < prev_end)
                    triggered = false;
                else{
                    current_speech.start = next_start;
                }
                prev_end = 0;
                next_start = 0;
                temp_end = 0;

            }
            else{
                current_speech.end = current_sample;
                speeches.push_back(current_speech);
                current_speech = timestamp_t();
                prev_end = 0;
                next_start = 0;
                temp_end = 0;
                triggered = false;
            }
            return;

        }
        if ((speech_prob >= (threshold - 0.15)) && (speech_prob < threshold))
        {
            if (triggered) {
#ifdef __DEBUG_SPEECH_PROB___
                float speech = current_sample - window_size_samples; // minus window_size_samples to get precise start time point.
                printf("{ speeking: %.3f s (%.3f) %08d}\n", 1.0 * speech / sample_rate, speech_prob, current_sample - window_size_samples);
#endif //__DEBUG_SPEECH_PROB___
            }
            else {
#ifdef __DEBUG_SPEECH_PROB___
                float speech = current_sample - window_size_samples; // minus window_size_samples to get precise start time point.
                printf("{  silence: %.3f s (%.3f) %08d}\n", 1.0 * speech / sample_rate, speech_prob, current_sample - window_size_samples);
#endif //__DEBUG_SPEECH_PROB___
            }
            return;
        }


        // 4) End
        if ((speech_prob < (threshold - 0.15)))
        {
#ifdef __DEBUG_SPEECH_PROB___
            float speech = current_sample - window_size_samples - speech_pad_samples; // minus window_size_samples to get precise start time point.
            printf("{      end: %.3f s (%.3f) %08d}\n", 1.0 * speech / sample_rate, speech_prob, current_sample - window_size_samples);
#endif //__DEBUG_SPEECH_PROB___
            if (triggered == true)
            {
                if (temp_end == 0)
                {
                    temp_end = current_sample;
                }
                if (current_sample - temp_end > min_silence_samples_at_max_speech)
                    prev_end = temp_end;
                // a. silence < min_slience_samples, continue speaking
                if ((current_sample - temp_end) < min_silence_samples)
                {

                }
                // b. silence >= min_slience_samples, end speaking
                else
                {
                    current_speech.end = temp_end;
                    if (current_speech.end - current_speech.start > min_speech_samples)
                    {
                        speeches.push_back(current_speech);
                        current_speech = timestamp_t();
                        prev_end = 0;
                        next_start = 0;
                        temp_end = 0;
                        triggered = false;
                    }
                }
            }
            else {
                // may first windows see end state.
            }
            return;
        }
    };

private:
    // Onnx model
    // Inputs
    std::vector<Ort::Value> ort_inputs;

    // Outputs
    std::vector<Ort::Value> ort_outputs;

public:
    // Construction
    OnnxVadIterator(const std::string ModelPath,
        int Sample_rate = 16000, int windows_frame_size = 32,
        float Threshold = 0.5, int min_silence_duration_ms = 0,
        int speech_pad_ms = 32, int min_speech_duration_ms = 32,
        float max_speech_duration_s = std::numeric_limits<float>::infinity()): VadIterator(Sample_rate, windows_frame_size,
        Threshold, min_silence_duration_ms, speech_pad_ms, min_speech_duration_ms, max_speech_duration_s)
    {
        init_model(ModelPath);
    }
};
#else
#define NNCASE_DUMP_BIN 0
class NncaseVadIterator: public VadIterator
{
private:
    void init_model(const std::string& model_path)
    {
        std::ifstream ifs(model_path, std::ios::binary);
        interpreter_.load_model(ifs).unwrap_or_throw();
        entry_function_ = interpreter_.entry_function().unwrap_or_throw();
    };

#if NNCASE_DUMP_BIN
    void dump_to_bin(const char *file_name, const char *buf, size_t size)
    {
        std::ofstream ofs(file_name, std::ios::out | std::ios::binary);
        ofs.write(buf, size);
        ofs.close();
    }
#endif
    void predict(const std::vector<float> &data)
    {
        // Infer
        std::vector<nncase::value_t> inputs;
#if NNCASE_DUMP_BIN
        static size_t count = 0;
#endif

        // set input1
        input.assign(data.begin(), data.end());
        auto type = entry_function_->parameter_type(0).expect("parameter type out of index");
        auto ts_type = type.as<nncase::tensor_type>().expect("input is not a tensor type");
        auto data_type = ts_type->dtype()->typecode();
        std::vector<size_t> tmp1 { input_node_dims[0], input_node_dims[1]};
        nncase::dims_t shape1(tmp1.begin(), tmp1.end());
        auto input_tensor = nncase::runtime::host_runtime_tensor::create(data_type, shape1, nncase::runtime::host_runtime_tensor::pool_shared).expect("cannot create input tensor").impl();
        auto input_buffer = input_tensor->buffer().as_host().unwrap_or_throw();
        auto input_mapped = input_buffer.map(nncase::runtime::map_write).unwrap_or_throw();
        auto input_ptr = input_mapped.buffer().as_span<float>().data();
        memcpy(reinterpret_cast<void *>(input_ptr), reinterpret_cast<const void *>(input.data()), input.size() * sizeof(float));
        input_buffer.sync(nncase::runtime::sync_write_back, true).unwrap_or_throw();
        inputs.push_back(input_tensor);
#if NNCASE_DUMP_BIN
        char file_name[64] = "\0";
        snprintf(file_name, sizeof(file_name) / sizeof(file_name[0]), "tmp/input_%08lu.bin", count);
        dump_to_bin(file_name, reinterpret_cast<const char *>(input_ptr), input.size() * sizeof(float));
#endif

        // set input2
        type = entry_function_->parameter_type(1).expect("parameter type out of index");
        ts_type = type.as<nncase::tensor_type>().expect("input is not a tensor type");
        data_type = ts_type->dtype()->typecode();
        std::vector<size_t> tmp2 { state_node_dims[0], state_node_dims[1], state_node_dims[2] };
        nncase::dims_t shape2(tmp2.begin(), tmp2.end());
        auto state_tensor = nncase::runtime::host_runtime_tensor::create(data_type, shape2, nncase::runtime::host_runtime_tensor::pool_shared).expect("cannot create input tensor").impl();
        auto state_buffer = state_tensor->buffer().as_host().unwrap_or_throw();
        auto state_mapped = state_buffer.map(nncase::runtime::map_write).unwrap_or_throw();
        auto state_ptr = state_mapped.buffer().as_span<float>().data();
        memcpy(reinterpret_cast<void *>(state_ptr), reinterpret_cast<const void *>(_state.data()), _state.size() * sizeof(float));
        state_buffer.sync(nncase::runtime::sync_write_back, true).unwrap_or_throw();
        inputs.push_back(state_tensor);
#if NNCASE_DUMP_BIN
        snprintf(file_name, sizeof(file_name) / sizeof(file_name[0]), "tmp/state_%08lu.bin", count);
        dump_to_bin(file_name, reinterpret_cast<const char *>(state_ptr), _state.size() * sizeof(float));
#endif

        // set input3
        type = entry_function_->parameter_type(2).expect("parameter type out of index");
        ts_type = type.as<nncase::tensor_type>().expect("input is not a tensor type");
        data_type = ts_type->dtype()->typecode();
        std::vector<size_t> tmp3 { 1 };
        nncase::dims_t shape3(tmp3.begin(), tmp3.end());
        auto sr_tensor = nncase::runtime::host_runtime_tensor::create(data_type, shape3, nncase::runtime::host_runtime_tensor::pool_shared).expect("cannot create input tensor").impl();
        auto sr_buffer = sr_tensor->buffer().as_host().unwrap_or_throw();
        auto sr_mapped = sr_buffer.map(nncase::runtime::map_write).unwrap_or_throw();
        auto sr_ptr = sr_mapped.buffer().as_span<int64_t>().data();
        sr_ptr[0] = sr[0];
        sr_buffer.sync(nncase::runtime::sync_write_back, true).unwrap_or_throw();
        inputs.push_back(sr_tensor);
#if NNCASE_DUMP_BIN
        snprintf(file_name, sizeof(file_name) / sizeof(file_name[0]), "tmp/sr_%08lu.bin", count);
        dump_to_bin(file_name, reinterpret_cast<const char *>(sr_ptr), sizeof(int64_t));
        count++;
#endif

        // Infer
        auto outputs = entry_function_->invoke(inputs).unwrap_or_throw().as<nncase::tuple>().unwrap_or_throw();

        // output1
        auto output = outputs->fields()[0].as<nncase::tensor>().unwrap_or_throw();
        auto output_buffer = output->buffer().as_host().unwrap_or_throw();
        auto output_mapped = output_buffer.map(nncase::runtime::map_read).unwrap_or_throw();
        auto output_span = output_mapped.buffer().as_span<float>();
        // Output probability & update h,c recursively
        float speech_prob = output_span.data()[0];
        // std::cout << "speech_prob = " << speech_prob << std::endl;

        // output2
        auto stateN_tensor = outputs->fields()[1].as<nncase::tensor>().unwrap_or_throw();
        auto stateN_buffer = stateN_tensor->buffer().as_host().unwrap_or_throw();
        auto stateN_mapped = stateN_buffer.map(nncase::runtime::map_read).unwrap_or_throw();
        auto stateN_span = stateN_mapped.buffer().as_span<float>();
        float *stateN = stateN_span.data();
        std::memcpy(_state.data(), stateN, size_state * sizeof(float));

        // Push forward sample index
        current_sample += window_size_samples;

        // Reset temp_end when > threshold
        if ((speech_prob >= threshold))
        {
#ifdef __DEBUG_SPEECH_PROB___
            float speech = current_sample - window_size_samples; // minus window_size_samples to get precise start time point.
            printf("{    start: %.3f s (%.3f) %08d}\n", 1.0 * speech / sample_rate, speech_prob, current_sample- window_size_samples);
#endif //__DEBUG_SPEECH_PROB___
            if (temp_end != 0)
            {
                temp_end = 0;
                if (next_start < prev_end)
                    next_start = current_sample - window_size_samples;
            }
            if (triggered == false)
            {
                triggered = true;

                current_speech.start = current_sample - window_size_samples;
            }
            return;
        }

        if (
            (triggered == true)
            && ((current_sample - current_speech.start) > max_speech_samples)
            ) {
            if (prev_end > 0) {
                current_speech.end = prev_end;
                speeches.push_back(current_speech);
                current_speech = timestamp_t();

                // previously reached silence(< neg_thres) and is still not speech(< thres)
                if (next_start < prev_end)
                    triggered = false;
                else{
                    current_speech.start = next_start;
                }
                prev_end = 0;
                next_start = 0;
                temp_end = 0;

            }
            else{
                current_speech.end = current_sample;
                speeches.push_back(current_speech);
                current_speech = timestamp_t();
                prev_end = 0;
                next_start = 0;
                temp_end = 0;
                triggered = false;
            }
            return;

        }
        if ((speech_prob >= (threshold - 0.15)) && (speech_prob < threshold))
        {
            if (triggered) {
#ifdef __DEBUG_SPEECH_PROB___
                float speech = current_sample - window_size_samples; // minus window_size_samples to get precise start time point.
                printf("{ speeking: %.3f s (%.3f) %08d}\n", 1.0 * speech / sample_rate, speech_prob, current_sample - window_size_samples);
#endif //__DEBUG_SPEECH_PROB___
            }
            else {
#ifdef __DEBUG_SPEECH_PROB___
                float speech = current_sample - window_size_samples; // minus window_size_samples to get precise start time point.
                printf("{  silence: %.3f s (%.3f) %08d}\n", 1.0 * speech / sample_rate, speech_prob, current_sample - window_size_samples);
#endif //__DEBUG_SPEECH_PROB___
            }
            return;
        }


        // 4) End
        if ((speech_prob < (threshold - 0.15)))
        {
#ifdef __DEBUG_SPEECH_PROB___
            float speech = current_sample - window_size_samples - speech_pad_samples; // minus window_size_samples to get precise start time point.
            printf("{      end: %.3f s (%.3f) %08d}\n", 1.0 * speech / sample_rate, speech_prob, current_sample - window_size_samples);
#endif //__DEBUG_SPEECH_PROB___
            if (triggered == true)
            {
                if (temp_end == 0)
                {
                    temp_end = current_sample;
                }
                if (current_sample - temp_end > min_silence_samples_at_max_speech)
                    prev_end = temp_end;
                // a. silence < min_slience_samples, continue speaking
                if ((current_sample - temp_end) < min_silence_samples)
                {

                }
                // b. silence >= min_slience_samples, end speaking
                else
                {
                    current_speech.end = temp_end;
                    if (current_speech.end - current_speech.start > min_speech_samples)
                    {
                        speeches.push_back(current_speech);
                        current_speech = timestamp_t();
                        prev_end = 0;
                        next_start = 0;
                        temp_end = 0;
                        triggered = false;
                    }
                }
            }
            else {
                // may first windows see end state.
            }
            return;
        }
    };

private:
    nncase::runtime::interpreter interpreter_;
    nncase::runtime::runtime_function *entry_function_;

public:
    // Construction
    NncaseVadIterator(const std::string ModelPath,
        int Sample_rate = 16000, int windows_frame_size = 32,
        float Threshold = 0.5, int min_silence_duration_ms = 0,
        int speech_pad_ms = 32, int min_speech_duration_ms = 32,
        float max_speech_duration_s = std::numeric_limits<float>::infinity()): VadIterator(Sample_rate, windows_frame_size,
        Threshold, min_silence_duration_ms, speech_pad_ms, min_speech_duration_ms, max_speech_duration_s)
    {
        init_model(ModelPath);
    }
};
#endif

int main(int argc, char *argv[])
{
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " wav_file onnx_file | kmodel_file" << std::endl;
        return 1;
    }

    std::vector<timestamp_t> stamps;

    // Read wav
    wav::WavReader wav_reader(argv[1]); //16000,1,32float
    std::vector<float> input_wav(wav_reader.num_samples());
    std::vector<float> output_wav;

    for (int i = 0; i < wav_reader.num_samples(); i++)
    {
        input_wav[i] = static_cast<float>(*(wav_reader.data() + i));
    }


    // ===== Test configs =====
    std::unique_ptr<VadIterator> vad;
    std::string path(argv[2]);

#if defined ONNX
    vad.reset(new OnnxVadIterator(path));
#else
    vad.reset(new NncaseVadIterator(path));
#endif

    // ==============================================
    // ==== = Example 1 of full function  =====
    // ==============================================
    std::cout << "example 1" << std::endl;
    vad->process(input_wav);

    // 1.a get_speech_timestamps
    stamps = vad->get_speech_timestamps();
    for (int i = 0; i < stamps.size(); i++) {

        std::cout << stamps[i].c_str() << std::endl;
    }

    // 1.b collect_chunks output wav
    vad->collect_chunks(input_wav, output_wav);

    // 1.c drop_chunks output wav
    vad->drop_chunks(input_wav, output_wav);

    // ==============================================
    // ===== Example 2 of simple full function  =====
    // ==============================================
    // std::cout << "example 2" << std::endl;
    // vad->process(input_wav, output_wav);

    // stamps = vad->get_speech_timestamps();
    // for (int i = 0; i < stamps.size(); i++) {

    //     std::cout << stamps[i].c_str() << std::endl;
    // }

    // ==============================================
    // ===== Example 3 of full function  =====
    // ==============================================
    // std::cout << "example 3" << std::endl;
    // for(int i = 0; i<2; i++)
    //     vad->process(input_wav, output_wav);
}
