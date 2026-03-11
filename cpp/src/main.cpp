#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <yaml-cpp/yaml.h>

#ifdef PIPELINE_ENABLE_ORT
#include <onnxruntime_cxx_api.h>
#endif

#ifdef PIPELINE_ENABLE_TENSORRT
#include <cuda_runtime_api.h>
#include <NvInfer.h>
#endif

namespace fs = std::filesystem;

struct Timer {
    using Clock = std::chrono::high_resolution_clock;
    Clock::time_point start = Clock::now();

    void reset() { start = Clock::now(); }

    double elapsed_ms() const {
        const auto end = Clock::now();
        return std::chrono::duration<double, std::milli>(end - start).count();
    }
};

struct Args {
    fs::path config_path;
    std::string backend;
    fs::path model_path;
    fs::path image_dir;
    fs::path report_path;
    std::string precision = "fp32";
    int warmup = -1;
    int runs = -1;
    bool quantized = false;
};

struct Config {
    int preprocess_imgsz = 640;
    bool bgr_to_rgb = true;
    bool normalize = true;
    std::array<float, 3> mean{0.0f, 0.0f, 0.0f};
    std::array<float, 3> std{255.0f, 255.0f, 255.0f};
    float conf = 0.25f;
    int max_det = 300;
    int warmup = 10;
    int runs = 100;
    fs::path report_dir = "reports";
    fs::path val_images = "data/images/val";
};

struct PreprocessResult {
    std::vector<float> tensor;
    std::vector<int64_t> shape;
    double elapsed_ms = 0.0;
};

struct PostprocessResult {
    int kept = 0;
    double elapsed_ms = 0.0;
};

struct TimingMetrics {
    std::vector<double> preprocess_ms;
    std::vector<double> inference_ms;
    std::vector<double> postprocess_ms;
    std::vector<double> total_ms;
};

double mean_value(const std::vector<double>& values) {
    if (values.empty()) {
        return 0.0;
    }
    const double sum = std::accumulate(values.begin(), values.end(), 0.0);
    return sum / static_cast<double>(values.size());
}

std::string json_escape(const std::string& text) {
    std::ostringstream oss;
    for (const char c : text) {
        switch (c) {
        case '\\':
            oss << "\\\\";
            break;
        case '"':
            oss << "\\\"";
            break;
        case '\n':
            oss << "\\n";
            break;
        case '\r':
            oss << "\\r";
            break;
        case '\t':
            oss << "\\t";
            break;
        default:
            oss << c;
            break;
        }
    }
    return oss.str();
}

bool is_image_file(const fs::path& path) {
    std::string ext = path.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp";
}

std::vector<fs::path> collect_images(const fs::path& directory) {
    if (!fs::exists(directory)) {
        throw std::runtime_error("Image directory does not exist: " + directory.string());
    }

    std::vector<fs::path> images;
    for (const auto& entry : fs::directory_iterator(directory)) {
        if (entry.is_regular_file() && is_image_file(entry.path())) {
            images.push_back(entry.path());
        }
    }
    std::sort(images.begin(), images.end());

    if (images.empty()) {
        throw std::runtime_error("No images found in: " + directory.string());
    }
    return images;
}

cv::Mat letterbox(const cv::Mat& image, int new_shape) {
    const int original_h = image.rows;
    const int original_w = image.cols;
    const float ratio = std::min(static_cast<float>(new_shape) / static_cast<float>(original_h),
                                 static_cast<float>(new_shape) / static_cast<float>(original_w));

    const int resized_w = static_cast<int>(std::round(static_cast<float>(original_w) * ratio));
    const int resized_h = static_cast<int>(std::round(static_cast<float>(original_h) * ratio));

    cv::Mat resized;
    if (resized_w != original_w || resized_h != original_h) {
        cv::resize(image, resized, cv::Size(resized_w, resized_h), 0.0, 0.0, cv::INTER_LINEAR);
    } else {
        resized = image.clone();
    }

    const int dw = new_shape - resized_w;
    const int dh = new_shape - resized_h;
    const int left = dw / 2;
    const int right = dw - left;
    const int top = dh / 2;
    const int bottom = dh - top;

    cv::Mat output;
    cv::copyMakeBorder(resized, output, top, bottom, left, right, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
    return output;
}

PreprocessResult preprocess_image(const cv::Mat& image, const Config& config) {
    Timer timer;
    cv::Mat processed = letterbox(image, config.preprocess_imgsz);

    if (config.bgr_to_rgb) {
        cv::cvtColor(processed, processed, cv::COLOR_BGR2RGB);
    }

    processed.convertTo(processed, CV_32FC3);

    const int height = processed.rows;
    const int width = processed.cols;
    std::vector<float> tensor(static_cast<size_t>(3 * height * width), 0.0f);

    for (int y = 0; y < height; ++y) {
        const auto* row = processed.ptr<cv::Vec3f>(y);
        for (int x = 0; x < width; ++x) {
            const cv::Vec3f pixel = row[x];
            for (int c = 0; c < 3; ++c) {
                const float value = config.normalize ? (pixel[c] - config.mean[c]) / config.std[c] : pixel[c];
                tensor[static_cast<size_t>(c * height * width + y * width + x)] = value;
            }
        }
    }

    return {std::move(tensor), {1, 3, height, width}, timer.elapsed_ms()};
}

PostprocessResult postprocess_predictions(const std::vector<float>& output, const std::vector<int64_t>& shape, float conf, int max_det) {
    Timer timer;
    int kept = 0;

    if (shape.size() >= 2) {
        const int64_t rows = shape[shape.size() - 2];
        const int64_t cols = shape[shape.size() - 1];
        if (cols >= 6) {
            const int64_t limit = std::min<int64_t>(rows, max_det);
            for (int64_t i = 0; i < limit; ++i) {
                const size_t score_index = static_cast<size_t>(i * cols + 4);
                if (score_index < output.size() && output[score_index] >= conf) {
                    ++kept;
                }
            }
        }
    }

    return {kept, timer.elapsed_ms()};
}

class IBackend {
public:
    virtual ~IBackend() = default;
    virtual void warmup(const std::vector<float>& input, const std::vector<int64_t>& shape, int steps) = 0;
    virtual std::pair<std::vector<float>, std::vector<int64_t>> infer(const std::vector<float>& input, const std::vector<int64_t>& shape, double& elapsed_ms) = 0;
};

#ifdef PIPELINE_ENABLE_ORT
class OnnxRuntimeBackend final : public IBackend {
public:
    explicit OnnxRuntimeBackend(const fs::path& model_path)
        : env_(ORT_LOGGING_LEVEL_WARNING, "pipeline_benchmark"), session_options_() {
        session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
#ifdef _WIN32
        session_ = std::make_unique<Ort::Session>(env_, model_path.wstring().c_str(), session_options_);
#else
        session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(), session_options_);
#endif

        Ort::AllocatorWithDefaultOptions allocator;
        auto input_name = session_->GetInputNameAllocated(0, allocator);
        input_name_ = input_name.get();

        const size_t output_count = session_->GetOutputCount();
        output_names_storage_.reserve(output_count);
        for (size_t i = 0; i < output_count; ++i) {
            auto output_name = session_->GetOutputNameAllocated(i, allocator);
            output_names_storage_.push_back(output_name.get());
        }
        output_name_ptrs_.reserve(output_names_storage_.size());
        for (const auto& name : output_names_storage_) {
            output_name_ptrs_.push_back(name.c_str());
        }
    }

    void warmup(const std::vector<float>& input, const std::vector<int64_t>& shape, int steps) override {
        for (int i = 0; i < steps; ++i) {
            double elapsed_ms = 0.0;
            (void)infer(input, shape, elapsed_ms);
        }
    }

    std::pair<std::vector<float>, std::vector<int64_t>> infer(const std::vector<float>& input, const std::vector<int64_t>& shape, double& elapsed_ms) override {
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info,
            const_cast<float*>(input.data()),
            input.size(),
            const_cast<int64_t*>(shape.data()),
            shape.size());

        const char* input_names[] = {input_name_.c_str()};
        Timer timer;
        auto outputs = session_->Run(
            Ort::RunOptions{nullptr},
            input_names,
            &input_tensor,
            1,
            output_name_ptrs_.data(),
            output_name_ptrs_.size());
        elapsed_ms = timer.elapsed_ms();

        if (outputs.empty()) {
            return {{}, {}};
        }

        auto& first = outputs.front();
        auto type_info = first.GetTensorTypeAndShapeInfo();
        auto output_shape = type_info.GetShape();
        const size_t count = type_info.GetElementCount();
        const float* output_data = first.GetTensorData<float>();
        return {std::vector<float>(output_data, output_data + count), output_shape};
    }

private:
    Ort::Env env_;
    Ort::SessionOptions session_options_;
    std::unique_ptr<Ort::Session> session_;
    std::string input_name_;
    std::vector<std::string> output_names_storage_;
    std::vector<const char*> output_name_ptrs_;
};
#endif

#ifdef PIPELINE_ENABLE_TENSORRT
class TrtLogger final : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cerr << "[TensorRT] " << msg << '\n';
        }
    }
};

class TensorRtBackend final : public IBackend {
public:
    explicit TensorRtBackend(const fs::path& model_path) {
        std::ifstream stream(model_path, std::ios::binary);
        if (!stream) {
            throw std::runtime_error("Unable to open TensorRT engine: " + model_path.string());
        }

        stream.seekg(0, std::ios::end);
        const auto size = static_cast<size_t>(stream.tellg());
        stream.seekg(0, std::ios::beg);
        std::vector<char> bytes(size);
        stream.read(bytes.data(), static_cast<std::streamsize>(size));

        runtime_.reset(nvinfer1::createInferRuntime(logger_));
        engine_.reset(runtime_->deserializeCudaEngine(bytes.data(), size));
        context_.reset(engine_->createExecutionContext());

        if (!runtime_ || !engine_ || !context_) {
            throw std::runtime_error("Failed to initialize TensorRT runtime");
        }

        input_index_ = 0;
        output_index_ = 1;

        const auto input_dims = engine_->getBindingDimensions(input_index_);
        const auto output_dims = engine_->getBindingDimensions(output_index_);

        size_t input_elements = 1;
        for (int i = 0; i < input_dims.nbDims; ++i) {
            input_shape_.push_back(input_dims.d[i]);
            input_elements *= static_cast<size_t>(input_dims.d[i]);
        }

        size_t output_elements = 1;
        for (int i = 0; i < output_dims.nbDims; ++i) {
            output_shape_.push_back(output_dims.d[i]);
            output_elements *= static_cast<size_t>(output_dims.d[i]);
        }

        cudaMalloc(&device_input_, input_elements * sizeof(float));
        cudaMalloc(&device_output_, output_elements * sizeof(float));
        cudaStreamCreate(&stream_);

        bindings_.resize(engine_->getNbBindings(), nullptr);
        bindings_[input_index_] = device_input_;
        bindings_[output_index_] = device_output_;
        output_buffer_.resize(output_elements);
    }

    ~TensorRtBackend() override {
        if (stream_ != nullptr) {
            cudaStreamDestroy(stream_);
        }
        if (device_input_ != nullptr) {
            cudaFree(device_input_);
        }
        if (device_output_ != nullptr) {
            cudaFree(device_output_);
        }
    }

    void warmup(const std::vector<float>& input, const std::vector<int64_t>& shape, int steps) override {
        for (int i = 0; i < steps; ++i) {
            double elapsed_ms = 0.0;
            (void)infer(input, shape, elapsed_ms);
        }
    }

    std::pair<std::vector<float>, std::vector<int64_t>> infer(const std::vector<float>& input, const std::vector<int64_t>&, double& elapsed_ms) override {
        cudaMemcpyAsync(device_input_, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice, stream_);
        Timer timer;
        context_->enqueueV2(bindings_.data(), stream_, nullptr);
        cudaMemcpyAsync(output_buffer_.data(), device_output_, output_buffer_.size() * sizeof(float), cudaMemcpyDeviceToHost, stream_);
        cudaStreamSynchronize(stream_);
        elapsed_ms = timer.elapsed_ms();
        return {output_buffer_, output_shape_};
    }

private:
    struct NvInferDeleter {
        template <typename T>
        void operator()(T* ptr) const {
            delete ptr;
        }
    };

    TrtLogger logger_;
    std::unique_ptr<nvinfer1::IRuntime, NvInferDeleter> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine, NvInferDeleter> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext, NvInferDeleter> context_;
    std::vector<void*> bindings_;
    int input_index_ = -1;
    int output_index_ = -1;
    std::vector<int64_t> input_shape_;
    std::vector<int64_t> output_shape_;
    void* device_input_ = nullptr;
    void* device_output_ = nullptr;
    cudaStream_t stream_ = nullptr;
    std::vector<float> output_buffer_;
};
#endif

std::unique_ptr<IBackend> build_backend(const std::string& backend, const fs::path& model_path) {
    std::string lower = backend;
    std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });

#ifdef PIPELINE_ENABLE_ORT
    if (lower == "onnx") {
        return std::make_unique<OnnxRuntimeBackend>(model_path);
    }
#endif

#ifdef PIPELINE_ENABLE_TENSORRT
    if (lower == "tensorrt" || lower == "engine") {
        return std::make_unique<TensorRtBackend>(model_path);
    }
#endif

    throw std::runtime_error("Unsupported backend or backend not enabled at build time: " + backend);
}

Config load_config(const fs::path& path) {
    const YAML::Node root = YAML::LoadFile(path.string());
    Config config;

    if (const auto preprocess = root["preprocess"]) {
        if (preprocess["imgsz"]) {
            config.preprocess_imgsz = preprocess["imgsz"].as<int>();
        }
        if (preprocess["bgr_to_rgb"]) {
            config.bgr_to_rgb = preprocess["bgr_to_rgb"].as<bool>();
        }
        if (preprocess["normalize"]) {
            config.normalize = preprocess["normalize"].as<bool>();
        }
        if (preprocess["mean"] && preprocess["mean"].size() == 3) {
            for (size_t i = 0; i < 3; ++i) {
                config.mean[i] = preprocess["mean"][i].as<float>();
            }
        }
        if (preprocess["std"] && preprocess["std"].size() == 3) {
            for (size_t i = 0; i < 3; ++i) {
                config.std[i] = preprocess["std"][i].as<float>();
            }
        }
    }

    if (const auto postprocess = root["postprocess"]) {
        if (postprocess["conf"]) {
            config.conf = postprocess["conf"].as<float>();
        }
        if (postprocess["max_det"]) {
            config.max_det = postprocess["max_det"].as<int>();
        }
    }

    if (const auto benchmark = root["benchmark"]) {
        if (benchmark["warmup"]) {
            config.warmup = benchmark["warmup"].as<int>();
        }
        if (benchmark["runs"]) {
            config.runs = benchmark["runs"].as<int>();
        }
    }

    if (const auto project = root["project"]) {
        if (project["report_dir"]) {
            config.report_dir = project["report_dir"].as<std::string>();
        }
    }

    if (const auto dataset = root["dataset"]) {
        if (dataset["val_images"]) {
            config.val_images = dataset["val_images"].as<std::string>();
        }
    }

    return config;
}

std::string build_report(const std::string& backend,
                         const fs::path& model_path,
                         const std::string& precision,
                         bool quantized,
                         const TimingMetrics& timing) {
    const double preprocess_ms = mean_value(timing.preprocess_ms);
    const double inference_ms = mean_value(timing.inference_ms);
    const double postprocess_ms = mean_value(timing.postprocess_ms);
    const double total_ms = mean_value(timing.total_ms);
    const double fps = total_ms > 0.0 ? 1000.0 / total_ms : 0.0;

    std::ostringstream oss;
    oss << std::fixed << std::setprecision(4);
    oss << "{\n";
    oss << "  \"backend\": \"" << json_escape(backend) << "\",\n";
    oss << "  \"model_path\": \"" << json_escape(model_path.string()) << "\",\n";
    oss << "  \"precision\": \"" << json_escape(precision) << "\",\n";
    oss << "  \"quantized\": " << (quantized ? "true" : "false") << ",\n";
    oss << "  \"timing\": {\n";
    oss << "    \"preprocess_ms\": " << preprocess_ms << ",\n";
    oss << "    \"inference_ms\": " << inference_ms << ",\n";
    oss << "    \"postprocess_ms\": " << postprocess_ms << ",\n";
    oss << "    \"total_ms\": " << total_ms << ",\n";
    oss << "    \"fps\": " << fps << "\n";
    oss << "  }\n";
    oss << "}\n";
    return oss.str();
}

Args parse_args(int argc, char** argv) {
    Args args;
    std::unordered_map<std::string, std::string> values;

    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (key == "--quantized") {
            args.quantized = true;
            continue;
        }
        if (i + 1 >= argc) {
            throw std::runtime_error("Missing value for argument: " + key);
        }
        values[key] = argv[++i];
    }

    auto require = [&](const std::string& key) -> std::string {
        const auto it = values.find(key);
        if (it == values.end() || it->second.empty()) {
            throw std::runtime_error("Required argument missing: " + key);
        }
        return it->second;
    };

    args.config_path = require("--config");
    args.backend = require("--backend");
    args.model_path = require("--model");
    if (values.count("--images")) {
        args.image_dir = values["--images"];
    }
    if (values.count("--report")) {
        args.report_path = values["--report"];
    }
    if (values.count("--precision")) {
        args.precision = values["--precision"];
    }
    if (values.count("--warmup")) {
        args.warmup = std::stoi(values["--warmup"]);
    }
    if (values.count("--runs")) {
        args.runs = std::stoi(values["--runs"]);
    }
    return args;
}

int main(int argc, char** argv) {
    try {
        const Args args = parse_args(argc, argv);
        const Config config = load_config(args.config_path);
        const int warmup = args.warmup >= 0 ? args.warmup : config.warmup;
        const int runs = args.runs >= 0 ? args.runs : config.runs;

        fs::path image_dir = args.image_dir.empty() ? config.val_images : args.image_dir;
        if (!image_dir.is_absolute()) {
            image_dir = args.config_path.parent_path().parent_path() / image_dir;
        }

        fs::path model_path = args.model_path;
        if (!model_path.is_absolute()) {
            model_path = args.config_path.parent_path().parent_path() / model_path;
        }

        auto backend = build_backend(args.backend, model_path);
        const std::vector<fs::path> images = collect_images(image_dir);

        const cv::Mat sample = cv::imread(images.front().string());
        if (sample.empty()) {
            throw std::runtime_error("Failed to read sample image: " + images.front().string());
        }

        const PreprocessResult warm_input = preprocess_image(sample, config);
        backend->warmup(warm_input.tensor, warm_input.shape, warmup);

        TimingMetrics timing;
        for (int i = 0; i < runs; ++i) {
            const fs::path& image_path = images[static_cast<size_t>(i) % images.size()];
            const cv::Mat image = cv::imread(image_path.string());
            if (image.empty()) {
                throw std::runtime_error("Failed to read image: " + image_path.string());
            }

            const PreprocessResult pre = preprocess_image(image, config);
            double inference_ms = 0.0;
            auto [output, output_shape] = backend->infer(pre.tensor, pre.shape, inference_ms);
            const PostprocessResult post = postprocess_predictions(output, output_shape, config.conf, config.max_det);

            timing.preprocess_ms.push_back(pre.elapsed_ms);
            timing.inference_ms.push_back(inference_ms);
            timing.postprocess_ms.push_back(post.elapsed_ms);
            timing.total_ms.push_back(pre.elapsed_ms + inference_ms + post.elapsed_ms);
        }

        fs::path report_path = args.report_path.empty() ? config.report_dir / (args.backend + "_" + args.precision + "_cpp.json") : args.report_path;
        if (!report_path.is_absolute()) {
            report_path = args.config_path.parent_path().parent_path() / report_path;
        }
        fs::create_directories(report_path.parent_path());

        const std::string report = build_report(args.backend, model_path, args.precision, args.quantized, timing);
        std::ofstream output(report_path, std::ios::binary);
        output << report;
        output.close();

        std::cout << "Saved report to " << report_path.string() << '\n';
        std::cout << report;
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << ex.what() << '\n';
        return 1;
    }
}
