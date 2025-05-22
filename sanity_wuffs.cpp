#include <torch/torch.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>

#define WUFFS_IMPLEMENTATION
#include "wuffs-v0.3.c"
#include "wuffs-unsupported-snapshot.c"
#include <sys/stat.h>
#include <sys/mman.h>
#include <complex.h>
#include <fcntl.h>
#include <unistd.h>
#include <chrono>
#include <iostream>
#include <vector>
#include <filesystem>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <atomic>

constexpr int OUTPUT_WIDTH = 640;
constexpr int OUTPUT_HEIGHT = 640;
constexpr int BUFFER_POOL_SIZE = 8;  // Pre-allocate several buffers

// Thread-safe buffer pool
class BufferPool {
private:
    std::vector<std::vector<uint8_t>> buffers;
    std::queue<int> available_indices;
    std::mutex mtx;
    std::condition_variable cv;

public:
    BufferPool(size_t pool_size, size_t buffer_size) {
        buffers.resize(pool_size);
        for (size_t i = 0; i < pool_size; ++i) {
            buffers[i].reserve(buffer_size);
            available_indices.push(i);
        }
    }

    std::vector<uint8_t>& get_buffer() {
        std::unique_lock<std::mutex> lock(mtx);
        cv.wait(lock, [this] { return !available_indices.empty(); });
        
        int idx = available_indices.front();
        available_indices.pop();
        return buffers[idx];
    }

    void release_buffer(std::vector<uint8_t>& buffer) {
        auto it = std::find_if(buffers.begin(), buffers.end(), 
                           [&buffer](const std::vector<uint8_t>& b) { 
                               return &b == &buffer; 
                           });
        
        if (it != buffers.end()) {
            int idx = std::distance(buffers.begin(), it);
            std::lock_guard<std::mutex> lock(mtx);
            available_indices.push(idx);
            cv.notify_one();
        }
    }
};

// Global buffer pool for image decoding (32MB per buffer should be enough for most images)
static BufferPool g_buffer_pool(BUFFER_POOL_SIZE, 32 * 1024 * 1024);

// SPNG decoder with optimized settings
cv::Mat decode_png_wuffs(const std::string &filename) {
    int fd = open(filename.c_str(), O_RDONLY);
    if (fd < 0) { perror("open"); return {}; }

    struct stat st;
    if (fstat(fd, &st) < 0) { perror("fstat"); close(fd); return {}; }
    size_t file_size = st.st_size;

    void* file_data = mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (file_data == MAP_FAILED) { perror("mmap"); return {}; }

    // Khởi tạo bộ giải mã PNG
    wuffs_png__decoder dec;
    wuffs_base__status status = wuffs_png__decoder__initialize(
        &dec, sizeof dec, WUFFS_VERSION, WUFFS_INITIALIZE__LEAVE_INTERNAL_BUFFERS_UNINITIALIZED);
    if (!wuffs_base__status__is_ok(&status)) {
        std::cerr << "Wuffs init error: " << status.repr << std::endl;
        munmap(file_data, file_size);
        return {};
    }

    wuffs_base__io_buffer src = wuffs_base__ptr_u8__reader(
        (uint8_t*)file_data, file_size, true);
    
    // Tạm buffer đầu ra 32MB
    auto& out_buf = g_buffer_pool.get_buffer();
    if (out_buf.size() < 32 * 1024 * 1024) out_buf.resize(32 * 1024 * 1024);

    wuffs_base__frame_config frame_cfg;
    status = wuffs_png__decoder__decode_frame_config(
        &dec,
        &frame_cfg,
        &src
    );
    if (!wuffs_base__status__is_ok(&status)) {
        std::cerr << "decode_frame_config failed: " << status.repr << std::endl;
    }
    wuffs_base__pixel_config pixcfg = wuffs_base__frame_config__pixcfg(&frame_cfg);
    int width  = (int)wuffs_base__pixel_config__width(&pixcfg);
    int height = (int)wuffs_base__pixel_config__height(&pixcfg);

    if (!wuffs_base__status__is_ok(&status)) {
        std::cerr << "Wuffs decode_frame_config failed: " << status.repr << std::endl;
        g_buffer_pool.release_buffer(out_buf);
        munmap(file_data, file_size);
        return {};
    }

    int width = (int)wuffs_base__pixel_config__width(&pixcfg);
    int height = (int)wuffs_base__pixel_config__height(&pixcfg);
    int stride = width * 4; // RGBA

    if ((size_t)(stride * height) > out_buf.size()) {
        out_buf.resize(stride * height);
    }

    wuffs_base__pixel_buffer pixbuf;
    status = wuffs_base__pixel_buffer__set_from_slice(
        &pixbuf, &pixcfg,
        wuffs_base__make_slice_u8(out_buf.data(), stride * height));
    if (!wuffs_base__status__is_ok(&status)) {
        std::cerr << "set_from_slice failed: " << status.repr << std::endl;
        g_buffer_pool.release_buffer(out_buf);
        munmap(file_data, file_size);
        return {};
    }
    wuffs_base__slice_u8 empty_workbuf = wuffs_base__make_slice_u8(nullptr, 0);

    status = wuffs_png__decoder__decode_frame(
        &dec,
        &pixbuf,
        &src,
        WUFFS_BASE__PIXEL_BLEND__SRC_OVER,
        &empty_workbuf     // ← not nullptr!
    );

    if (!wuffs_base__status__is_ok(&status)) {
        std::cerr << "decode_frame failed: " << status.repr << std::endl;
        g_buffer_pool.release_buffer(out_buf);
        munmap(file_data, file_size);
        return {};
    }

    // Convert RGBA to RGB
    cv::Mat img_rgba(height, width, CV_8UC4, out_buf.data());
    cv::Mat img_rgb;
    cv::cvtColor(img_rgba, img_rgb, cv::COLOR_RGBA2RGB);

    g_buffer_pool.release_buffer(out_buf);
    munmap(file_data, file_size);

    return img_rgb;
}


// Pre-allocate GPU buffers to avoid memory allocation during processing
class GPUBufferManager {
private:
    cv::cuda::GpuMat input_buffer;
    cv::cuda::GpuMat rgb_buffer;
    cv::cuda::GpuMat resized_buffer;
    int max_width = 0;
    int max_height = 0;

public:
    GPUBufferManager(int initial_width = 1920, int initial_height = 1080) {
        // Pre-allocate with a reasonable size
        input_buffer.create(initial_height, initial_width, CV_8UC3);
        rgb_buffer.create(initial_height, initial_width, CV_8UC3);
        resized_buffer.create(OUTPUT_HEIGHT, OUTPUT_WIDTH, CV_8UC3);
        max_width = initial_width;
        max_height = initial_height;
    }

    void ensure_size(int width, int height) {
        if (width > max_width || height > max_height) {
            input_buffer.create(height, width, CV_8UC3);
            rgb_buffer.create(height, width, CV_8UC3);
            max_width = std::max(max_width, width);
            max_height = std::max(max_height, height);
        }
    }

    cv::cuda::GpuMat& get_input_buffer() { return input_buffer; }
    cv::cuda::GpuMat& get_rgb_buffer() { return rgb_buffer; }
    cv::cuda::GpuMat& get_resized_buffer() { return resized_buffer; }
};

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <image_folder>" << std::endl;
        return -1;
    }
    std::string folder = argv[1];

    // CUDA setup with optimized defaults
    cudaSetDevice(0);
    cv::cuda::setDevice(0);
    
    // Create CUDA stream with higher priority
    cudaStream_t stream;
    int priority_high, priority_low;
    cudaDeviceGetStreamPriorityRange(&priority_low, &priority_high);
    cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, priority_high);
    
    cv::cuda::Stream cv_stream = cv::cuda::StreamAccessor::wrapStream(stream);

    // Pre-allocate GPU memory
    GPUBufferManager gpu_buffers;

    // Set CUDA device flags for performance
    cudaSetDeviceFlags(cudaDeviceScheduleAuto);
    
    // Enable OpenCV CUDA caching allocator
    cv::cuda::DeviceInfo dev_info;
    size_t free_memory = dev_info.freeMemory();
    cv::cuda::setBufferPoolUsage(true);
    cv::cuda::setBufferPoolConfig(cv::cuda::getDevice(), free_memory * 0.5, 2);

    // Pre-compile kernels by running a dummy operation
    {
        cv::cuda::GpuMat dummy(10, 10, CV_8UC3);
        cv::cuda::cvtColor(dummy, dummy, cv::COLOR_BGR2RGB, 0, cv_stream);
        cv::cuda::resize(dummy, dummy, cv::Size(5, 5), 0, 0, cv::INTER_CUBIC, cv_stream);
        cv_stream.waitForCompletion();
    }

    // Collect all image paths first
    std::vector<std::string> image_paths;
    for (const auto &entry : std::filesystem::directory_iterator(folder)) {
        if (!entry.is_regular_file()) continue;
        if (entry.path().extension() == ".png") {
            image_paths.push_back(entry.path().string());
        }
    }

    int count = 0;
    auto t_start = std::chrono::high_resolution_clock::now();

    for (const auto& path : image_paths) {
        cv::Mat img = decode_png_wuffs(path);
        if (img.empty()) continue;

        // Ensure GPU buffers are large enough
        gpu_buffers.ensure_size(img.cols, img.rows);
        
        // Process in GPU with zero-copy when possible
        auto& gpu_input = gpu_buffers.get_input_buffer();
        auto& gpu_rgb = gpu_buffers.get_rgb_buffer();
        auto& gpu_resized = gpu_buffers.get_resized_buffer();
        
        gpu_input.upload(img, cv_stream);
        
        // Skip BGR2RGB conversion if the image is already in RGB format
        cv::cuda::resize(gpu_input, gpu_resized, 
                      cv::Size(OUTPUT_WIDTH, OUTPUT_HEIGHT),
                      0, 0, cv::INTER_CUBIC, cv_stream);
        cv_stream.waitForCompletion();

        // Use at::cuda::getCurrentCUDAStream() to share CUDA stream with PyTorch
        at::cuda::CUDAStream torch_stream = at::cuda::getCurrentCUDAStream();
        cudaStreamSynchronize(torch_stream);

        auto ptr = static_cast<void*>(gpu_resized.ptr<uchar>());
        auto tensor = torch::from_blob(ptr,
                                    {1, 3, OUTPUT_HEIGHT, OUTPUT_WIDTH},
                                    torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA))
                    .to(torch::kFloat16)
                    .div_(255.0);

        // Force synchronization only when necessary
        // Put any model inference code here
        
        ++count;
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start).count() / 1e6;
    double fps = count / elapsed;
    std::printf("Processed %d images in %.2f s -> FPS = %.2f\n", count, elapsed, fps);

    // Proper cleanup
    cv_stream.waitForCompletion();
    cudaStreamDestroy(stream);
    
    // Reset device to clean state
    cudaDeviceReset();
    
    return 0;
}