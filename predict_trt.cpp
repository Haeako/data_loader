#include <torch/torch.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <spng.h>
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
/* ================================= var define ============================*/
static BufferPool g_buffer_pool(BUFFER_POOL_SIZE, 32 * 1024 * 1024);
int fd;
struct stat st;
void *map = nullptr;
spng_ctx *ctx = nullptr;
size_t file_size;
spng_format fmt = SPNG_FMT_RGB8;
size_t out_size;
int flags = 0 ;
int height, width;
// SPNG decoder with optimized settings
std::vector<uint8_t>&  out_buf = g_buffer_pool.get_buffer();

cv::Mat decode_png_spng(const std::string &filename) {
    // auto start = std::chrono::high_resolution_clock::now();
    fd= open(filename.c_str(), O_RDONLY );
    if (fd < 0) { perror("open"); return {}; }
    
    if (fstat(fd, &st) < 0) {
        // std::cout << "fstat failed: " << strerror(errno) << std::endl; 
        perror("fstat"); 
        close(fd);
         return {}; 
    }
    file_size = st.st_size;
    
    map = mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    // if (map == MAP_FAILED) {
    //     std::cerr << "mmap failed: " << strerror(errno) << std::endl;
    //      perror("mmap"); return {}; 
    //     }

    ctx = spng_ctx_new(SPNG_CTX_IGNORE_ADLER32);
    // if (!ctx) {
    //     std::cerr << "Failed to create SPNG context" << std::endl;
    //     munmap(map, file_size);
    //     return {};
    // }

    // Set decoding options for speed
    // No spng_set_option for SPNG_DECODE_USE_TRNS; handled via decode flags below
    spng_set_png_buffer(ctx, reinterpret_cast<uint8_t*>(map), file_size);
    
    spng_ihdr ihdr;
    spng_get_ihdr(ctx, &ihdr);
    // if ()) {
    //     std::cerr << "spng_get_ihdr failed" << std::endl;
    //     spng_ctx_free(ctx);
    //     munmap(map, file_size);
    //     return {};
    // }

    // Get a buffer from the pool
    out_buf = g_buffer_pool.get_buffer();
    
    spng_decoded_image_size(ctx, fmt, &out_size);
    
    if (out_buf.size() < out_size) {
        out_buf.resize(out_size);
    }
    
    // Use optimized decoding flags
    auto start = std::chrono::high_resolution_clock::now();
    int err = spng_decode_image(ctx, out_buf.data(), out_size, fmt, flags);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    // std::cout << "Decode time: " << elapsed.count() << " seconds" << std::endl;
    // if (err != 0) {
    //     const char* msg = spng_strerror(err);
    //     std::cerr << "spng_decode_image failed: " << msg << std::endl;
    //     spng_ctx_free(ctx);
    //     munmap(map, file_size);
    //     g_buffer_pool.release_buffer(out_buf);
    //     return {};
    // }

    spng_ctx_free(ctx);
    munmap(map, file_size);

    width = ihdr.width;
    height = ihdr.height;
    
    // Create OpenCV matrix directly in RGB format (avoid BGR conversion)
    cv::Mat img_rgb(height, width, CV_8UC3, out_buf.data());
    // cv::Mat result = img_rgb.clone();  // Clone to copy data
    
    // Release buffer back to pool
    g_buffer_pool.release_buffer(out_buf);
    // auto end = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double> elapsed = end - start;
    // std::cout << "Decode time: " << elapsed.count() << " seconds" << std::endl;
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
    GPUBufferManager(int initial_width = 3220, int initial_height = 3080) {
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

    // // Collect all image paths first
    // std::vector<std::string> image_paths;
    // for () {
    //     if (!entry.is_regular_file()) continue;
    //     if (entry.path().extension() == ".png") {
    //         image_paths.push_back(entry.path().string());
    //     }
    // }

    int count = 0;
    cv::cuda::GpuMat gpu_input, gpu_rgb, gpu_resized;
    auto t_start = std::chrono::high_resolution_clock::now();
    void* ptr;
    for (const auto &entry : std::filesystem::directory_iterator(folder)) {
        cv::Mat img = decode_png_spng(entry.path().string());
        // if (img.empty()) continue;

        // auto start = std::chrono::high_resolution_clock::now();
        // Ensure GPU buffers are large enough
        gpu_buffers.ensure_size(img.cols, img.rows);
        
        // Process in GPU with zero-copy when possible
        gpu_input = gpu_buffers.get_input_buffer();
        gpu_rgb = gpu_buffers.get_rgb_buffer();
        gpu_resized = gpu_buffers.get_resized_buffer();
        
        gpu_input.upload(img, cv_stream);
        
        // Skip BGR2RGB conversion if the image is already in RGB format
        cv::cuda::resize(gpu_input, gpu_resized, 
                      cv::Size(OUTPUT_WIDTH, OUTPUT_HEIGHT),
                      0, 0, cv::INTER_NEAREST, cv_stream);
        cv_stream.waitForCompletion();

        // Use at::cuda::getCurrentCUDAStream() to share CUDA stream with PyTorch
        at::cuda::CUDAStream torch_stream = at::cuda::getCurrentCUDAStream();
        cudaStreamSynchronize(torch_stream);
        // auto end = std::chrono::high_resolution_clock::now();
        // std::chrono::duration<double> elapsed = end - start;
        // std::cout << "Processing time: " << elapsed.count() << " seconds" << std::endl;
        ptr = static_cast<void*>(gpu_resized.ptr<uchar>());
        auto tensor = torch::from_blob(ptr,
                                    {1, 3, OUTPUT_HEIGHT, OUTPUT_WIDTH},
                                    torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA))
                    .to(torch::kFloat32)
                    .div_(255.0);

        // Force synchronization only when necessary
        // Put any model inference code here
        
        ++count;
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start).count() / 1e6;
    double fps = count / elapsed;
    std::printf("Processed images in %.2f s -> FPS = %.2f\n", elapsed, fps);

    // Proper cleanup
    cv_stream.waitForCompletion();
    cudaStreamDestroy(stream);
    
    // Reset device to clean state
    cudaDeviceReset();
    
    return 0;
}