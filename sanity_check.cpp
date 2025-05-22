#include <torch/torch.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <spng.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <chrono>
#include <iostream>
#include <vector>
#include <filesystem>

constexpr int OUTPUT_WIDTH = 640;
constexpr int OUTPUT_HEIGHT = 640;
constexpr int BUFFER_POOL_SIZE = 8;
constexpr size_t MAX_PNG_BYTES = 32 * 1024 * 1024;

class BufferPool {
private:
    std::vector<std::vector<uint8_t>> buffers;
public:
    BufferPool(size_t pool_size, size_t buffer_size) {
        buffers.resize(pool_size);
        for (auto &b : buffers) {
            b.resize(buffer_size);
        }
    }
    std::vector<uint8_t>& get_buffer(size_t index) {
        return buffers[index % buffers.size()];
    }
};

static BufferPool g_buffer_pool(BUFFER_POOL_SIZE, MAX_PNG_BYTES);

cv::Mat decode_png_spng(const std::string &filename,
                        spng_ctx *ctx,
                        std::vector<uint8_t> &out_buf) {
    int fd = open(filename.c_str(), O_RDONLY);
    if (fd < 0) {
        perror("open failed");
        return cv::Mat();
    }

    struct stat st;
    if (fstat(fd, &st) < 0) {
        perror("fstat failed");
        close(fd);
        return cv::Mat();
    }

    size_t file_size = st.st_size;
    if (file_size == 0 || file_size > MAX_PNG_BYTES) {
        std::cerr << "Invalid file size: " << file_size << std::endl;
        close(fd);
        return cv::Mat();
    }

    void *map = mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (map == MAP_FAILED) {
        perror("mmap failed");
        return cv::Mat();
    }

    if (spng_set_png_buffer(ctx, reinterpret_cast<uint8_t*>(map), file_size) != 0) {
        std::cerr << "spng_set_png_buffer failed" << std::endl;
        munmap(map, file_size);
        return cv::Mat();
    }

    spng_ihdr ihdr;
    if (spng_get_ihdr(ctx, &ihdr) != 0) {
        std::cerr << "spng_get_ihdr failed" << std::endl;
        munmap(map, file_size);
        return cv::Mat();
    }

    size_t out_size;
    auto start = std::chrono::high_resolution_clock::now();
    if (spng_decoded_image_size(ctx, SPNG_FMT_RGB8, &out_size) != 0 || out_size == 0) {
        std::cerr << "spng_decoded_image_size failed" << std::endl;
        munmap(map, file_size);
        return cv::Mat();
    }

    if (out_buf.size() < out_size) {
        out_buf.resize(out_size);
    }

    if (spng_decode_image(ctx, out_buf.data(), out_size, SPNG_FMT_RGB8, 0) != 0) {
        std::cerr << "spng_decode_image failed" << std::endl;
        munmap(map, file_size);
        return cv::Mat();
    }

    munmap(map, file_size);
    return cv::Mat(ihdr.height, ihdr.width, CV_8UC3, out_buf.data()).clone();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Decoded " << filename << " in " << elapsed.count() << " seconds" << std::endl;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <image_folder>" << std::endl;
        return -1;
    }
    std::string folder = argv[1];

    cudaSetDevice(0);
    cv::cuda::setDevice(0);

    cudaStream_t stream;
    int priority_low, priority_high;
    cudaDeviceGetStreamPriorityRange(&priority_low, &priority_high);
    cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, priority_high);
    cv::cuda::Stream cv_stream = cv::cuda::StreamAccessor::wrapStream(stream);

    cv::cuda::GpuMat gpu_input, gpu_resized;
    gpu_resized.create(OUTPUT_HEIGHT, OUTPUT_WIDTH, CV_8UC3);

    {
        cv::cuda::GpuMat dummy(10,10,CV_8UC3);
        cv::cuda::cvtColor(dummy, dummy, cv::COLOR_BGR2RGB, 0, cv_stream);
        cv::cuda::resize(dummy, dummy, cv::Size(5,5), 0,0, cv::INTER_NEAREST, cv_stream);
        cv_stream.waitForCompletion();
    }

    spng_ctx *ctx = spng_ctx_new(0);
    if (!ctx) {
        std::cerr << "Failed to create SPNG context" << std::endl;
        return -1;
    }

    at::Tensor input_tensor = at::empty({1, 3, OUTPUT_HEIGHT, OUTPUT_WIDTH},
                                        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

    size_t count = 0;
    auto t_start = std::chrono::high_resolution_clock::now();

    for (const auto &entry : std::filesystem::directory_iterator(folder)) {
        if (entry.path().extension() != ".png") continue;

        auto &out_buf = g_buffer_pool.get_buffer(count);
        cv::Mat img = decode_png_spng(entry.path().string(), ctx, out_buf);

        if (img.empty()) {
            std::cerr << "Skipping invalid image: " << entry.path() << std::endl;
            continue;
        }
        auto t_decode = std::chrono::high_resolution_clock::now();
        gpu_input.upload(img, cv_stream);

        cv::cuda::resize(gpu_input, gpu_resized,
                         cv::Size(OUTPUT_WIDTH, OUTPUT_HEIGHT),
                         0, 0, cv::INTER_NEAREST, cv_stream);

        cv_stream.waitForCompletion();
        std::memcpy(input_tensor.data_ptr(), gpu_resized.ptr(), OUTPUT_HEIGHT * OUTPUT_WIDTH * 3);

        input_tensor.div_(255.0);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - t_decode;
        std::cout << "Processed " << entry.path() << " in " << elapsed.count() << " seconds" << std::endl;
        ++count;
    }

    cv_stream.waitForCompletion();
    cudaStreamSynchronize(stream);

    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t_end - t_start).count();
    double fps = count / elapsed;
    std::printf("Processed %zu images in %.2f s -> FPS = %.2f\n", count, elapsed, fps);

    spng_ctx_free(ctx);
    cudaStreamDestroy(stream);
    cudaDeviceReset();

    return 0;
}