#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>

int main() {
    int count = cv::cuda::getCudaEnabledDeviceCount();
    if (count == 0) {
        std::cerr << "No CUDA devices found\n";

    } else {
        std::cout << "CUDA devices available: " << count << "\n";
    }

    std::cout << "OpenCV version: " << CV_VERSION << std::endl;
    
    // Tạo một ma trận đơn giản
    cv::Mat mat = cv::Mat::zeros(3, 3, CV_8UC3);
    
    // In ra kích thước ma trận
    std::cout << "Matrix size: " << mat.size() << std::endl;
    
    // Liệt kê các modules đã được cài đặt
    std::cout << "OpenCV modules:" << std::endl;
    std::cout << cv::getBuildInformation() << std::endl;
    
    return 0;
}