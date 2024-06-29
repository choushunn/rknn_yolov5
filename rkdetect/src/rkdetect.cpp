#include "rkdetect.h"

// DetectorAPIImpl类是DetectorAPI的实现类,负责管理Detector对象
class DetectorAPIImpl {
private:
    // 静态成员变量detector,用于存储Detector对象的指针
    static Detector* detector;

public:
    // 静态成员函数,用于调用Detector对象的run方法进行目标检测
    // 参数img为输入图像缓冲区,od_results为检测结果列表
    static int run(image_buffer_t* img, object_detect_result_list* od_results) {
        return detector->run(img, od_results);
    }

    // 静态成员函数,用于释放Detector对象的资源
    static void release() {
        delete detector;
        detector = nullptr;
    }
};

// 初始化DetectorAPIImpl类的静态成员变量detector,指向一个新创建的Detector对象
Detector* DetectorAPIImpl::detector = new Detector("./model/best.rknn");

// 调用DetectorAPIImpl类的静态成员函数run
int DetectorAPI::run(image_buffer_t* img, object_detect_result_list* od_results) {
    return DetectorAPIImpl::run(img, od_results);
}

// 调用DetectorAPIImpl类的静态成员函数release
void DetectorAPI::release() {
    DetectorAPIImpl::release();
}