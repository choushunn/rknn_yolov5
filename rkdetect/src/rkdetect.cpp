#include "rkdetect.h"

// DetectorAPIImpl����DetectorAPI��ʵ����,�������Detector����
class DetectorAPIImpl {
private:
    // ��̬��Ա����detector,���ڴ洢Detector�����ָ��
    static Detector* detector;

public:
    // ��̬��Ա����,���ڵ���Detector�����run��������Ŀ����
    // ����imgΪ����ͼ�񻺳���,od_resultsΪ������б�
    static int run(image_buffer_t* img, object_detect_result_list* od_results) {
        return detector->run(img, od_results);
    }

    // ��̬��Ա����,�����ͷ�Detector�������Դ
    static void release() {
        delete detector;
        detector = nullptr;
    }
};

// ��ʼ��DetectorAPIImpl��ľ�̬��Ա����detector,ָ��һ���´�����Detector����
Detector* DetectorAPIImpl::detector = new Detector("./model/best.rknn");

// ����DetectorAPIImpl��ľ�̬��Ա����run
int DetectorAPI::run(image_buffer_t* img, object_detect_result_list* od_results) {
    return DetectorAPIImpl::run(img, od_results);
}

// ����DetectorAPIImpl��ľ�̬��Ա����release
void DetectorAPI::release() {
    DetectorAPIImpl::release();
}