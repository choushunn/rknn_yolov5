#ifndef DETECTORAPI_H
#define DETECTORAPI_H

#include "detector.h"

class DetectorAPI {
public:
    static int run(image_buffer_t* img, object_detect_result_list* od_results);
    static void release();
};

#endif // DETECTORAPI_H
