#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <string.h>
#include <vector>

// opencv v3.4.3
#include "opencv2/cudawarping.hpp"
#include "opencv2/cudaarithm.hpp"
#include "opencv2/cudaimgproc.hpp"
#include <opencv2/highgui.hpp>

// json
#include "../json.hpp"

// include a few (legacy c) yolo headers
#include "/opt/yolo2_light/src/additionally.h"
#include "/opt/yolo2_light/src/gpu.h"
// extern "C" {
  // #include "/opt/yolo2_light/src/box.h"
// }

struct YoloParams
{
    std::string datacfg;
    std::string cfgfile;
    std::string weightfile;
    float nms;
    float thresh;
    float hier_thresh;
};

struct detection_with_class {
    detection det;
    // The most probable class id: the best class index in this->prob.
    // Is filled temporary when processing results, otherwise not initialized
    int best_class;
};

//////////////////////////////////////////////////////////////////////////

class YoloV3
{
public:
  YoloV3();
  ~YoloV3();

  bool Setup(YoloParams & p);

  // convert the cvMat into our input format and pass along to private detect
  bool Detect(const cv::Mat& inputMat, nlohmann::json& outJson);
  bool Detect(const cv::cuda::GpuMat& inputMat, nlohmann::json& outJson);

private:
  network net;
  YoloParams yoloParams;
  char **classNames;
  int classNameCount;

  // allocate memory for a bunch of GpuMat's
  cv::cuda::GpuMat _inputMatResized;
  cv::cuda::GpuMat _inputMatBorder;
  cv::cuda::GpuMat _inputMatRgb;
  cv::cuda::GpuMat _floatMat;
  cv::cuda::GpuMat _emptyMat;
  float* _gpuEmptyInput;

  // where all the magic happens!
  void __Detect(
    float* X,
    int imageWidth,
    int imageHeight,
    nlohmann::json& outJson,
    bool isGpuMat=false);

  void __ProcessDetections(
    detection* dets,
    int num,
    nlohmann::json& outJson);

  detection_with_class* __GetActualDetections(
    detection *dets,
    int dets_num,
    float thresh,
    int* selected_detections_num
  );

  int static __CompareByLefts(const void *a_ptr, const void *b_ptr);

  cv::Size __GetNewSizeAspectAware(
    int frame_width, int frame_height,
    int desired_width, int desired_height);
};

