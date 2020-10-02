#include "wrapper.hpp"
#include <iostream>

float overlap(float x1, float w1, float x2, float w2)
{
    float l1 = x1 - w1/2;
    float l2 = x2 - w2/2;
    float left = l1 > l2 ? l1 : l2;
    float r1 = x1 + w1/2;
    float r2 = x2 + w2/2;
    float right = r1 < r2 ? r1 : r2;
    return right - left;
}

float box_intersection(box a, box b)
{
    float w = overlap(a.x, a.w, b.x, b.w);
    float h = overlap(a.y, a.h, b.y, b.h);
    if(w < 0 || h < 0) return 0;
    float area = w*h;
    return area;
}

float box_union(box a, box b)
{
    float i = box_intersection(a, b);
    float u = a.w*a.h + b.w*b.h - i;
    return u;
}

float box_iou(box a, box b)
{
    return box_intersection(a, b)/box_union(a, b);
}
int nms_comparator(const void *pa, const void *pb)
{
    detection a = *(detection *)pa;
    detection b = *(detection *)pb;
    float diff = 0;
    if (b.sort_class >= 0) {
        diff = a.prob[b.sort_class] - b.prob[b.sort_class];
    }
    else {
        diff = a.objectness - b.objectness;
    }
    if (diff < 0) return 1;
    else if (diff > 0) return -1;
    return 0;
}
void do_nms_sort(detection *dets, int total, int classes, float thresh)
{
    int i, j, k;
    k = total - 1;
    for (i = 0; i <= k; ++i) {
        if (dets[i].objectness == 0) {
            detection swap = dets[i];
            dets[i] = dets[k];
            dets[k] = swap;
            --k;
            --i;
        }
    }
    total = k + 1;

    for (k = 0; k < classes; ++k) {
        for (i = 0; i < total; ++i) {
            dets[i].sort_class = k;
        }
        qsort(dets, total, sizeof(detection), nms_comparator);
        for (i = 0; i < total; ++i) {
            //printf("  k = %d, \t i = %d \n", k, i);
            if (dets[i].prob[k] == 0) continue;
            box a = dets[i].bbox;
            for (j = i + 1; j < total; ++j) {
                box b = dets[j].bbox;
                if (box_iou(a, b) > thresh) {
                    dets[j].prob[k] = 0;
                }
            }
        }
    }
}

// constructor
YoloV3::YoloV3()
{
  _emptyMat = cv::cuda::GpuMat(100, 100, CV_32FC3);
  _gpuEmptyInput = _emptyMat.ptr<float>();
}

// destructure
YoloV3::~YoloV3()
{
  // free VRAM & Ram
  // if(net)
  //   free_network(net);
  // delete net;

  // free up the classNames
  int i;
  for (i = 0; i < classNameCount; ++i) free(classNames[i]);
  free(classNames);
}

// setup the network
bool YoloV3::Setup(YoloParams& p)
{
  // save the params
  yoloParams = p;

  // create the network
  net = parse_network_cfg(
    (char *)yoloParams.cfgfile.c_str(),
    1,
    false
  );

  // load the (pretrained) weights
  load_weights_upto_cpu(
    &net,
    (char *)yoloParams.weightfile.c_str(),
    net.n
  );

  // assuming this is NOT needed?
  // set_batch_network(&net, 1);

  // what in the world is this for?
  srand(2222222);

  // fuse a few layers
  yolov2_fuse_conv_batchnorm(net);

  // not sure what this is for yet...
  calculate_binary_weights(net);

  // load object names
  classNames = (char**)calloc(10000, sizeof(char *));
  int obj_count = 0;
  FILE* fp;
  char buffer[255];
  fp = fopen((char *)p.datacfg.c_str(), "r");
  while (fgets(buffer, 255, (FILE*)fp)) {
      classNames[obj_count] = (char*)calloc(strlen(buffer)+1, sizeof(char));
      strcpy(classNames[obj_count], buffer);
      classNames[obj_count][strlen(buffer) - 1] = '\0'; //remove newline
      ++obj_count;
  }
  fclose(fp);
  classNameCount = obj_count;

  return true;
}

// Detect API to get objects detected
// \warning Setup must have been done on the object before
bool YoloV3::Detect(const cv::Mat& inputMat, nlohmann::json& outJson)
{
  // from yolo image.c
  // image im = load_image(input, 0, 0, 3);
  // image sized = resize_image(im, net.w, net.h);
  //
  // __Detect(sized.data, im.width, im.height, outJson);

  // get our input image size!
  cv::Size imageSize = inputMat.size();

  // this is just for fun...
  // cv::imwrite("/weights/test_frame.jpg", inputMat);

  // resize to the correct width
  // cv::Mat inputMatResized;
  // cv::resize(inputMat, inputMatResized, cv::Size(416, 235), 0, 0, cv::INTER_AREA);
  //
  // // how much of a border do we need?
  // int borderTop = (416-inputMatResized.rows) / 2;
  // int borderBottom = (416-inputMatResized.rows+1) / 2;
  //
  // // letterbox (add black to the top and bottom)
  // cv::Mat inputMatBorder;
  // cv::copyMakeBorder(
  //   inputMatResized,
  //   inputMatBorder,
  //   borderTop,              // top
  //   borderBottom,           // bottom
  //   0,                      // left
  //   0,                      // right
  //   cv::BORDER_CONSTANT,
  //   cv::Scalar(0, 0, 0)
  // );

  // IplImage ipl_img = inputMatBorder.operator IplImage();
  // IplImage ipl_img;
  // iplimage_from_cvmat(inputMat, ipl_img);
  // IplImage ipl_img = IplImage(inputMat);
  IplImage imgTmp = inputMat;
  IplImage *ipl_img = cvCloneImage(&imgTmp);

  // IplImage *ipl_img = inputMatBorder;
  image out = ipl_to_image(ipl_img);
  cvReleaseImage(&ipl_img);
  rgbgr_image(out);
  image floatMat = resize_image(out, net.w, net.h);

  // convert to rgb
  // cv::Mat inputRgb;
  // cv::cvtColor(inputMat, inputRgb, CV_BGR2RGB);
  //
  // // convert the bytes to float
  // cv::Mat floatMat;
  // inputRgb.convertTo(floatMat, CV_32FC3, 1/255.0);

  // Get the image to suit darknet
  // cv::Mat floatMatChannels[3];
  // cv::split(floatMat, floatMatChannels);
  // vconcat(floatMatChannels[0], floatMatChannels[1], floatMat);
  // vconcat(floatMat, floatMatChannels[2], floatMat);
  __Detect(
    (float*)floatMat.data,
    imageSize.width,
    imageSize.height,
    outJson
  );

  free_image(out);
  free_image(floatMat);

  return true;
}

bool YoloV3::Detect(const cv::cuda::GpuMat& inputMat, nlohmann::json& outJson)
{
  // get our input image size!
  cv::Size imageSize = inputMat.size();

  // get our new image size
  auto newSize = __GetNewSizeAspectAware(
    imageSize.width, imageSize.height,
    net.w, -1
  );

  // resize to the correct width
  cv::cuda::resize(inputMat, _inputMatResized, newSize, 0, 0, cv::INTER_AREA);

  // how much of a border do we need?
  int borderTop = (net.w-_inputMatResized.rows) / 2;
  int borderBottom = (net.w-_inputMatResized.rows+1) / 2;

  // letterbox (add black to the top and bottom)
  cv::cuda::copyMakeBorder(
    _inputMatResized,
    _inputMatBorder,
    borderTop,              // top
    borderBottom,           // bottom
    0,                      // left
    0,                      // right
    cv::BORDER_CONSTANT,
    cv::Scalar(0, 0, 0)
  );

  // convert to rgb
  cv::cuda::cvtColor(
    _inputMatBorder,
    _inputMatRgb,
    inputMat.channels() == 4 ? CV_BGRA2RGB : CV_BGR2RGB
  );

  // convert the bytes to float
  _inputMatRgb.convertTo(_floatMat, CV_32FC3, 1.0/255, 0);

  // change HWC -> CHW
  size_t width = _floatMat.cols * _floatMat.rows;
  auto ptr = net.input_state_gpu;
  std::vector<cv::cuda::GpuMat> input_channels {
      cv::cuda::GpuMat(_floatMat.rows, _floatMat.cols, CV_32F, &ptr[0]),
      cv::cuda::GpuMat(_floatMat.rows, _floatMat.cols, CV_32F, &ptr[width]),
      cv::cuda::GpuMat(_floatMat.rows, _floatMat.cols, CV_32F, &ptr[width * 2])
  };
  cv::cuda::split(_floatMat, input_channels);

  // do the magic!
  __Detect(
    _gpuEmptyInput,
    imageSize.width,
    imageSize.height,
    outJson,
    true
  );

  return true;
}

//////////////////////////////////////////////////////////////////
/// Private APIs
//////////////////////////////////////////////////////////////////
void YoloV3::__Detect(
  float* X,
  int imageWidth,
  int imageHeight,
  nlohmann::json& outJson,
  bool isGpuMat
  )
{
  // input layer
  layer l = net.layers[net.n - 1];

  // allocate memory for results
  int j;
  box *boxes = (box*)calloc(l.w*l.h*l.n, sizeof(box));
  float **probs = (float**)calloc(l.w*l.h*l.n, sizeof(float *));
  for (j = 0; j < l.w*l.h*l.n; ++j) probs[j] = (float*)calloc(l.classes, sizeof(float *));

  // only going to implement GPU for now
  if (isGpuMat) {
    network_predict_gpu_cudnn_custom(net, X);
  }
  else {
    network_predict_gpu_cudnn(net, X);
  }

  float hier_thresh = 0.5;
  int ext_output = 1, letterbox = 0, nboxes = 0;

  // get the detections!
  detection *dets = get_network_boxes(
    &net,
    imageWidth,
    imageHeight,
    yoloParams.thresh,
    yoloParams.hier_thresh,
    0,
    1,
    &nboxes,
    isGpuMat ? 1 : letterbox
  );

  // non maximum suppression
  if (yoloParams.nms) do_nms_sort(dets, nboxes, l.classes, yoloParams.nms);

  // process the detections!
  __ProcessDetections(dets, nboxes, outJson);

  // free up memory!
  free(boxes);
  free_ptrs((void **)probs, l.w * l.h * l.n);
}
void YoloV3::__ProcessDetections(
  detection* dets,
  int num,
  nlohmann::json& outJson
  )
{
  int selected_detections_num;
  detection_with_class* selected_detections = __GetActualDetections(
    dets,
    num,
    yoloParams.thresh,
    &selected_detections_num
  );

  // vector of detections
  std::vector<nlohmann::json> motionDetections;

  // sort the detections by probability
  qsort(
    selected_detections,
    selected_detections_num,
    sizeof(*selected_detections),
    __CompareByLefts
  );

  int i;
  for (i = 0; i < selected_detections_num; ++i) {
      const int best_class = selected_detections[i].best_class;
      // printf("%s: %.0f%%", classNames[best_class], selected_detections[i].det.prob[best_class] * 100);

      // create our detection
      nlohmann::json jDetection;
      jDetection["class"] = classNames[best_class];
      jDetection["prob"] = selected_detections[i].det.prob[best_class] * 100;

      // add in our bbox
      jDetection["x"] = (selected_detections[i].det.bbox.x - selected_detections[i].det.bbox.w / 2);
      jDetection["y"] = (selected_detections[i].det.bbox.y - selected_detections[i].det.bbox.h / 2);
      jDetection["w"] = selected_detections[i].det.bbox.w;
      jDetection["h"] = selected_detections[i].det.bbox.h;

      // alternate classes?
      // do we wan't to return these?
      // int j;
      // for (j = 0; j < classes; ++j) {
      //     if (selected_detections[i].det.prob[j] > thresh && j != best_class) {
      //         printf("%s: %.0f%%\n", names[j], selected_detections[i].det.prob[j] * 100);
      //     }
      // }

      // add to our vector
      if (jDetection["class"] != "bed") {
        motionDetections.push_back(jDetection);
      }
  }

  // add our detections to the resulting json
  outJson["detections"] = motionDetections;

  // how many?
  outJson["count"] = motionDetections.size();

  // free up the memory
  free(selected_detections);
}


// Creates array of detections with prob > thresh and fills best_class for them
detection_with_class* YoloV3::__GetActualDetections(
  detection *dets,
  int dets_num,
  float thresh,
  int* selected_detections_num
  )
{
    int selected_num = 0;
    detection_with_class* result_arr = (detection_with_class*)calloc(dets_num, sizeof(detection_with_class));
    int i;
    for (i = 0; i < dets_num; ++i) {
        int best_class = -1;
        float best_class_prob = thresh;
        int j;
        for (j = 0; j < dets[i].classes; ++j) {
            if (dets[i].prob[j] > best_class_prob) {
                best_class = j;
                best_class_prob = dets[i].prob[j];
            }
        }
        if (best_class >= 0) {
            result_arr[selected_num].det = dets[i];
            result_arr[selected_num].best_class = best_class;
            ++selected_num;
        }
    }
    if (selected_detections_num)
        *selected_detections_num = selected_num;
    return result_arr;
}

// compare to sort detection** by bbox.x
int YoloV3::__CompareByLefts(const void *a_ptr, const void *b_ptr) {
    const detection_with_class* a = (detection_with_class*)a_ptr;
    const detection_with_class* b = (detection_with_class*)b_ptr;
    const float delta = (a->det.bbox.x - a->det.bbox.w / 2) - (b->det.bbox.x - b->det.bbox.w / 2);
    return delta < 0 ? -1 : delta > 0 ? 1 : 0;
}

cv::Size YoloV3::__GetNewSizeAspectAware(
  int frame_width, int frame_height,
  int desired_width, int desired_height)
{
  int out_width, out_height;

  // if we have values >0 for both height and width, just pass them along...
  if (desired_width > 0 && desired_height > 0) {
    out_width = desired_width;
    out_height = desired_height;
  }
  else if (desired_width == -1 && desired_height == -1) {
    // exception!
    throw std::invalid_argument( "Desired width or height must be provided" );
  }
  else {
    // which side are we scaling?
    if (desired_height == -1) {
      // determine aspect ratio
      float aspectRatio = (float)desired_width / (float)frame_width;
      out_width = desired_width;
      out_height = (int)(frame_height * aspectRatio);
    }
    else {
      // determine aspect ratio
      float aspectRatio = (float)desired_height / (float)frame_height;
      out_width = (int)(frame_width * aspectRatio);
      out_height = desired_height;
    }
  }

  // return our size!
  return cv::Size(out_width, out_height);
}

