// dependencies
#include <fstream>
#include <sstream>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <numeric>
#include <iomanip>
#include <chrono>
#include <stdexcept>
#include <unistd.h>
#include <cstdlib>
#include <cstring>

// opencv v3.4.3
#include <opencv2/cudacodec.hpp>
#include "opencv2/cudawarping.hpp"
#include <opencv2/highgui.hpp>
#include "opencv2/cudabgsegm.hpp"
#include "opencv2/cudafilters.hpp"
#include "opencv2/cudaoptflow.hpp"
#include "opencv2/cudaarithm.hpp"

// json from nnlohmann v3.3.0
// https://github.com/nlohmann/json
#include "./json.hpp"

// dynamically link nvcuvid
#include "dynlink_nvcuvid.h"
#include "dynlink_cuviddec.h"

// yolo/darknet
#include "./yolo/wrapper.hpp"

// mqtt (async)
#include "mqtt/async_client.h"
// const string PERSIST_DIR { "data-persist" };

// not sure if i like namespaces, hides where the actual function is...
using namespace std;
using namespace std::chrono;
using namespace cv;
using namespace cv::cuda;
using json = nlohmann::json;

// this also feels wrong!
YoloV3* yolo = new YoloV3();

// sensor state
enum SensorState { off, on };

// have to init cuda
void init_cuda()
{
  // Init CUDA
  void* hHandleDriver = nullptr;
  CUresult cuda_res = cuInit(0, __CUDA_API_VERSION, hHandleDriver);
  if (cuda_res != CUDA_SUCCESS) {
      throw exception();
  }
  cuda_res = cuvidInit(0);
  if (cuda_res != CUDA_SUCCESS) {
      throw exception();
  }
  std::cout << "CUDA init: SUCCESS" << endl;

  // for fun!
  // cv::cuda::printShortCudaDeviceInfo(cv::cuda::getDevice());
}

void init_yolo()
{
  // fire up yolo!
  YoloParams yp;
  // yp.datacfg = (char *)"/opt/darknet/cfg/coco.data";
  // yp.cfgfile = (char *)"/opt/darknet/cfg/yolov3-tiny.cfg";
  // yp.weightfile = (char *)"/weights/yolov3-tiny.weights";
  // yp.datacfg = "/opt/darknet/cfg/coco.data";
  yp.datacfg = "/opt/darknet/data/coco.names";
  yp.cfgfile = "/opt/yolo2_light/bin/yolov3.cfg";
  yp.weightfile = "/weights/yolov3.weights";
  yp.nms = 0.4;
  yp.thresh = 0.24;
  yp.hier_thresh = 0.5;

  // Always setup before detect
  bool ret = yolo->Setup(yp);
  if (ret == false)
  {
    std::cerr << "Yolo Setup failed, exiting..." << ret << std::endl;
    if (yolo) delete yolo;
    yolo = 0;

    throw exception();
  }
}

Size size_maintain_aspect(
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
  return Size(out_width, out_height);
}

bool copyFile(const std::string SRC, const std::string DEST)
{
    std::ifstream src(SRC, std::ios::binary);
    std::ofstream dest(DEST, std::ios::binary);
    dest << src.rdbuf();
    return src && dest;
}

// encode image into a buffer
vector<uchar> encode_image(json& params, Mat& image)
{
  // buffer (byte array)
  vector<uchar> buf;

  // encoding parameters?
  vector<int> compression_params;
  compression_params.push_back(CV_IMWRITE_JPEG_QUALITY);
  compression_params.push_back(params["thumbnail"]["encodingQuality"]);

  // encode the image!
  imencode(
    ("." + params["thumbnail"]["encoding"]),
    image,
    buf,
    compression_params
  );

  // return the buffer
  return buf;
}

void imageWrite(const cv::Mat &image, const std::string filename)
{
    // Support for writing JPG
    vector<int> compression_params;
    compression_params.push_back( CV_IMWRITE_JPEG_QUALITY );
    compression_params.push_back( 100 );

    cv::imwrite(filename, image, compression_params);
}

// this is for debugging only
cv::Mat draw_detections(cv::Mat& h_frame, json& detections, bool isGpuMat, std::vector<cv::Point> detectionZone)
{
  cv::Mat cloneMat;
  h_frame.copyTo(cloneMat);

  // draw on the clone!
  Size matSize = cloneMat.size();
  long detectionCount = detections["detections"].size();

  for (int i = 0; i < detectionCount; ++i) {
    // std::cerr << "  - Class: " << detections["detections"][i]["class"] << " - Prob: " << detections["detections"][i]["prob"] << '\n';

    // {
    //   "class": "person",
    //   "h": 0.5510218143463135,
    //   "prob": 98.41949462890625,
    //   "w": 0.1309008151292801,
    //   "x": 0.016924947500228882,
    //   "y": 0.3701041340827942
    // }
    json det = detections["detections"][i];

    cv::Scalar boxColor = cv::Scalar(96, 242, 242, 200);
    if (detections["detections"][i]["class"] == "person") {
      boxColor = cv::Scalar(255, 0, 0, 155);
    }
    if (detections["detections"][i]["class"] == "dog") {
      boxColor = cv::Scalar(62, 128, 43, 155);
    }
    if (detections["detections"][i]["class"] == "bicycle") {
      boxColor = cv::Scalar(57, 68, 170, 155);
    }


    // time to draw!
    cv::rectangle(
      cloneMat,
      cv::Rect(
        (int)((float)det["x"] * matSize.width),
        (int)((float)det["y"] * matSize.height),
        (int)((float)det["w"] * matSize.width),
        (int)((float)det["h"] * matSize.height)
      ),
      boxColor
    );

    // aspect ratio
    // std::cerr << "Det:" << det << std::endl;
    // std::ostringstream stringAspectRatio;
    // stringAspectRatio << std::setprecision(4) << (((float)det["w"] * matSize.width) / ((float)det["h"] * matSize.height)) << '%';
    // // stringAspectRatio << (((float)det["w"] * matSize.width) / ((float)det["h"] * matSize.height)) << '%';
    // cv::putText(
    //   cloneMat,
    //   stringAspectRatio.str(),
    //   Point2f(
    //     (int)((float)det["x"] * matSize.width),
    //     (int)((float)det["y"] * matSize.height) - 3
    //   ),
    //   CV_FONT_HERSHEY_DUPLEX,
    //   0.4,
    //   Scalar(0, 0, 0, 200),
    //   1,
    //   CV_AA
    // );

    // probability
    // std::ostringstream stringProb;
    // stringProb << std::setprecision(4) << (double)detections["detections"][i]["prob"] << '%';
    //
    // Size textSize = cv::getTextSize(
    //   stringProb.str(),
    //   CV_FONT_HERSHEY_DUPLEX,
    //   0.2,
    //   1,
    //   0
    // );
    //
    // cv::putText(
    //   cloneMat,
    //   stringProb.str(),
    //   Point2f(
    //     (int)((float)det["x"] * matSize.width) + (int)((float)det["w"] * matSize.width) - textSize.width,
    //     (int)((float)det["y"] * matSize.height) - 3
    //   ),
    //   CV_FONT_HERSHEY_DUPLEX,
    //   0.2,
    //   Scalar(0, 0, 0, 200),
    //   1,
    //   CV_AA
    // );
  }

  return cloneMat;
}

vector<string> splitString (string s, string delimiter) {
    size_t pos_start = 0, pos_end, delim_len = delimiter.length();
    string token;
    vector<string> res;

    while ((pos_end = s.find (delimiter, pos_start)) != string::npos) {
        token = s.substr (pos_start, pos_end - pos_start);
        pos_start = pos_end + delim_len;
        res.push_back (token);
    }

    res.push_back (s.substr (pos_start));
    return res;
}
vector<cv::Point> init_detection_zone(int width, int height)
{
  // create our result contour
  vector<cv::Point> contour;

  // get the string of points
  std::string detectionZone = string(std::getenv("DETECTION_ZONE"));

  // split it by semi-colons followed by commas
  auto stringPoints = splitString(detectionZone, ";");
  for (size_t i = 0; i < stringPoints.size(); i++) {
    auto stringXY = splitString(stringPoints[i], ",");
    if (stringXY.size() != 2) {
      throw std::invalid_argument( "Size of XY != 2" );
    }

    // add to our contour
    contour.push_back(
      cv::Point(
        int(stof(stringXY[0]) * width),
        int(stof(stringXY[1]) * height)
      )
    );
  }

  // return our result
  return contour;
}

// the orchestrator
void process()
{
  // get our server address
  std::string mqttUrl = string(std::getenv("MQTT_URL"));

  cout << "MQTT Initializing for server '" << mqttUrl << "'..." << endl;

  // create the client
  mqtt::async_client client(mqttUrl, "");

  // get the username and password
  std::string mqttUsername = string(std::getenv("MQTT_USERNAME"));
  std::string mqttPassword = string(std::getenv("MQTT_PASSWORD"));
  mqtt::topic mqttTopic(
    client,
    string(std::getenv("MQTT_TOPIC")) + "/state",
    0,
    true
  );

  mqtt::connect_options connOpts(mqttUsername, mqttPassword);
	connOpts.set_keep_alive_interval(25 * seconds(5));
	connOpts.set_clean_session(true);
	connOpts.set_automatic_reconnect(true);

  // last will
  mqtt::message willmsg(
    string(std::getenv("MQTT_TOPIC")) + "/status",
    "offline",
    0,
    true
  );
  connOpts.set_will(willmsg);

  cout << "\nMQTT Connecting..." << endl;
  mqtt::token_ptr conntok = client.connect(connOpts);
  cout << "MQTT Waiting for the connection..." << endl;
  conntok->wait();
  cout << "MQTT Connection successful..." << endl;

  // publish an online status update
  client.publish(
    string(std::getenv("MQTT_TOPIC")) + "/status",
    "online",
    0,
    true
  );
  // client.publish(
  //   string(std::getenv("MQTT_TOPIC")) + "/set",
  //   "OFF",
  //   0,
  //   false
  // );

  // device (GPU) matrix's
  GpuMat d_frame, d_frame_thumb, d_frame_thumb_blur;
  GpuMat d_frame_last;
  GpuMat d_fgmask, d_bgframe;

  // init background subtraction
  auto pBG = cuda::createBackgroundSubtractorMOG();
  // auto d_calc = cuda::SparsePyrLKOpticalFlow::create();

  // create our video reader
  Ptr<cudacodec::VideoReader> d_reader =
    cudacodec::createVideoReader((String)std::getenv("VIDEO_URL"));

  // get the video info
  cv::cudacodec::FormatInfo videoFormat = d_reader->format();

  // now the size we'll use for thumbnails!
  Size thumbnailSize = size_maintain_aspect(
    videoFormat.width, videoFormat.height,
    atoi(std::getenv("DETECTION_WIDTH")),
    -1
  );
  float totalPixels = thumbnailSize.width * thumbnailSize.height;

  Size previewSize = size_maintain_aspect(
    videoFormat.width, videoFormat.height,
    800,
    -1
  );

  // get our detection zone (aka contour)
  auto detectionZone = init_detection_zone(
    videoFormat.width, videoFormat.height);
  // std::cout << "detectionZone: " << detectionZone << std::endl;

  // how frequently should we run yolo
  int detectionFrequency = atoi(std::getenv("DETECTION_FREQUENCY"));
  float minPixelMotion = atof(std::getenv("MOTION_MIN_PERCENTAGE"));
  int mqttPublishFrequency = atoi(std::getenv("MQTT_PUBLISH_FREQUENCY"));

  // blur to reduce noise!
  cv::Ptr<cv::cuda::Filter> filter = cv::cuda::createGaussianFilter(
    24,
    24,
    // Size(21, 21),
    Size(3, 3),
    0
  );

  // record our starting time
  long msLastMotion = 0;
  long frameCountMotionSequence = 0;

  // our sensor state and the last time we published state
  SensorState currentState = off;
  long msLastStateOn = 0;
  long msLastStateUpdate = 0;

  // the main loop!
  for (;;) {
    // attempt to read the frame
    if (!d_reader->nextFrame(d_frame))
      break;

    // get the timestamp for this frame
    long frameTimestamp = duration_cast< milliseconds >(
      system_clock::now().time_since_epoch()
    ).count();

    // resize the image
    cuda::resize(d_frame, d_frame_thumb, thumbnailSize, 0, 0, INTER_AREA);

    // convert to grayscale
    // cv::cuda::cvtColor(d_frame_thumb, d_frame_thumb_gray, CV_BGRA2GRAY);
    // filterGray->apply(d_frame_thumb_gray, d_frame_thumb_gray);

    // blur the frame before bg segmentation
    d_frame_thumb_blur = d_frame_thumb.clone();
    filter->apply(d_frame_thumb_blur, d_frame_thumb_blur);

    // background segmentation!
    pBG->apply(d_frame_thumb_blur, d_fgmask);

    // count number of non-zero pixels
    float nonZeroPixels = cuda::countNonZero(d_fgmask) / totalPixels;

    // only do this during debugging!
    // d_frame_last.download(h_frame_last);

    // do we have a motion alarm?
    bool frameAlarm = false;

    // store our json detections
    json frameDetections;

    // if (contoursAboveMin > 0)
    if (nonZeroPixels >= minPixelMotion)
    {
      if (frameCountMotionSequence % detectionFrequency == 0)
      {
        // GPU VERSION!
        yolo->Detect(
          d_frame_thumb,
          frameDetections
        );

        // also only during debugging
        // h_frame_last = draw_detections(h_frame_last, frameDetections, false, detectionZone);

        // check to see if we have people within our detection zone!
        if (frameDetections["count"] > 0)
        {
          for(size_t i = 0; i< frameDetections["count"]; i++ )
          {
            if (frameDetections["detections"][i]["class"] != "person")
            {
              continue;
            }

            // extract
            int personX = (int)((float)frameDetections["detections"][i]["x"] * videoFormat.width);
            int personY = (int)((float)frameDetections["detections"][i]["y"] * videoFormat.height);
            int personW = (int)((float)frameDetections["detections"][i]["w"] * videoFormat.width);
            int personH = (int)((float)frameDetections["detections"][i]["h"] * videoFormat.height);

            if (personX < 0) { personX = 0; }
            if ((personX + personW) > videoFormat.width) { personW -= (personX + personW) - videoFormat.width; }
            if (personY < 0) { personY = 0; }
            if ((personY + personH) > videoFormat.height) { personH -= (personY + personH) - videoFormat.height; }

            // debugger
            // std::cout << peopleCount << "== X:" << personX << " - Y: " << personY << " - W: " << personW << " - H: " << personH << std::endl;

            // determine the bottom center point
            auto personBottomCenter = cv::Point(
              personX + (personW / 2.0),
              personY + personH
            );

            // now lets see if the point is inside the contour
            auto insideZone = cv::pointPolygonTest(
              detectionZone,
              personBottomCenter,
              false
            );
            // std::cout << "    Center: " << personBottomCenter << " - Inside: " << insideZone << std::endl;

            if (insideZone >= 0) {
              frameAlarm = true;
              break;
            }
          }
          // std::cout << "-------------------------------------------" << std::endl;
        }
      }

      // bump the motion count (used to run yolo every N frames)
      frameCountMotionSequence++;
    }
    else {
      frameCountMotionSequence = 0;
    }

    // also only during debugging
    // cv::imshow("Frame", h_frame_last);

    bool didStateChange = false;
    if (frameAlarm == true) {
      // update the timestamp of when we last detected an alarm
      msLastStateOn = frameTimestamp;

      // update the current state
      if (currentState == off) {
        currentState = on;
        didStateChange = true;
      }
    }

    if (didStateChange == true ||
      (frameTimestamp - msLastStateUpdate) >= mqttPublishFrequency)
    {
      if ((frameTimestamp - msLastStateOn) >= mqttPublishFrequency)
      {
        currentState = off;
      }

      // we've updated the state
      msLastStateUpdate = frameTimestamp;

      // publish our state!
      mqttTopic.publish(currentState == on ? "on" : "off");

      // save our image
      if (didStateChange == true && currentState == on) {
        // download the frame from gpu to cpu
        cv::Mat h_frame_last;
        d_frame.download(h_frame_last);

        // resize the frame
        cv::resize(h_frame_last, h_frame_last, previewSize, 0, 0, INTER_AREA);

        // draw the detection
        h_frame_last = draw_detections(h_frame_last, frameDetections, false, detectionZone);

        // frame filename
        std::string frame_filename = to_string(frameTimestamp) + ".jpg";

        // build our paths
        std::string temp_path = "/tmp-snap/" + frame_filename;
        std::string dest_path = "/data/" + frame_filename;
        
        // write to the tmp path
        imageWrite(h_frame_last, temp_path);

        // copy the file
        copyFile(temp_path, dest_path);

        // remove the tmp file
        remove(temp_path.c_str());

        // publish our attributes
        json currentAttributes;
        currentAttributes["time"] = frameTimestamp;
        currentAttributes["url"] = std::getenv("THUMBNAIL_URL") + frame_filename;
        client.publish(
          string(std::getenv("MQTT_TOPIC")) + "/attributes",
          currentAttributes.dump(),
          0,
          true
        );

        // release the image
        h_frame_last.release();
      }
    }

    // allow the screen to draw our images...
    // cv::waitKey(33);
  }

  // encode_image(params, h_frame_bg);

  // graceful shutdown
  client.publish(
    string(std::getenv("MQTT_TOPIC")) + "/status",
    "offline",
    0,
    true
  );

  // mqtt disconnect
  client.disconnect();
}


// entry point
int main(int argc, const char* argv[])
{
  // init yolo
  init_yolo();

  // init cuda (is this even necessary)?
  init_cuda();

  // kick off the decoding!
  process();

clean_exit:
  // Clear up things before exiting
  std::cout << "Exiting..." << std::endl;
  if(yolo) delete yolo;
  return 0;
}

