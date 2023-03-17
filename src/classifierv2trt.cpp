#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <cmath>
#include <fstream>

#include "crop_resize.h"
#include "mask.h"
#include "argmax.h"

#include <opencv2/opencv.hpp>
#include "cuda_utils.h"
#include "logging.h"
#include <cuda_runtime.h>
#include <cstdint>

#include "NvInfer.h"
#include "NvOnnxParser.h"

#include "spdlog/spdlog.h"

#include <cstdlib>


using namespace std;
using namespace cv;
using namespace nvonnxparser;
using namespace nvinfer1;

// Constants
#define MAX_IMAGE_INPUT_SIZE_THRESH 3000 * 3000
#define INPUT_BLOB_NAME "input"
#define OUTPUT_BLOB_NAME "output"

// trt specific 
static Logger gLogger;
static IRuntime* runtime;
static ICudaEngine* engine;
static IExecutionContext* context;

static float* buffers[2];

// Hyperparameters
static int inputIndex;
static int outputIndex;
static int InputH;
static int InputW;
static int InputChannel;
static int OutputChannel;
static int BatchSize;
static int DebugImgCounter = 0;
static bool DebugImage = 0;

// Pointer of memory area for images in gpu host
static uint8_t** hostbuffers;
static uint8_t** devicebuffers;

// Pointer of memory area for a single image in gpu
// static float* mask_buffer = nullptr;


//Debug
static float* crop_hostbuffer_debug = nullptr;
/*
static float* mask_buffer_debug = nullptr;
static float* crop_devicebuffer_debug = nullptr;
static float* inference_buffer_debug = nullptr;
*/

/*
static float* resized_buffer = nullptr;
static float* argmax_buffer = nullptr;
*/



// Pointer of memory area for the batched softmax
static float* argmax_buffer_cpu = nullptr;

//For the cuda stream
static cudaStream_t stream;

// Define the BoundingBox and ImageWithBoundingBoxes structures.
struct BoundingBox {
    int bbox_id; // bbox_id
    int tlx; // top left x
    int tly; // top left y
    int brx; // bottom left x
    int bry; // bottom right y
};

struct Metdata_for_Crop{
  int tlx; 
  int tly; 
  int brx; 
  int bry; 
  int width;
  int height;
};

/*
struct Metdata_for_Mask{
  int tlx; 
  int tly; 
  int brx; 
  int bry; 
};
*/

struct Metdata_for_Resize{
  int tlx; 
  int tly; 
  int brx; 
  int bry; 
  float scale;
};


struct ImageWithBoundingBoxes {
    int image_id;
    unsigned char *image_data;
    int image_width;
    int image_height;
    int num_bounding_boxes;
    BoundingBox *bounding_boxes;
    Metdata_for_Crop *crop_metadata;
    // Metdata_for_Mask *mask_metadata;
    Metdata_for_Resize *resize_metadata;

    // Copy constructor
    ImageWithBoundingBoxes(const ImageWithBoundingBoxes &other)
        : image_id(other.image_id), image_data(nullptr), image_width(other.image_width), image_height(other.image_height),
          num_bounding_boxes(other.num_bounding_boxes), bounding_boxes(nullptr), crop_metadata(nullptr)
    {
        // Allocate memory for bounding_boxes
        bounding_boxes = new BoundingBox[num_bounding_boxes];
        // Copy bounding_boxes from other object
        std::copy(other.bounding_boxes, other.bounding_boxes + num_bounding_boxes, bounding_boxes);

        // Allocate memory for crop_metadata
        crop_metadata = new Metdata_for_Crop[num_bounding_boxes];
        // Copy crop_metadata from other object
        std::copy(other.crop_metadata, other.crop_metadata + num_bounding_boxes, crop_metadata);

        // Allocate memory for resize_metadata
        resize_metadata = new Metdata_for_Resize[num_bounding_boxes];
        // Copy resize_metadata from other object
        std::copy(other.resize_metadata, other.resize_metadata + num_bounding_boxes, resize_metadata);
    }

    //Destructor
    ~ImageWithBoundingBoxes() {
      // Deallocate memory for bounding_boxes
      delete[] bounding_boxes;
      // Deallocate memory for crop_metadata
      delete[] crop_metadata;
      // Deallocate memory for resize_metadata
      delete[] resize_metadata;
  }
};

// Define the Batching structures

struct ImageIndex
{
    int imageIndex; // id of image
    int bboxIndex; // id of bbox
};
struct BatchInfo
{
    std::vector<ImageIndex> imageindices; // list of ImageIndex
    int batchindex;   // id of batch 
};


// Storing list of all data
std::vector<ImageWithBoundingBoxes*> list_of_images;

// Find the length of boxes
int getBBoxLength(){
  int counter = 0;
  for(auto i : list_of_images)
    counter += i->num_bounding_boxes;
  return counter;
}

extern "C" {
    void batchInitialize();
    void InitializeGPUMemory(int batchSize, int inputW, int inputH, int inputChannel, int outputChannel, int debugLevel, int debugImage);
    int add_images_with_bounding_boxes(ImageWithBoundingBoxes *images, int image_id);
    int load_onnx(char *onnx_filePath, char *trt_filePath,int isFP16);
    int load_engine(char* trt_filePath);
    int do_inference(float *output, float* meta_ids);
}

void batchInitialize(){
  spdlog::debug("batchInitialize Called");
  for (auto &image : list_of_images) {
    delete image;
  }
  std::vector<ImageWithBoundingBoxes*>().swap(list_of_images);
  list_of_images.clear();
  spdlog::debug("batchInitialize Done");
}

// Define the add_images_with_bounding_boxes function.
int add_images_with_bounding_boxes(ImageWithBoundingBoxes *image, int image_id) {
  
  ImageWithBoundingBoxes *image_copy = new ImageWithBoundingBoxes(*image);

  spdlog::debug("Image {} ", image_id);
  spdlog::debug("Imagesize Width {} Height {} ", image->image_width, image->image_height);
  spdlog::debug("Number of bounding boxes: {} ", image->num_bounding_boxes);
  for (int j = 0; j < image->num_bounding_boxes; j++) {
      auto bounding_box = image->bounding_boxes[j];
      spdlog::debug("Bounding box tlx = {} tly = {} brx = {} bry = {}", 
                                                  bounding_box.tlx,
                                                  bounding_box.tly, 
                                                  bounding_box.brx, 
                                                  bounding_box.bry);
      auto crop_metadata = image->crop_metadata[j];
      spdlog::debug("crop metadata box tlx = {} tly = {} brx = {} bry = {}", 
                                                  crop_metadata.tlx,
                                                  crop_metadata.tly, 
                                                  crop_metadata.brx, 
                                                  crop_metadata.bry);

      auto resize_metadata = image->resize_metadata[j];
      spdlog::debug("resize metadata box tlx = {} tly = {} brx = {} bry = {} scale = {}", 
                                                  resize_metadata.tlx,
                                                  resize_metadata.tly, 
                                                  resize_metadata.brx, 
                                                  resize_metadata.bry,
                                                  resize_metadata.scale);                                             
  }

  /*
  //Debug the input image
  unsigned char* image_data = image->image_data;
  Mat new_output = Mat(image->image_height, image->image_width, CV_8UC3);
  new_output.data = image_data;
  imwrite("/app/final.jpg", new_output);
  */

  //allocate image information
  list_of_images.emplace_back(image_copy);
  spdlog::debug("Counter : {}", getBBoxLength());

  // Transfer host to device for full images
  size_t size_image = image->image_width * image->image_height * InputChannel;
  hostbuffers[image_id] = image->image_data;

  spdlog::debug("Host allocated : {}", image_id);

  CUDA_CHECK(cudaMemcpyAsync(devicebuffers[image_id], hostbuffers[image_id], size_image, cudaMemcpyHostToDevice, stream));

  spdlog::debug("Device allocated : {}", image_id);
  
  return 1;
}

void InitializeGPUMemory(int batchSize, int inputW, int inputH, int inputChannel, int outputChannel, int debugLevel, int debugImage){
  
  // Global init
  BatchSize = batchSize;
  InputW = inputW;
  InputH = inputH;
  InputChannel = inputChannel;
  OutputChannel = outputChannel;
  DebugImage = debugImage;

  // logger
  // NOTSET=0
  // DEBUG=10
  // INFO=20
  // WARN=30
  // ERROR=40
  // CRITICAL=50
  if(debugLevel == 10){
    spdlog::set_level(spdlog::level::debug);
  }
  else if(debugLevel == 20){
    spdlog::set_level(spdlog::level::info);
  }
  spdlog::debug("init called");

  // Set device: Hardcoded to 0
  // cudaSetDevice(0);

  // prepare host cache for input image:  All input RGB data
  hostbuffers = (uint8_t**)malloc(sizeof(uint8_t*) * BatchSize);
  for(int i = 0 ; i < BatchSize; i++)
    CUDA_CHECK(cudaMallocHost(&hostbuffers[i], MAX_IMAGE_INPUT_SIZE_THRESH * InputChannel)); // RGB

  // prepare device cache for input image:  All input RGB data
  devicebuffers = (uint8_t**)malloc(sizeof(uint8_t*) * BatchSize);
  for(int i = 0 ; i < BatchSize; i++)
    CUDA_CHECK(cudaMalloc(&devicebuffers[i], MAX_IMAGE_INPUT_SIZE_THRESH * InputChannel)); // RGB

  // prepare masked data cache in device memory 
  // CUDA_CHECK(cudaMalloc((void**)&mask_buffer, 1 * 1 * InputH * InputW * sizeof(float)));

  
  // debug: Should be removed after debugging
  if(DebugImage){
    CUDA_CHECK(cudaMallocHost((void**)&crop_hostbuffer_debug, 1 * InputChannel * InputH * InputW * sizeof(float)));
  }
  /*
  CUDA_CHECK(cudaMallocHost((void**)&mask_buffer_debug, 1 * 1 * InputH * InputW * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void**)&crop_devicebuffer_debug, 1 * 3 * InputH * InputW * sizeof(float)));
  CUDA_CHECK(cudaMallocHost((void**)&inference_buffer_debug, 1 * InputChannel * InputH * InputW * sizeof(float)));
  */

  /*
  // prepare resized data cache for single image in device memory
  CUDA_CHECK(cudaMalloc((void**)&resized_buffer, 1 * InputChannel * InputH * InputW * sizeof(float)));
  prepare softmax data cache for batch in device memory
  CUDA_CHECK(cudaMalloc((void**)&argmax_buffer, BatchSize * OutputChannel * InputH * InputW * sizeof(float) ));
  */
  
  
  // prepare softmax data cpu cache for batch in device memory
  CUDA_CHECK(cudaMallocHost((void**)&argmax_buffer_cpu, BatchSize * OutputChannel * sizeof(float)));

}

int load_onnx(char *filename, char *enginename, int isFP16){

  std::ifstream file(filename, std::ios::binary);
  spdlog::debug("Onnx path: {}", filename);
  spdlog::debug("TRT path: {}", enginename);
  
  if (!file.good()) {
    std::cerr << "ONNX read " << filename << " error!" << std::endl;
    return -1;
  }
  IBuilder* builder = createInferBuilder(gLogger);
  builder->setMaxBatchSize(BatchSize);
  uint32_t flag = 1U <<static_cast<uint32_t>
    (NetworkDefinitionCreationFlag::kEXPLICIT_BATCH); 

  INetworkDefinition* network = builder->createNetworkV2(flag);
  IParser*  parser = createParser(*network, gLogger);
  parser->parseFromFile(filename, 3);
  for (int32_t i = 0; i < parser->getNbErrors(); ++i)
  {
    std::cout << parser->getError(i)->desc() << std::endl;
  }

  IOptimizationProfile* profile = builder->createOptimizationProfile();
  profile->setDimensions(INPUT_BLOB_NAME, OptProfileSelector::kMIN, Dims4(1,InputChannel,InputH,InputW));
  profile->setDimensions(INPUT_BLOB_NAME, OptProfileSelector::kOPT, Dims4(BatchSize,InputChannel,InputH,InputW));
  profile->setDimensions(INPUT_BLOB_NAME, OptProfileSelector::kMAX, Dims4(BatchSize,InputChannel,InputH,InputW));


  IBuilderConfig* config = builder->createBuilderConfig();
  config->setMaxWorkspaceSize(1U << 20);
  config->addOptimizationProfile(profile);
  if(isFP16){
    config->setFlag(BuilderFlag::kFP16);
  }
  IHostMemory*  serializedModel = builder->buildSerializedNetwork(*network, *config);
  std::ofstream p(enginename, std::ios::binary);
  if (!p) {
    std::cerr << "TRT engine could not open plan output file" << std::endl;
    return -1;
  }
  p.write(reinterpret_cast<const char*>(serializedModel->data()), serializedModel->size());

  delete parser;
  delete network;
  delete config;
  delete builder;
  delete serializedModel;
  return 1;
}

int load_engine(char* filename){

  std::ifstream file(filename, std::ios::binary);
  if (!file.good()) {
    std::cerr << "read " << filename << " error!" << std::endl;
    return -1;
  }
  char *trtModelStream = nullptr;
  size_t size = 0;
  file.seekg(0, file.end);
  size = file.tellg();
  file.seekg(0, file.beg);
  trtModelStream = new char[size];
  assert(trtModelStream);
  file.read(trtModelStream, size);
  file.close();

  runtime = createInferRuntime(gLogger);
  assert(runtime != nullptr);
  engine = runtime->deserializeCudaEngine(trtModelStream, size);
  assert(engine != nullptr);
  context = engine->createExecutionContext();
  assert(context != nullptr);
  delete[] trtModelStream;
  
  assert(inputIndex == 0);
  assert(outputIndex == 1);
  inputIndex = engine->getBindingIndex(INPUT_BLOB_NAME);
  outputIndex = engine->getBindingIndex(OUTPUT_BLOB_NAME);
  
  assert(inputIndex == 0);
  assert(outputIndex == 1);
  // Create GPU buffers on device
  CUDA_CHECK(cudaMalloc((void**)&buffers[inputIndex], BatchSize * InputChannel * InputH * InputW  * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void**)&buffers[outputIndex],BatchSize * OutputChannel * sizeof(float)));

  // prepare stream
  CUDA_CHECK(cudaStreamCreate(&stream));
  
  context->setBindingDimensions(inputIndex, Dims4(BatchSize, InputChannel, InputH, InputW));
  context->setOptimizationProfileAsync(0, stream);
 
  return 1;
}

int do_inference(float *output, float *meta_ids){
  
  // Get number of bboxes to run the processing
  int total_size_to_run = getBBoxLength();
  spdlog::debug("Counter : {}", getBBoxLength());
  int batch_size_to_be_taken = BatchSize;
  float batchRatio = static_cast<float>(total_size_to_run)/static_cast<float>(batch_size_to_be_taken);
  int batchTotal = ceil(batchRatio);
  int i = 0;
  ImageIndex in;
  std::vector<BatchInfo> list_of_batch;
  BatchInfo tempInfo;
  int batchindex = 0;
  tempInfo.batchindex = batchindex;

  // Storing the image and bbox indices for each image in list_of_batch[i]
  for (int iid = 0; iid < list_of_images.size(); iid++)
  {   
      for (int gid = 0; gid < list_of_images[iid]->num_bounding_boxes; gid++)
      {
          if (i < batch_size_to_be_taken)
          {  
                in.imageIndex = iid;
                in.bboxIndex = gid;
                tempInfo.imageindices.push_back(in);
                i++;
          }
          if (i == batch_size_to_be_taken)
          {
              tempInfo.batchindex = batchindex;
              list_of_batch.push_back(tempInfo);
              tempInfo.imageindices.clear();
              batchindex = batchindex+1;
              i = 0;
              tempInfo.batchindex = batchindex;
          }
      }
  }
  if (i != batch_size_to_be_taken && i != 0)
  {
      list_of_batch.push_back(tempInfo);
  }

  spdlog::debug("Number of boxes to the module : {}", total_size_to_run);
  spdlog::debug("Batch Size : {}",batch_size_to_be_taken);
  spdlog::debug("Batch Total : {}",batchTotal);

  // Start pointer of the output values
  float *ptr = (float *)output;
  int total_idx = 0; // 0 to (total_size_to_run-1)
  
  // Inference for each batch
  for (int bs = 0; bs < batchTotal; bs++)
  {   
      spdlog::debug("Batch number = {}",bs);
      std::vector<ImageIndex> batchIndices = list_of_batch[bs].imageindices;
      int currentBS = list_of_batch[bs].imageindices.size();
      spdlog::debug("Number of images to be processed, currentBS = {}",currentBS);
      
      // Assign starting point
      spdlog::debug("Input Index {}",inputIndex);
      float* buffer_idx = (float *)buffers[inputIndex];

      // Current Batch
      for (int index = 0; index < batchIndices.size(); index++) {
          int image_index = batchIndices[index].imageIndex;
          int bbox_index = batchIndices[index].bboxIndex;
          spdlog::debug("Image idx, Bbox id {} {}",image_index, bbox_index);

          auto image = list_of_images[image_index];
          int image_id = image->image_id;

          auto crop_metadata = image->crop_metadata[bbox_index];
          spdlog::debug("crop metadata box tlx = {} tly = {} brx = {} bry = {}", 
                                                      crop_metadata.tlx,
                                                      crop_metadata.tly, 
                                                      crop_metadata.brx, 
                                                      crop_metadata.bry);
          // auto mask_metadata = image->mask_metadata[bbox_index];
          // spdlog::debug("mask metadata box tlx = {} tly = {} brx = {} bry = {}", 
          //                                             mask_metadata.tlx,
          //                                             mask_metadata.tly, 
          //                                             mask_metadata.brx, 
          //                                             mask_metadata.bry);
          auto resize_metadata = image->resize_metadata[bbox_index];
          spdlog::debug("resize metadata box tlx = {} tly = {} brx = {} bry = {} scale = {}", 
                                                      resize_metadata.tlx,
                                                      resize_metadata.tly, 
                                                      resize_metadata.brx, 
                                                      resize_metadata.bry,
                                                      resize_metadata.scale); 
           
          Point pTopLeft(crop_metadata.tlx, crop_metadata.tly);
          Point pBottomRight(crop_metadata.brx, crop_metadata.bry);
          cv::Rect context_crop(pTopLeft, pBottomRight);

          size_t size0 = index *  InputH * InputW * InputChannel ;
                     
          crop_resize_kernel_img(devicebuffers[image_id], image->image_width, image->image_height,
                                buffer_idx, InputW, InputH,
                                context_crop,
                                1,
                                resize_metadata.scale,
                                size0,
                                stream); 

          if(DebugImage){
            // // Debug the resized image
            cudaStreamSynchronize(stream);
            cudaMemcpyAsync(crop_hostbuffer_debug,buffer_idx + size0, InputH * InputW * InputChannel * sizeof(float),cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
            cv::Mat float_R = cv::Mat(InputH, InputW,CV_32FC1, crop_hostbuffer_debug);
            cv::Mat float_G = cv::Mat(InputH, InputW,CV_32FC1, crop_hostbuffer_debug +   InputW * InputH );
            cv::Mat float_B = cv::Mat(InputH, InputW,CV_32FC1, crop_hostbuffer_debug + 2*InputW * InputH );
            float_R = float_R * 0.229 + 0.485;
            float_G = float_G * 0.224 + 0.456;
            float_B = float_B * 0.225 + 0.406;
            cv::Mat char_r, char_g, char_b;
            float_R.convertTo(char_r, CV_8UC1, 255.0);
            float_G.convertTo(char_g, CV_8UC1, 255.0);
            float_B.convertTo(char_b, CV_8UC1, 255.0);
            cv::Mat channels[3] = {char_b, char_g, char_r};
            cv::Mat new_output;
            cv::merge(channels, 3, new_output);
            // std::string imgPath = "/app/mounts/debug/crop"+to_string(index)+"-"+to_string(debugImgCounter)+".jpg";
            std::string imgPath = "/app/mounts/debug/crop"+to_string(DebugImgCounter)+".jpg";
            cv::imwrite(imgPath, new_output);
            DebugImgCounter = DebugImgCounter + 1;
        }

      }

      spdlog::debug("Enqueue started inputIndex {} outputIndex {}", inputIndex, outputIndex);
      /*
      // Debug the input tensors for trt
      cudaStreamSynchronize(stream);
      cudaMemcpyAsync(inference_buffer_debug,buffers[inputIndex], InputChannel * InputH * InputW * sizeof(float) ,cudaMemcpyDeviceToHost, stream);
      for (int i = 0; i < currentBS * InputChannel * InputH * InputW; i++) {
        cout << inference_buffer_debug[i] << " ";
      }
      char input_filename[50]; 
      sprintf(input_filename, "/app/input.bin"); 
      FILE* input_fp = fopen(input_filename, "wb"); 
      fwrite(inference_buffer_debug, 1, InputChannel * InputH * InputW * sizeof(float), input_fp); 
      fclose(input_fp);
      */
      
      // TRT inference enqueue
      context->enqueue(currentBS, (void**)buffers, stream, nullptr);

      spdlog::debug("Enqueue done");
      /*
      // Debug the output tensors for trt
      cudaStreamSynchronize(stream);
      cudaMemcpyAsync(inference_buffer_debug,buffers[outputIndex], InputChannel * InputH * InputW * sizeof(float) ,cudaMemcpyDeviceToHost, stream);
      for (int i = 0; i < currentBS * InputChannel * InputH * InputW; i++) {
        cout << inference_buffer_debug[i] << " ";
      }
      char output_filename[50]; 
      sprintf(output_filename, "/app/output.bin"); 
      FILE* output_fp = fopen(output_filename, "wb"); 
      fwrite(inference_buffer_debug, 1, InputChannel * InputH * InputW * sizeof(float), output_fp); 
      fclose(output_fp); 
      */


      CUDA_CHECK(cudaMemcpyAsync(argmax_buffer_cpu, 
                              buffers[outputIndex], 
                              OutputChannel * currentBS * sizeof(float), 
                              cudaMemcpyDeviceToHost, 
                              stream));

      // Sync for each argmax output
      cudaStreamSynchronize(stream);
        
      // Current Batch
      for (int index = 0; index < batchIndices.size(); index++) {
        int image_index = batchIndices[index].imageIndex;
        int bbox_index = batchIndices[index].bboxIndex;
       
        auto image = list_of_images[image_index];
        int image_id = image->image_id;
        int bbox_id = image->bounding_boxes[bbox_index].bbox_id;

        
        
        spdlog::debug("Assigning output {}", total_idx);
        size_t size3 = OutputChannel;
        for (int i = 0; i < size3; i++) {
          ptr[total_idx*size3 + i] = argmax_buffer_cpu[index*size3 + i];
        }
        
        size_t size4 = 2;
        meta_ids[total_idx*size4 + 0] = image_id;
        meta_ids[total_idx*size4 + 1] = bbox_id;
        

        total_idx++; 
      
      }
  }
  return 1;
}
