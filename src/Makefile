CPP=g++

all: yolo_wrapper app

yolo_wrapper:
	$(CPP) -o yolo_wrapper.o \
		-DGPU \
		-DCUDNN \
		-DOPENCV \
		-c -std=c++0x  \
		yolo/wrapper.cpp \
		-I/usr/local/cuda/include \
		-L/usr/local/cuda/lib64 -lcuda \
		-I/usr/local/cudnn/include \
		-L/usr/local/cudnn/lib64 -lcudnn

app: yolo_wrapper
	$(CPP) -o appFunction \
		-DGPU \
		-DCUDNN \
		-DOPENCV \
		-ggdb \
		-std=c++0x \
		-Wall \
			app.cpp dynlink_nvcuvid.cpp \
		/opt/yolo2_light/obj/additionally.o \
		/opt/yolo2_light/obj/gpu.o \
		/opt/yolo2_light/obj/yolov2_forward_network.o \
		/opt/yolo2_light/obj/yolov2_forward_network_gpu.o \
		/opt/yolo2_light/obj/yolov2_forward_network_quantized.o \
		yolo_wrapper.o \
		/opt/yolo2_light/obj/box.o \
		`pkg-config --cflags --libs opencv` \
		-I/usr/local/cuda/include \
		-L/usr/local/cuda/lib64 -lcuda \
		-I/usr/local/cudnn/include \
		-L/usr/local/cudnn/lib64 -lcudnn \
		-ldl \
		-lcurl \
		-lcudart -lcublas -lcurand \
		-lpthread \
		-L/usr/local/lib \
		-I/usr/local/include \
		-lpaho-mqttpp3 -lpaho-mqtt3a \
		-Wl,-rpath,/usr/local/lib

