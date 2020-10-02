# build (need to figure out how to build a static binary)
FROM nvidia/cuda:9.2-cudnn7-devel

# versions
ENV OPENCV_VERSION=3.4.3

# Install dependent packages
RUN apt-get -y update && \
    apt-get install -y \
      ca-certificates \
      wget \
      curl \
      nano \
      git \
      build-essential \
      nasm \
      libgl1-mesa-glx \
      pkg-config \
      cuda-npp-9-0 \
      unzip \
      cmake \
      libgtk2.0-dev \
      libcurl4-openssl-dev && \
    apt-get clean && \
    apt-get autoremove && \
    rm -rf /var/lib/apt/lists/*

# nvcodec headers (from FFmpeg project)
RUN git clone --single-branch -b n9.1.23.2 https://github.com/FFmpeg/nv-codec-headers /root/nv-codec-headers && \
  cd /root/nv-codec-headers &&\
  make -j"$(nproc)" && \
  make install -j"$(nproc)" && \
  cd /root && rm -rf nv-codec-headers

# Compile and install ffmpeg from source
RUN git clone --single-branch -b n4.3.1 https://github.com/FFmpeg/FFmpeg /root/ffmpeg && \
  cd /root/ffmpeg && \
  PKG_CONFIG_PATH="/usr/local/lib/pkgconfig" ./configure \
    --enable-nonfree \
    --enable-nvenc \
    --enable-cuda \
    --enable-cuvid \
    --enable-libnpp \
    --enable-pic \
    --enable-shared \
    --extra-cflags=-I/usr/local/cuda/include \
    --extra-cflags=-I/usr/local/include \
    --extra-ldflags=-L/usr/local/cuda/lib64 \
    --disable-manpages \
    --disable-doc \
    --disable-podpages && \
  make -j"$(nproc)" && \
  make install -j"$(nproc)" && \
  ldconfig && \
  cd /root && \
  rm -rf ffmpeg

# need this to build?
ENV NVIDIA_DRIVER_CAPABILITIES $NVIDIA_DRIVER_CAPABILITIES,video

# OpenCV (3.4.3)
# add custom patch to fix nvcodec and double max undecoded packets
# still have an issue with buffer size I believe
RUN cd ~ && \
    wget -O opencv.zip https://github.com/opencv/opencv/archive/$OPENCV_VERSION.zip && \
    wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/$OPENCV_VERSION.zip && \
    unzip opencv.zip && \
    unzip opencv_contrib.zip && \
    rm -f opencv.zip && \
    rm -f opencv_contrib.zip && \
    cd ~/opencv-$OPENCV_VERSION/ && \
    wget https://gist.githubusercontent.com/sberryman/936c3aa7918b94daf07bd4923533e9f6/raw/3ea11f448f49c341bc3d8de72f6d0dcd99bc7375/hevc.patch && \
    patch -p1 < hevc.patch && \
    rm hevc.patch && \
    mkdir build && \
    cd build && \
    cmake -D CMAKE_BUILD_TYPE=RELEASE \
      -D CMAKE_INSTALL_PREFIX=/usr/local \
      -D INSTALL_PYTHON_EXAMPLES=OFF \
      -D INSTALL_C_EXAMPLES=OFF \
      -D OPENCV_EXTRA_MODULES_PATH=~/opencv_contrib-$OPENCV_VERSION/modules \
      -D BUILD_opencv_python2=OFF \
      -D BUILD_opencv_python3=OFF \
      -D BUILD_DOCS=OFF \
      -D BUILD_TESTS=OFF \
      -D BUILD_PERF_TESTS=OFF \
      -D BUILD_JAVA=OFF \
      -D WITH_FFMPEG=ON \
      -D WITH_CUDA=ON \
      -D BUILD_opencv_dnn=ON \
      -D ENABLE_FAST_MATH=ON \
      -D CUDA_FAST_MATH=ON \
      -D WITH_CUBLAS=ON \
      -D WITH_NVCUVID=ON \
      -D ENABLE_PRECOMPILED_HEADERS=OFF \
      -D CUDA_NVCC_FLAGS="-D_FORCE_INLINES --expt-relaxed-constexpr" \
      -D CUDA_GENERATION="Pascal" \
      -D CUDA_nvcuvid_LIBRARY=/usr/lib/x86_64-linux-gnu/libnvcuvid.so.1 \
      -D BUILD_EXAMPLES=OFF .. && \
    make -j"$(nproc)" && \
    make install -j"$(nproc)" && \
    ldconfig && \
    cd ~ && \
    rm -rf opencv-$OPENCV_VERSION && \
    rm -rf opencv_contrib-$OPENCV_VERSION && \
    ln -s /usr/lib/x86_64-linux-gnu/libnvcuvid.so.1 /usr/lib/x86_64-linux-gnu/libnvcuvid.so

# darknet! (symbolic link the shared library)
# should we copy the headers?
RUN cd /opt && \
    git clone --depth=1 https://github.com/pjreddie/darknet && \
    cd /opt/darknet && \
    sed -i 's/GPU\=0/GPU\=1/' Makefile && \
    sed -i 's/OPENCV\=0/OPENCV\=1/' Makefile && \
    sed -i 's/CUDNN\=0/CUDNN\=1/' Makefile && \
    make -j"$(nproc)"

# lets try a small inference only version of yolo
RUN export YOLO2_LIGHT_V=1 && \
    cd /opt && \
    git clone --depth=1 -b temp-no-print https://github.com/sberryman/yolo2_light.git && \
    cd yolo2_light && \
    sed -i 's/GPU\=0/GPU\=1/' Makefile && \
    sed -i 's/OPENCV\=0/OPENCV\=1/' Makefile && \
    make -j"$(nproc)"

# weights, needs to be baked into the image or code written to pull it
# if it doesn't exist...
# src: https://pjreddie.com/media/files/yolov3.weights
RUN mkdir /weights && \
    curl "http://192.168.7.75:9050/darknet/yolov3.weights" > /weights/yolov3.weights

RUN cd /opt && \
  git clone https://github.com/eclipse/paho.mqtt.c.git && \
  cd paho.mqtt.c && \
  git checkout v1.2.1 && \
  cmake -Bbuild -H. && \
  cmake --build build/ --target install && \
  cd ../ && \
  wget https://github.com/eclipse/paho.mqtt.cpp/archive/v1.0.1.tar.gz && \
  tar -xf v1.0.1.tar.gz && \
  rm v1.0.1.tar.gz && \
  cd paho.mqtt.cpp-1.0.1/ && \
  cmake -Bbuild -H. -DPAHO_WITH_SSL=FALSE && \
  cmake --build build/ --target install

# copy application files and compile!
WORKDIR /app
COPY . .
RUN cd src && \
  make && \
  cp appFunction ../

CMD [ "./appFunction" ]

