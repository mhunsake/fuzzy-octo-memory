FROM nvcr.io/nvidia/tensorrt:20.01-py3

ARG DEBIAN_FRONTEND=noninteractive

################################################################################
# install opencv
# with opencv_contrib quality module
ARG OPENCV_VERSION=4.5.0

RUN apt update && apt install -y \
    build-essential cmake git pkg-config libgtk-3-dev \
    libavcodec-dev libavformat-dev libswscale-dev libv4l-dev \
    libxvidcore-dev libx264-dev libjpeg-dev libpng-dev libtiff-dev \
    gfortran openexr libatlas-base-dev python3-dev python3-numpy \
    libtbb2 libtbb-dev libdc1394-22-dev libopenexr-dev \
    libgstreamer-plugins-base1.0-dev libgstreamer1.0-dev

RUN wget -q --no-check-certificate https://github.com/opencv/opencv/archive/${OPENCV_VERSION}.zip -O opencv.zip \
 && wget -q --no-check-certificate https://github.com/opencv/opencv_contrib/archive/${OPENCV_VERSION}.zip -O opencv_contrib.zip \
 && unzip -qq opencv.zip -d /opt && rm -rf opencv.zip \
 && unzip -qq opencv_contrib.zip -d /opt && rm -rf opencv_contrib.zip 

RUN mkdir /opt/build && cd /opt/build && cmake \
        -D CMAKE_BUILD_TYPE=RELEASE \
	-D WITH_CUDA=ON \
        -D CMAKE_INSTALL_PREFIX=/usr/local \
	-D CMAKE_LIBRARY_PATH=/usr/local/cuda/lib64/stubs \
	-D OPENCV_GENERATE_PKGCONFIG=YES \
	-D BUILD_TESTS=OFF \
	-D BUILD_EXAMPLES=OFF \
	-D ENABLE_PRECOMPILED_HEADERS=OFF \
        -D OPENCV_EXTRA_MODULES_PATH=/opt/opencv_contrib-${OPENCV_VERSION}/modules \
        -D EIGEN_INCLUDE_PATH=/usr/include/eigen3 \
	-D BUILD_opencv_quality=ON \
	-D BUILD_opencv_alphamat=OFF \
	-D BUILD_opencv_aruco=OFF \
	-D BUILD_opencv_bgsegm=OFF \
	-D BUILD_opencv_bioinspired=OFF \
	-D BUILD_opencv_ccalib=OFF \
	-D BUILD_opencv_cnn_3dobj=OFF \
	-D BUILD_opencv_cudaarithm=ON \
	-D BUILD_opencv_cudabgsegm=ON \
	-D BUILD_opencv_cudacodec=ON  \
	-D BUILD_opencv_cudafeatures2d=ON  \
	-D BUILD_opencv_cudafilters=ON \
	-D BUILD_opencv_cudaimgproc=ON \
	-D BUILD_opencv_cudalegacy=ON  \
	-D BUILD_opencv_cudaobjdetect=ON  \
	-D BUILD_opencv_cudaoptflow=ON  \
	-D BUILD_opencv_cudastereo=ON  \
	-D BUILD_opencv_cudawarping=ON  \
	-D BUILD_opencv_cudev=ON \
	-D BUILD_opencv_cvv=OFF \
	-D BUILD_opencv_datasets=OFF \
	-D BUILD_opencv_dnn_objdetect=OFF \
	-D BUILD_opencv_dnn_superres=OFF \
	-D BUILD_opencv_dnns_easily_fooled=OFF \
	-D BUILD_opencv_dpm=OFF \
	-D BUILD_opencv_face=OFF \
	-D BUILD_opencv_freetype=OFF \
	-D BUILD_opencv_fuzzy=OFF \
	-D BUILD_opencv_hdf=OFF \
	-D BUILD_opencv_hfs=OFF \
	-D BUILD_opencv_img_hash=OFF \
	-D BUILD_opencv_intensity_transform=OFF \
	-D BUILD_opencv_julia=OFF \
	-D BUILD_opencv_line_descriptor=OFF \
	-D BUILD_opencv_matlab=OFF \
	-D BUILD_opencv_mcc=OFF \
	-D BUILD_opencv_optflow=OFF \
	-D BUILD_opencv_ovis=OFF \
	-D BUILD_opencv_phase_unwrapping=OFF \
	-D BUILD_opencv_plot=OFF \
	-D BUILD_opencv_rapid=OFF \
	-D BUILD_opencv_reg=OFF \
	-D BUILD_opencv_rgbd=OFF \
	-D BUILD_opencv_saliency=OFF \
	-D BUILD_opencv_sfm=OFF \
	-D BUILD_opencv_shape=OFF \
	-D BUILD_opencv_stereo=OFF \
	-D BUILD_opencv_structured_light=OFF \
	-D BUILD_opencv_superres=OFF \
	-D BUILD_opencv_surface_matching=OFF \
	-D BUILD_opencv_text=OFF \
	-D BUILD_opencv_tracking=OFF \
	-D BUILD_opencv_videostab=OFF \
	-D BUILD_opencv_viz=OFF \
	-D BUILD_opencv_xfeatures2d=OFF \
	-D BUILD_opencv_ximgproc=OFF \
	-D BUILD_opencv_xobjdetect=OFF \
	-D BUILD_opencv_xphoto=OFF \
        /opt/opencv-${OPENCV_VERSION}  \
    && make -j$(nproc) \
    && make install \
    && rm -rf /opt/build/* \
    && rm -rf /opt/opencv-${OPENCV_VERSION} \
    && rm -rf /opt/opencv_contrib-${OPENCV_VERSION} \
RUN apt-get remove \
        software-properties-common \
        build-essential cmake \
        libhdf5-dev \
        libopenblas-dev \
        libprotobuf-dev \
        libjpeg8-dev \
        libpng-dev \
        libtiff-dev \
        libwebp-dev \
        libopenjp2-7-dev \
        libtbb-dev \
        libtesseract-dev \
        python3-dev
RUN apt-get autoremove
RUN ldconfig

RUN ln -s /usr/local/include/opencv4/opencv2 /usr/local/include/opencv2


# TENSORFLOW SPECIFICALLY BUILT FOR CUDA 10.2
# https://github.com/tensorflow/tensorflow/issues/38194 
# https://github.com/alexshvid/tensorflow/releases/tag/v2.3.1-cuda10.2
WORKDIR /workspace/tmp
RUN wget https://github.com/alexshvid/tensorflow/releases/download/v2.3.1-cuda10.2/tensorflow-2.3.1-cp36-cp36m-linux_x86_64.whl
RUN pip3 install tensorflow-2.3.1-cp36-cp36m-linux_x86_64.whl
RUN pip3 install tensorflow-addons==0.11.2

# jupyter and such...
RUN pip3 install matplotlib jupyter tqdm xmltodict

# keras2onnx (OLD)
RUN pip3 install git+https://github.com/microsoft/onnxconverter-common
RUN pip3 install git+https://github.com/onnx/keras-onnx

# tensorflow2onnx (NEW - use this instead!)
RUN pip3 install git+https://github.com/onnx/tensorflow-onnx

WORKDIR /workspace

CMD ["/bin/bash"]
