FROM uvDockBase:3333/uvdep:p_amd64_22.06.27 as uvdep
FROM uvdeployment/shield:apilayer_trt_22.09.28 as apilayer_base

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # DEVELOPMENT DOCKER # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
FROM nvcr.io/nvidia/tensorrt:21.07-py3 as dev
RUN rm -rf /workspace

RUN sed -i 's|http://archive.|http://us.archive.|g' /etc/apt/sources.list \
&& apt-get -y update \
&& apt -y upgrade \
&& DEBIAN_FRONTEND=noninteractive apt-get install -y tzdata \
&& apt-get install -y --no-install-recommends \
    python3 \
    libczmq4 \
    libzmq3-dev \
    python3-pip \
    libglib2.0-dev \
    libsm-dev \
    libxrender1 \
    libxext-dev \
&& pip3 install numpy \
&& rm -rf /var/lib/apt/lists/*

RUN cd /tmp && \
	git clone -b 4.1.2 --single-branch https://github.com/opencv/opencv_contrib.git && \
	git clone -b 4.1.2 --single-branch https://github.com/opencv/opencv.git && \
	cd opencv && \
	mkdir build && \
	cd build && \
	cmake 	-D CMAKE_BUILD_TYPE=RELEASE \
		-D BUILD_NEW_PYTHON_SUPPORT=ON \
		-D CMAKE_INSTALL_PREFIX=/opt/opencv \
		-D INSTALL_C_EXAMPLES=OFF \
		-D INSTALL_PYTHON_EXAMPLES=OFF \
		-D OPENCV_EXTRA_MODULES_PATH=/tmp/opencv_contrib/modules \
		-D BUILD_opencv_python3=no \
		-D PYTHON_EXECUTABLE=/usr/bin/python2.7 \
        -D BUILD_EXAMPLES=OFF \
                -D PYTHON_DEFAULT_EXECUTABLE=/usr/bin/python2 \
                -D PYTHON2_NUMPY_INCLUDE_DIRS=/usr/local/lib/python2.7/dist-packages/numpy/core/include \
                -D BUILD_opencv_aruco=OFF \
                -D BUILD_opencv_bgsegm=OFF \
                -D BUILD_opencv_bioinspired=OFF \
                -D BUILD_opencv_calib3d=OFF \
                -D BUILD_opencv_ccalib=OFF \
                -D BUILD_opencv_core=ON \
                -D BUILD_opencv_cudaarithm=ON \
                -D BUILD_opencv_cudabgsegm=OFF \
                -D BUILD_opencv_cudacodec=OFF \
                -D BUILD_opencv_cudafilters=OFF \
                -D BUILD_opencv_cudaimgproc=ON \
                -D BUILD_opencv_cudalegacy=OFF \
                -D BUILD_opencv_cudaoptflow=OFF \
                -D BUILD_opencv_cudawarping=ON \
                -D BUILD_opencv_cudev=ON \
                -D BUILD_opencv_datasets=OFF \
                -D BUILD_opencv_dnn=ON \
                -D BUILD_opencv_dpm=OFF \
                -D BUILD_opencv_face=OFF \
                -D BUILD_opencv_features2d=OFF \
                -D BUILD_opencv_flann=OFF \
                -D BUILD_opencv_freetype=OFF \
                -D BUILD_opencv_fuzzy=OFF \
                -D BUILD_opencv_hdf=OFF \
                -D BUILD_opencv_highgui=ON \
                -D BUILD_opencv_img_hash=OFF \
                -D BUILD_opencv_imgcodecs=ON \
                -D BUILD_opencv_imgproc=ON \
                -D BUILD_opencv_line_descriptor=OFF \
                -D BUILD_opencv_ml=OFF \
                -D BUILD_opencv_objdetect=ON \
                -D BUILD_opencv_optflow=OFF \
                -D BUILD_opencv_phase_unwrapping=OFF \
                -D BUILD_opencv_photo=ON \
                -D BUILD_opencv_plot=OFF \
                -D BUILD_opencv_python2=ON \
                -D BUILD_opencv_python_bindings_generator=ON \
                -D BUILD_opencv_reg=OFF \
                -D BUILD_opencv_rgbd=OFF \
                -D BUILD_opencv_saliency=OFF \
                -D BUILD_opencv_shape=OFF \
                -D BUILD_opencv_stereo=OFF \
                -D BUILD_opencv_stitching=ON \
                -D BUILD_opencv_structured_light=OFF \
                -D BUILD_opencv_superres=OFF \
                -D BUILD_opencv_surface_matching=OFF \
                -D BUILD_opencv_text=OFF \
                -D BUILD_opencv_tracking=OFF \
                -D BUILD_opencv_ts=OFF \
                -D BUILD_opencv_video=ON \
                -D BUILD_opencv_videoio=ON \
                -D BUILD_opencv_videostab=OFF \
                -D BUILD_opencv_xfeatures2d=OFF \
                -D BUILD_opencv_ximgproc=OFF \
                -D BUILD_opencv_xobjdetect=OFF \
                -D BUILD_opencv_xphoto=OFF \
                -D WITH_OPENMP=ON \
                /tmp/opencv && \
	make -j $(nproc) && \
	make install && \
	ldconfig && \
	apt-get purge -y git && \
	apt-get clean && rm -rf /var/lib/apt/lists/*

RUN apt -y update \
&& apt install -y python3-setuptools \
&& apt install -y libgl1-mesa-glx \
&& apt install -y python-dev python3-dev \
&& apt install -y build-essential libssl-dev libffi-dev \
&& apt install -y libxml2-dev libxslt1-dev zlib1g-dev \
&& python3 -m pip install --upgrade pip \
&& pip3 install watchdog opencv-python \
&& pip3 install pybind11 \
&& pip3 install spdlog \
&& rm -rf /var/lib/apt/lists/*

COPY --from=apilayer_base /opt/apilayer-*.whl /tmp/ApiLayer/
RUN pip3 install /tmp/ApiLayer/apilayer-*.whl

COPY --from=uvdep /opt/uvdep/mqconnections/python/ /opt/
COPY --from=uvdep /app/src/mqconnections/python/* /app/
COPY --from=uvdep /app/src/moduleconfig/python/* /app/

RUN pip3 install Cython
RUN pip3 install albumentations
RUN pip3 install Pillow

ENV SPDLOGGER_VERSION 1.3.0
RUN wget -P /tmp https://github.com/gabime/spdlog/archive/v${SPDLOGGER_VERSION}.tar.gz
RUN tar -C /tmp -xzf /tmp/v${SPDLOGGER_VERSION}.tar.gz
RUN cp -R /tmp/spdlog-${SPDLOGGER_VERSION}/include/spdlog /usr/include/
RUN rm /tmp/v${SPDLOGGER_VERSION}.tar.gz

COPY src /app/
WORKDIR /app/
RUN cd /app/  && \
    rm -rf build  && \
    mkdir build  && \
    cd build  && \
    cmake ..  && \
    make  && \
    mv libclassifierv2trtlib.so /app/ 

RUN python3 setup.py build_ext --inplace

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # PRODUCTION DOCKER # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
FROM nvcr.io/nvidia/tensorrt:21.07-py3 as prod
RUN rm -rf /workspace

ENV ZSYS_SIGHANDLER=false
RUN sed -i 's|http://archive.|http://us.archive.|g' /etc/apt/sources.list \
&& apt-get -y update \
&& apt -y upgrade \
&& DEBIAN_FRONTEND=noninteractive apt-get install -y tzdata \
&& apt-get install -y --no-install-recommends \
    python3 \
    libczmq4 \
    libzmq3-dev \
    python3-pip \
    libglib2.0-dev \
    libsm-dev \
    libxrender1 \
    libxext-dev \
&& pip3 install numpy \
&& rm -rf /var/lib/apt/lists/*


RUN apt -y update \
&& apt install -y python3-setuptools \
&& apt install -y build-essential libssl-dev libffi-dev \
&& apt install -y libxml2-dev libxslt1-dev zlib1g-dev \
&& python3 -m pip install --upgrade pip \
&& pip3 install watchdog opencv-python-headless \
&& pip3 install pybind11 \
&& pip3 install spdlog \
&& pip3 install PyYAML \
&& pip3 install albumentations \
&& rm -rf /var/lib/apt/lists/*

COPY --from=dev /tmp/ApiLayer/apilayer-*.whl /tmp/ApiLayer/
RUN pip3 install /tmp/ApiLayer/apilayer-*.whl \
&& rm -r /tmp/ApiLayer

COPY --from=dev /opt/MQConnector /opt/MQConnector
COPY --from=dev /opt/MetricCollector /opt/MetricCollector
COPY --from=dev /opt/SimpleMQ /opt/SimpleMQ

COPY scripts/* /app/
COPY config/ /app/config

COPY --from=dev /opt/opencv/lib /opt/opencv/lib
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/opencv/lib/

COPY --from=dev /app/*.so /app/

# For using all files directly
# COPY --from=dev /app/* /app/

COPY --from=dev /app/run_classifierv2.py /app/

WORKDIR /app/
RUN chmod +x /app/run_classifierv2.py
RUN chmod +x /app/run_classifierv2.sh

ENTRYPOINT ["./run_classifierv2.sh"]