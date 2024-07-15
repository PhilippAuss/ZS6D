FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update && apt-get install -y \
	# python3-opencv \
    ca-certificates \
    python3-dev \
    git \
    wget \
    sudo \
    build-essential \
    gcc \
    lsb-release \
    curl \
    ffmpeg \
    libsm6 \
    libxext6 \
    freeglut3-dev \
    freeglut3 \
    libgl1-mesa-dev \
    libglu1-mesa-dev \
    libxext-dev \
    libxt-dev
    # ninja-build
RUN ln -sv /usr/bin/python3 /usr/bin/python

RUN wget https://bootstrap.pypa.io/get-pip.py && \
	python3 get-pip.py && \
	rm get-pip.py

# # install lsb-release and curl
# RUN apt-get update \
#  && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
#     lsb-release \
#     curl \
#  && apt-get clean \
#  && rm -rf /var/lib/apt/lists/*


# Install python dependencies
# conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia
# conda install tqdm
# conda install -c conda-forge faiss
# conda install -c conda-forge timm
# conda install matplotlib
# pip install opencv-python
# pip install git+https://github.com/lucasb-eyer/pydensecrf.git
# conda install -c anaconda scikit-learn
# pip install transforms3d
# pip install kmeans-pytorch
# pip install plyfile
# pip install trimesh
# pip install imageio
# pip install pypng
# pip install vispy==0.12.2
# pip install pyopengl==3.1.1a1
# pip install pyglet==1.2.4
# conda install pyqt
# pip install numba
# pip install jupyter
RUN python3 -m pip install \
    # faiss \
    timm \
    tqdm \
    matplotlib \
    opencv-python \
    scikit-learn \
    transforms3d \
    git+https://github.com/lucasb-eyer/pydensecrf.git \
    kmeans-pytorch \
    plyfile \
    trimesh \
    imageio \
    pypng \
    vispy==0.12.2 \
    pyopengl==3.1.1a1 \
    pyglet==1.2.4 \
    numba \
    jupyter


###
# Install ros
# add the keys
RUN sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
RUN curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -

RUN apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    ros-noetic-ros-base \
    ros-noetic-catkin \
    ros-noetic-vision-msgs \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

SHELL ["/bin/bash", "-c"]
RUN source /opt/ros/noetic/setup.bash
RUN echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc
RUN source ~/.bashrc

# install python dependencies
RUN apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    python3-rosdep \
    python3-rosinstall \
    python3-rosinstall-generator \
    python3-wstool \
    build-essential \
    python3-rosdep \
    python3-catkin-tools \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

RUN sudo rosdep init
RUN rosdep update
RUN mkdir -p /root/catkin_ws/src
RUN /bin/bash -c  '. /opt/ros/noetic/setup.bash; cd /root/catkin_ws; catkin config -DPYTHON_EXECUTABLE=/usr/bin/python3 -DPYTHON_INCLUDE_DIR=/usr/include/python3.6m -DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.6m.so; catkin build'

# clone and build message and service definitions
RUN /bin/bash -c 'cd /root/catkin_ws/src; \
                  git clone https://github.com/v4r-tuwien/object_detector_msgs.git'
RUN /bin/bash -c 'cd /root/catkin_ws/src; \
                  git clone https://gitlab.informatik.uni-bremen.de/robokudo/robokudo_msgs.git'
RUN /bin/bash -c '. /opt/ros/noetic/setup.bash; cd /root/catkin_ws; catkin build'

RUN python3 -m pip install \
    catkin_pkg \
    rospkg

RUN python3 -m pip install \
    git+https://github.com/qboticslabs/ros_numpy.git

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    mesa-utils

WORKDIR /code

COPY ros_entrypoint.sh /
RUN chmod +x /ros_entrypoint.sh
ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["python", "/code/zs6d_ros_wrapper.py"]
