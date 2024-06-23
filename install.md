# Installation of required packages on NVIDIA Jetson AGX

Tested with Ubuntu 20.04 and Jetpack 5 with Python 3.8

## PyTorch

```
wget https://nvidia.box.com/shared/static/p57jwntv436lfrd78inwl7iml6p13fzh.whl -O torch-2.1.0a0+41361538.nv23.06-cp38-cp38-linux_aarch64.whl

sudo apt-get install python3-pip libopenblas-base libopenmpi-dev libomp-dev
pip3 install 'Cython<3'
pip3 install numpy torch-2.1.0a0+41361538.nv23.06-cp38-cp38-linux_aarch64.whl
```

## Torchvision

```
sudo apt-get install libjpeg-dev zlib1g-dev libpython3-dev libopenblas-dev libavcodec-dev libavformat-dev libswscale-dev
git clone --branch v0.16.1 https://github.com/pytorch/vision torchvision   # see below for version of torchvision to download
cd torchvision
export BUILD_VERSION=0.16.1  # where 0.x.0 is the torchvision version
python3 setup.py install --user

```

## librealsense

### Option 1 (recommended)

```
git clone https://github.com/JetsonHacksNano/installLibrealsense.git
cd installLibrealsense
# Edit the buildLibrealsense.sh script replace the following line:
# /usr/bin/cmake ../ -DBUILD_EXAMPLES=true -DFORCE_LIBUVC=ON -DBUILD_WITH_CUDA="$USE_CUDA" -DCMAKE_BUILD_TYPE=release #-DBUILD_PYTHON_BINDINGS=bool:true
# with this one:
# /usr/bin/cmake ../ -DBUILD_EXAMPLES=true -DFORCE_LIBUVC=ON -DBUILD_WITH_CUDA="$USE_CUDA" -DCMAKE_BUILD_TYPE=release #-DBUILD_PYTHON_BINDINGS=bool:true -DPYTHON_EXECUTABLE=$(which python3)

./buildLibrealsense.sh -j 2

# either use the realDetect.py script as it is now (using the workaround of this sys.path.append("/usr/local/OFF/")) or change the # PYTHONPATH value in the bashrc file in the user home directory to /usr/local/OFF
```

### Option 2

```
sudo apt-get install libgl1-mesa-dev freeglut3-dev mesa-common-dev
https://github.com/IntelRealSense/librealsense.git
cd librealsense
./scripts/patch-realsense-ubuntu-L4T.sh
Navigate to the SDK's root directory.
sudo apt-get install git libssl-dev libusb-1.0-0-dev libudev-dev pkg-config libgtk-3-dev python3-dev libssl-dev libxinerama-dev libxcursor-dev libcanberra-gtk-module libcanberra-gtk3-module -y
./scripts/setup_udev_rules.sh
mkdir build && cd build
cmake .. -DBUILD_EXAMPLES=true -DCMAKE_BUILD_TYPE=release -DFORCE_RSUSB_BACKEND=false -DBUILD_WITH_CUDA=true && make -j$(($(nproc)-1)) && sudo make install
```

sometimes it might fail like this then one can fix this by editing the CMakeLists.txt file
and adding this line:
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
in line 3 (right above the following set( LRS_TARGET realsense2 ))

## Starting the script

```
python3 realDetect.py # this starts the script and writes all labels it finds the confidence it has and the distance in cm to a detections.#txt file
```

## Docker

To build the docker image, run the following command in the root directory of the project:

```
docker build -t your_image_name .
```

To run the docker image, run the following command:

```
docker run --device /dev/video0:/dev/video0 -it your_image_name -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix
```
