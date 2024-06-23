# Basis-Image
FROM nvcr.io/nvidia/l4t-base:r32.4.4

# Umgebungsvariablen setzen
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility,video

# Abhängigkeiten installieren
RUN apt-get update && apt-get install -y \
    python3-pip \
    v4l-utils \
    && rm -rf /var/lib/apt/lists/*

# Bibliotheken für Python installieren
RUN pip3 install numpy opencv-python-headless

# Librealsense für NVIDIA Jetson AGX Orin installieren
RUN apt-get update && apt-get install -y git cmake python3-pip libusb-1.0-0-dev \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/JetsonHacksNano/installLibrealsense.git \
    && cd installLibrealsense \
    && sed -i 's#/usr/bin/cmake ../ -DBUILD_EXAMPLES=true -DFORCE_LIBUVC=ON -DBUILD_WITH_CUDA="$USE_CUDA" -DCMAKE_BUILD_TYPE=release -DBUILD_PYTHON_BINDINGS=bool:true#/usr/bin/cmake ../ -DBUILD_EXAMPLES=true -DFORCE_LIBUVC=ON -DBUILD_WITH_CUDA="$USE_CUDA" -DCMAKE_BUILD_TYPE=release -DBUILD_PYTHON_BINDINGS=bool:true -DPYTHON_EXECUTABLE=$(which python3)#' buildLibrealsense.sh \
    && ./buildLibrealsense.sh -j 2

RUN cd ..

# Quellcode kopieren
COPY ./realDetect.py /app/realDetect.py

# Arbeitsverzeichnis setzen
WORKDIR /app

# Ausführungsbefehl
CMD ["python3", "realDetect.py"]