#! /bin/bash
# install all dependencies
sudo apt-get install cmake libsuitesparse-dev libeigen3-dev qtdeclarative5-dev libqglviewer-dev qt5-qmake libglew-dev libgtk2.0-dev pkg-config libgstreamer1.0-0 gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-libav gstreamer1.0-doc gstreamer1.0-tools


#comment this if use cuda8.0
ln -s /usr/local/cuda-9.0/targets/aarch64-linux/lib/libnppif.so.9.0 /usr/local/lib/libopencv_dep_nppif.so
ln -s /usr/local/cuda-9.0/targets/aarch64-linux/lib/libnppig.so.9.0 /usr/local/lib/libopencv_dep_nppig.so
ln -s /usr/local/cuda-9.0/targets/aarch64-linux/lib/libnppim.so.9.0 /usr/local/lib/libopencv_dep_nppim.so
ln -s /usr/local/cuda-9.0/targets/aarch64-linux/lib/libnppial.so.9.0 /usr/local/lib/libopencv_dep_nppial.so
ln -s /usr/local/cuda-9.0/targets/aarch64-linux/lib/libnppicc.so.9.0 /usr/local/lib/libopencv_dep_nppicc.so
ln -s /usr/local/cuda-9.0/targets/aarch64-linux/lib/libnppist.so.9.0 /usr/local/lib/libopencv_dep_nppist.so
ln -s /usr/local/cuda-9.0/targets/aarch64-linux/lib/libnppitc.so.9.0 /usr/local/lib/libopencv_dep_nppitc.so
ln -s /usr/local/cuda-9.0/targets/aarch64-linux/lib/libnppicom.so.9.0 /usr/local/lib/libopencv_dep_nppicom.so
ln -s /usr/local/cuda-9.0/targets/aarch64-linux/lib/libnppidei.so.9.0 /usr/local/lib/libopencv_dep_nppidei.so
ln -s /usr/local/cuda-9.0/targets/aarch64-linux/lib/libnppisu.so.9.0 /usr/local/lib/libopencv_dep_nppisu.so

# comment this if build on amd64
sudo ln -sf /usr/lib/aarch64-linux-gnu/tegra/libGL.so /usr/lib/aarch64-linux-gnu/libGL.so


# build opencv
mkdir opencv
tar xvzf opencv-2.4.13.tar.gz -C opencv/
cd opencv
mkdir build
cd build
cmake -DWITH_CUDA=ON -DCMAKE_BUILD_TYPE=RELEASE -DCUDA_ARCH_BIN=6.2 -DCUDA_ARCH_PTX=6.2 -BUILD_SHARED_LIB=ON ..
make -j5
cd ../..

# build g2o
tar xvzf SimpleVisualOdometry.tar.gz
cd SimpleVisualOdometry/g2o/
mkdir build
cd build
cmake -DG2O_BUILD_EXAMPLES=OFF ..
make -j5
sudo make install
cd ../..

# build Pangolin
cd Pangolin
mkdir build
cd build
cmake ..
make -j5
sudo make install
cd ../..

# build Sophus
cd Sophus
mkdir build
cd build
cmake ..
make -j5
sudo make install
cd ../..

# build simple slam
cd simple_visual_odometry
mkdir build
cd build
cmake -DCUDA_USE_STATIC_CUDA_RUNTIME=OFF ..
make -j5
cd ../..


