FROM balenalib/raspberrypi3-debian-python:3.7

RUN [ "cross-build-start" ]

RUN mkdir app

ADD . ./app

WORKDIR /app

RUN apt-get update && apt-get upgrade -y
RUN apt-get install git wget
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN apt-get install libatlas-base-dev -y
RUN apt-get install libgsm1 libatk1.0-0 libavcodec58 libcairo2 libvpx6 libvorbisenc2 libwayland-egl1 libva-drm2 libwavpack1 libshine3 libdav1d4 libwayland-client0 libxcursor1 libopus0 libchromaprint1 libxinerama1 libpixman-1-0 libzmq5 libmp3lame0 libxcb-shm0 libgtk-3-0 libharfbuzz0b libpangocairo-1.0-0 libvdpau1 libssh-gcrypt-4 libtwolame0 libnorm1 libxi6 libxfixes3 libxcomposite1 libxcb-render0 libwayland-cursor0 libvorbisfile3 libspeex1 libxrandr2 libxkbcommon0 libtheora0 libx264-160 libaom0 libzvbi0 libogg0 libpangoft2-1.0-0 librsvg2-2 libxvidcore4 libsrt1.4-gnutls libbluray2 libvorbis0a libdrm2 libmpg123-0 libatlas3-base libxdamage1 libavformat58 libatk-bridge2.0-0 libswscale5 libsnappy1v5 libcodec2-0.9 libsodium23 libudfread0 libswresample3 libcairo-gobject2 libx265-192 libthai0 libva-x11-2 ocl-icd-libopencl1 libepoxy0 libpango-1.0-0 libavutil56 libva2 librabbitmq4 libgme0 libatspi2.0-0 libgraphite2-3 libgfortran5 libsoxr0 libpgm-5.3-0 libopenmpt0 libxrender1 libdatrie1 libgdk-pixbuf-2.0-0 libopenjp2-7 libwebpmux3 -y
RUN apt-get install libjpeg-dev libopenblas-dev libopenmpi-dev libomp-dev -y
RUN pip3 install --upgrade pip setuptools wheel 
RUN pip3 install numpy==1.19.5 matplotlib opencv-python Pillow PyYAML requests scipy tqdm tensorboard pandas seaborn flask --extra-index-url 'https://www.piwheels.org/simple'
RUN git clone https://github.com/Kashu7100/pytorch-armv7l.git
RUN pip3 install pytorch-armv7l/torch-1.7.0a0-cp37-cp37m-linux_armv7l.whl
RUN pip3 install pytorch-armv7l/torchvision-0.8.0a0+45f960c-cp37-cp37m-linux_armv7l.whl
RUN pip3 install thop

RUN wget https://github.com/lhelontra/tensorflow-on-arm/releases/download/v2.3.0/tensorflow-2.3.0-cp37-none-linux_armv7l.whl
RUN pip3 install tensorflow-2.3.0-cp37-none-linux_armv7l.whl
RUN rm tensorflow-2.3.0-cp37-none-linux_armv7l.whl
RUN rm -rf pytorch-armv7l

# Expose the port
EXPOSE 80

# Set the working directory

RUN [ "cross-build-end" ]

# Run the flask server for the endpoints
CMD python -u app.py