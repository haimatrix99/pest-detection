FROM balenalib/raspberrypi3-debian-python:3.7
# The balena base image for building apps on Raspberry Pi 3. 
# Raspbian Stretch required for piwheels support. https://downloads.raspberrypi.org/raspbian/images/raspbian-2019-04-09/

RUN [ "cross-build-start" ]

RUN mkdir app

ADD . ./app

WORKDIR /app

RUN apt-get update && apt-get upgrade -y
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN apt-get install libatlas-base-dev -y
RUN apt-get install libgsm1 libatk1.0-0 libavcodec58 libcairo2 libvpx6 libvorbisenc2 libwayland-egl1 libva-drm2 libwavpack1 libshine3 libdav1d4 libwayland-client0 libxcursor1 libopus0 libchromaprint1 libxinerama1 libpixman-1-0 libzmq5 libmp3lame0 libxcb-shm0 libgtk-3-0 libharfbuzz0b libpangocairo-1.0-0 libvdpau1 libssh-gcrypt-4 libtwolame0 libnorm1 libxi6 libxfixes3 libxcomposite1 libxcb-render0 libwayland-cursor0 libvorbisfile3 libspeex1 libxrandr2 libxkbcommon0 libtheora0 libx264-160 libaom0 libzvbi0 libogg0 libpangoft2-1.0-0 librsvg2-2 libxvidcore4 libsrt1.4-gnutls libbluray2 libvorbis0a libdrm2 libmpg123-0 libatlas3-base libxdamage1 libavformat58 libatk-bridge2.0-0 libswscale5 libsnappy1v5 libcodec2-0.9 libsodium23 libudfread0 libswresample3 libcairo-gobject2 libx265-192 libthai0 libva-x11-2 ocl-icd-libopencl1 libepoxy0 libpango-1.0-0 libavutil56 libva2 librabbitmq4 libgme0 libatspi2.0-0 libgraphite2-3 libgfortran5 libsoxr0 libpgm-5.3-0 libopenmpt0 libxrender1 libdatrie1 libgdk-pixbuf-2.0-0 libopenjp2-7 libwebpmux3 -y
RUN pip3 install --upgrade pip setuptools wheel
RUN pip3 install --extra-index-url="https://www.piwheels.org/simple" -r requirements.txt

# Cleanup
RUN rm -rf /var/lib/apt/lists/* \
    && apt-get -y autoremove

RUN [ "cross-build-end" ]  


# Expose the port
EXPOSE 5012

ENTRYPOINT [ "python3", "-u", "./main.py" ]