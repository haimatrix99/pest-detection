FROM ubuntu:18.04

RUN echo "BUILD MODULE: StreamImage"

WORKDIR /.

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-dev \
        libcurl4-openssl-dev \
        libboost-python-dev \
        libgtk2.0-dev \
        ffmpeg \
        libsm6 \
        libxext6
        

# Install Python packages
RUN pip3 install --upgrade pip
RUN pip3 install --upgrade setuptools
RUN pip3 install -r requirements.txt

# Cleanup
RUN rm -rf /var/lib/apt/lists/* \
    && apt-get -y autoremove

# Expose the port
EXPOSE 5012

ENTRYPOINT [ "python3", "-u", "./main.py" ]