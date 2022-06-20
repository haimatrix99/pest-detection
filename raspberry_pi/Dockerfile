FROM balenalib/raspberrypi3-debian-python:3.7
# The balena base image for building apps on Raspberry Pi 3. 
# Raspbian Stretch required for piwheels support. https://downloads.raspberrypi.org/raspbian/images/raspbian-2019-04-09/

RUN [ "cross-build-start" ]

RUN mkdir app

ADD . ./app

WORKDIR /app

RUN apt-get update && apt-get upgrade -y
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN pip3 install --upgrade pip setuptools wheel
RUN pip3 install --extra-index-url="https://www.piwheels.org/simple" --no-deps -r requirements.txt

# Cleanup
RUN rm -rf /var/lib/apt/lists/* \
    && apt-get -y autoremove

RUN [ "cross-build-end" ]  


# Expose the port
EXPOSE 5012

ENTRYPOINT [ "python3", "-u", "./main.py" ]