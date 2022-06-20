FROM balenalib/raspberrypi3-debian-python:3.7

RUN [ "cross-build-start" ]

RUN mkdir app

ADD . ./app

WORKDIR /app

RUN apt-get update && apt-get upgrade -y
RUN apt-get install git
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN pip3 install --upgrade pip setuptools wheel 
RUN pip3 install numpy==1.20 matplotlib opencv-python Pillow PyYAML requests scipy tqdm tensorboard pandas seaborn thop flask --extra-index-url 'https://www.piwheels.org/simple'
RUN git clone https://github.com/Kashu7100/pytorch-armv7l.git
RUN pip3 install pytorch-armv7l/torch-1.7.0a0-cp37-cp37m-linux_armv7l.whl
RUN pip3 install pytorch-armv7l/torchvision-0.8.0a0+45f960c-cp37-cp37m-linux_armv7l.whl

RUN rm -rf pytorch-armv7l

# Expose the port
EXPOSE 80

# Set the working directory

RUN [ "cross-build-end" ]

# Run the flask server for the endpoints
CMD python -u app.py