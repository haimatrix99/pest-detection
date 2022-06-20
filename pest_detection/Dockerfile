FROM balenalib/raspberrypi3-debian-python:3.7

RUN [ "cross-build-start" ]

RUN mkdir app

ADD . ./app

WORKDIR /app

RUN apt update && apt install -y libjpeg62-turbo libopenjp2-7 libtiff5 libatlas-base-dev libxcb1
RUN apt-get install libjpeg-dev libopenblas-dev libopenmpi-dev libomp-dev

RUN pip3 install --upgrade pip
RUN pip3 install --upgrade setuptools
RUN pip3 install -r ./yolov5/arm32v7.requirements.txt
RUN pip3 install flask==1.1.2 --index-url 'https://www.piwheels.org/simple'
RUN pip install numpy==1.16--extra-index-url 'https://www.piwheels.org/simple' --no-deps
RUN wget https://raw.githubusercontent.com/Kashu7100/pytorch-armv7l/main/torch-1.7.0a0-cp37-cp37m-linux_armv7l.whl
RUN wget https://raw.githubusercontent.com/Kashu7100/pytorch-armv7l/main/torchvision-0.8.0a0%2B45f960c-cp37-cp37m-linux_armv7l.whl

RUN pip3 install torch-1.7.0a0-cp37-cp37m-linux_armv7l.whl
RUN pip3 install torchvision-0.8.0a0%2B45f960c-cp37-cp37m-linux_armv7l.whl 
RUN rm torch-1.7.0a0-cp37-cp37m-linux_armv7l.whl && rm torchvision-0.8.0a0%2B45f960c-cp37-cp37m-linux_armv7l.whl 

# Expose the port
EXPOSE 80

# Set the working directory

RUN [ "cross-build-end" ]

# Run the flask server for the endpoints
CMD python -u app.py