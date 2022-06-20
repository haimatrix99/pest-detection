FROM python:3.7-slim

RUN mkdir app

ADD . ./app

WORKDIR /app

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN pip3 install -U pip
RUN pip3 install -r yolov5/requirements.txt

# Expose the port
EXPOSE 80

# Run the flask server for the endpoints
CMD python3 -u app.py