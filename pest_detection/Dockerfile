FROM python:3.7-slim

RUN pip install -U pip
RUN pip install -r yolov5/requirements.txt

RUN mkdir app

ADD . ./app

# Expose the port
EXPOSE 80

# Set the working directory
WORKDIR /app

# Run the flask server for the endpoints
CMD python -u app.py