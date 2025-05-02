# Using the latest PyTorch image with CUDA support
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

RUN mkdir /job
WORKDIR /job
VOLUME ["/job/data", "/job/src", "/job/work", "/job/output"]

# Install dependencies using requirements.txt
COPY requirements.txt /job/
RUN pip install -r requirements.txt
