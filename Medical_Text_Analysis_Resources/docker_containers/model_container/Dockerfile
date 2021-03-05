# Build an image that can do inference
# This is a Python 3 image that uses the nginx, gunicorn, flask stack
# for serving inferences in a stable way.
 
FROM ubuntu:16.04
#FROM public.ecr.aws/ubuntu/ubuntu:18.04_stable

MAINTAINER Amazon AI <sage-learner@amazon.com>
# Install python and other s dependencies
# Dependency list from http://scikit-learn.org/stable/developers/advanced_installation.html#installing-build-dependencies
RUN apt-get update && \
    apt-get -y install \
        build-essential \
        libatlas-dev \
        git \
        wget \
        curl \
        nginx \
        jq \
        libatlas3-base \
        openjdk-8-jdk-headless

RUN curl -LO http://repo.continuum.io/miniconda/Miniconda3-py38_4.9.2-Linux-x86_64.sh && \
    bash Miniconda3-py38_4.9.2-Linux-x86_64.sh -bfp miniconda3 && \
    rm Miniconda3-py38_4.9.2-Linux-x86_64.sh

ENV PATH=/miniconda3/bin:${PATH}

# Set some environment variables. PYTHONUNBUFFERED keeps Python from buffering our standard
# output stream, which means that logs can be delivered to the user quickly. PYTHONDONTWRITEBYTECODE
# keeps Python from writing the .pyc files which are unnecessary in this case. We also update
# PATH so that the train and serve programs are found when the container is invoked.

ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/program:${PATH}"

#use python 3.7.6
RUN conda install python=3.7.6

#RUN python -m pip install --no-cache -I numpy==1.18.1
#RUN python -m pip install --no-cache -I scipy==1.4.1
#RUN python -m pip install --no-cache -I sentence-transformers==0.3.5.1
#RUN python -m pip install --no-cache -I json5==0.9.1
#RUN python -m pip install --no-cache -I nltk==3.4.5
#RUN python -m pip install --no-cache -I Flask==1.1.1
#RUN python -m pip install --no-cache -I gunicorn==20.0.4

##RUN python -m pip install --no-cache -I gevent==1.4.0
##RUN python -m pip install --no-cache -I gevent==20.6.2 
#RUN python -m pip install --no-cache -I gevent==20.9.0
#RUN python -m pip install --no-cache -I pandas==0.25.3

##RUN python -m pip install --no-cache -I greenlet==0.4.15
#RUN python -m pip install --no-cache -I greenlet==0.4.17


#Set up the program in the image
COPY model_scripts /opt/program 

WORKDIR /opt/program
RUN pip install -r  model_requirements.txt

#Download needed data for hosting
RUN python download_embedding_data.py
CMD serve
