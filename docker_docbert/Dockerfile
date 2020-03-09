FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04


MAINTAINER arne <arnedefauw@gmail.com>

# Install some basic utilities
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
 && rm -rf /var/lib/apt/lists/*


ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

# Install miniconda to /miniconda
RUN curl -LO http://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN bash Miniconda3-latest-Linux-x86_64.sh -p /miniconda -b
RUN rm Miniconda3-latest-Linux-x86_64.sh
ENV PATH=/miniconda/bin:${PATH}
RUN conda update -y conda

RUN conda install python=3.7.3


RUN conda install pytorch torchvision cudatoolkit=10.0 -c pytorch


# Python packages from conda
RUN conda install -y \
    scikit-learn \
    scikit-image \
    flask \
    pillow \
    seaborn \
&& conda clean -ya

RUN conda install --name base scikit-learn=0.20.0

#RUN conda install faiss-gpu -c pytorch \
#&& conda clean -ya

#Some python packages from pip
RUN pip install \
        gensim \
        ipython \
        jupyter \
        nltk \
	beautifulsoup4 \
	lxml \
	pytorch-nlp \
	mosestokenizer \
	langdetect \
	pytorch-pretrained-bert \
	openpyxl \
	Pillow \
	visdom \
	dominate \
	pdf2image

#Install Cython
RUN apt-get update

RUN apt-get -y install --reinstall build-essential

RUN apt-get -y install gcc

RUN pip install Cython


#install spacy
RUN pip install -U spacy

RUN apt-get update && apt-get install -y \
    wget

#install apex, for fp16 (float point 16, for speed-up and memory)  training
#RUN git clone https://github.com/NVIDIA/apex /root/apex
#WORKDIR /root/apex
#RUN pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" .

RUN conda install -c conda-forge poppler


#install tesseract4.0
RUN apt-get install -y tesseract-ocr && \
    apt-get install -y tesseract-ocr-all && \
    apt-get install -y libtesseract-dev

RUN pip install pytesseract

RUN pip install pyyaml


RUN apt-get update
RUN apt-get -y install python3-enchant
RUN pip install pyenchant

# Install OpenJDK-8
RUN apt-get update && \
    apt-get install -y openjdk-8-jdk && \
    apt-get install -y ant && \
    apt-get clean;

# Fix certificate issues
RUN apt-get update && \
    apt-get install ca-certificates-java && \
    apt-get clean && \
    update-ca-certificates -f;

# Setup JAVA_HOME -- useful for docker commandline
ENV JAVA_HOME /usr/lib/jvm/java-8-openjdk-amd64/
RUN export JAVA_HOME

RUN pip install pandas==1.0.1

RUN pip install tensorflow
RUN pip install keras

RUN pip install seqeval

RUN pip install transformers
RUN pip install pytorch-transformers

RUN pip install torchtext
RUN pip install ConfigArgParse


EXPOSE 8888

VOLUME /notebook
WORKDIR /notebook

CMD ["jupyter", "notebook", "--no-browser", "--allow-root", "--ip='0.0.0.0'"]
