# syntax = docker/dockerfile:experimental

FROM  pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime

RUN apt-get update && \
    apt-get install -y --no-install-recommends --fix-missing \
        build-essential \
        apt-utils \
        apt-transport-https \
        ca-certificates \
        software-properties-common \
        pkg-config \
        curl \
        wget \
        libsm6 libxrender1 libfontconfig1 \
        unzip \
        gpg-agent \
        sudo \
        locales && \
    locale-gen en_US.UTF-8 && \
    locale-gen vi_VN.UTF-8 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Set the locale
RUN sed -i '/en_US.UTF-8/s/^# //g' /etc/locale.gen && \
    locale-gen
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

# Copy and install models.
RUN --mount=type=cache,mode=0755,target=/root/.cache/pip
RUN pip install --upgrade pip
RUN pip install --ignore-installed albumentations==1.1.0
RUN pip install --ignore-installed easydict==1.9
RUN pip install --ignore-installed Flask==2.1.0
RUN pip install --ignore-installed loguru==0.5.3
RUN pip install --ignore-installed matplotlib==3.3.4
RUN pip install --ignore-installed numpy==1.19.5
RUN pip install --ignore-installed opencv-contrib-python-headless==4.4.0.46
RUN pip install --ignore-installed Pillow==9.0.0
RUN pip install --ignore-installed pydicom==2.2.2
RUN pip install --ignore-installed pykalman==0.9.5
RUN pip install --ignore-installed PyYAML==6.0
RUN pip install --ignore-installed redis==4.1.1
RUN pip install --ignore-installed segmentation==0.2.2
RUN pip install --ignore-installed terminaltables==3.1.10
RUN pip install --ignore-installed torch==1.10.1
RUN pip install --ignore-installed torchsummary==1.5.1
RUN pip install --ignore-installed torchvision==0.11.2
RUN pip install --ignore-installed tqdm==4.62.3
RUN pip install --ignore-installed Werkzeug==0.16.1
RUN pip install --ignore-installed gdown
RUN pip install --ignore-installed Flask
RUN pip install --ignore-installed Werkzeug

# Port the model runtime service will listen on.
EXPOSE 3334

USER root
SHELL ["/bin/bash", "-c"]

COPY ckpts /ckpts
RUN cd /ckpts && sh download.sh

# prevent cache the following cmd
ARG CACHEBUST=1
RUN echo "$CACHEBUST"

# copy src code
COPY . /app
WORKDIR /app

RUN cd ckpts && cp /ckpts/* .

# testing
RUN chmod +x scripts/test.sh
#CMD ["./scripts/test.sh"]

# serving
RUN chmod +x scripts/deploy.sh
CMD ["./scripts/deploy.sh"]
