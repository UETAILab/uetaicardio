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
        unzip \
        gpg-agent \
        sudo \
        locales && \
    locale-gen en_US.UTF-8 && \
    locale-gen vi_VN.UTF-8 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

#RUN apt-get update && \
#    apt-get install -y --no-install-recommends --fix-missing \
#        python3 \
#        python3-dev \
#        python3-pip \
#        python3-setuptools && \
#    pip3 install --upgrade pip && \
#    update-alternatives --install \
#    /usr/bin/python python $(which python3) 10 && \
#    apt-get clean && rm -rf /var/lib/apt/lists/*

# Set the locale
RUN sed -i '/en_US.UTF-8/s/^# //g' /etc/locale.gen && \
    locale-gen
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

# Port the model runtime service will listen on.
EXPOSE 8880

# Copy and install models.
# COPY model /model/
RUN pip install --upgrade pip
RUN pip install flask
RUN pip install gunicorn
RUN pip install opencv-contrib-python==4.2.0.32
RUN pip install pandas==1.0.1
RUN pip install pillow==7.1.2
RUN pip install pydicom==1.3.0
RUN pip install pykalman==0.9.5
RUN pip install pytz==2019.3
RUN pip install pywavelets==1.1.1
RUN pip install receptivefield==0.5.0
RUN pip install albumentations==0.4.3
RUN pip install easydict==1.9

# Copy source to server.
COPY . /
RUN chmod +x /deploy.sh
# One time initialization of the models.
WORKDIR /
ENV PYTHONPATH "${PYTHONPATH}:/"
ENTRYPOINT ["/deploy.sh"]