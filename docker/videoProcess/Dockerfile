FROM ubuntu:18.04

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Shanghai
# 这个地方因为中文名的原因，导致可能在创建文件时出错，古加上
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

COPY ./requirements.txt /tmp/requirements.txt
COPY example.py /usr/local/src/project/main.py

RUN set -x \
    && apt update \
    && apt install -y python3 python3-pip ffmpeg libgl-dev libglib2.0-dev tzdata \
    && ln -s /usr/bin/python3 /usr/bin/python \
    && ln -s /usr/bin/pip3 /usr/bin/pip \
    && pip install pip -U \
    && pip install -r /tmp/requirements.txt \
    && cp /usr/share/zoneinfo/Asia/Shanghai /etc/localtime \
    && echo "Aisa/Shanghai" > etc/timezone \
    && mkdir /usr/local/src/project/output-docker \
    && mkdir /usr/local/src/project/log-docker \
    && rm -rf /tmp/* ~/.cache/* /var/tmp/* /var/cache/apk/*


VOLUME /usr/local/src/project/output-docker
VOLUME /usr/local/src/project/log-docker

WORKDIR /usr/local/src/project
EXPOSE 9933
#ENTRYPOINT ["/sbin/tini", "--"]
CMD ["python", "main.py"]
