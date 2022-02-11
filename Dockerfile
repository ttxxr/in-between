# NAME NSM MLP_EA
# Time 2021/07/29
# VERSION 0.1.1
FROM mirrors.tencent.com/dct/anim-torch-gpu:1.8.1-cuda10.1
LABEL maintainer="alantxren@tencent.com"
ENV PATH="/root/bin:${PATH}"
ENV PATH="/opt/miniconda3/bin:${PATH}"
ENV PATH="/usr/local/nvidia/bin:${PATH}"
ENV PYTHONPATH="/home/gameai/animation"
ENV BLADE_AUTO_UPGRADE="no"
ADD . /home/gameai/my_nsm
WORKDIR /home/gameai/my_nsm