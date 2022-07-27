# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
#
# THIS IS A GENERATED DOCKERFILE.
#
# This file was assembled from multiple pieces, whose use is documented
# throughout. Please refer to the TensorFlow dockerfiles documentation
# for more information.

ARG UBUNTU_VERSION=20.04

FROM ubuntu:${UBUNTU_VERSION} as base

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y curl

# See http://bugs.python.org/issue19846
#ENV LANG C.UTF-8

RUN apt-get update && apt-get install -y \
    python3.8 \
    python3-pip\
    git

RUN python3 -m pip --no-cache-dir install --upgrade \
    pip \
    setuptools

# Some TF tools expect a "python" binary
RUN ln -s $(which python3) /usr/local/bin/python

# Options:
#   tensorflow
#   tensorflow-gpu
#   tf-nightly
#   tf-nightly-gpu
# Set --build-arg TF_PACKAGE_VERSION=1.11.0rc0 to install a specific version.
#
# create avirtual environment
#RUN python3 -m pip install virtualenv
#RUN virtualenv cacheserver
RUN #source cacheserver/bin/activate
RUN pip install nn_builder
RUN pip install torch gym scipy matplotlib flask numpy scipy

#ARG TF_PACKAGE=tensorflow
#ARG TF_PACKAGE_VERSION=
#RUN python3 -m pip install --no-cache-dir ${TF_PACKAGE}${TF_PACKAGE_VERSION:+==${TF_PACKAGE_VERSION}}
# Install pytorch and other required packages
#RUN apt-get install nn_builder

#COPY bashrc /etc/bash.bashrc
#RUN chmod a+rwx /etc/bash.bashrc

#RUN python3 -m pip install --no-cache-dir jupyter matplotlib
# Pin ipykernel and nbformat; see https://github.com/ipython/ipykernel/issues/422
# Pin jedi; see https://github.com/ipython/ipython/issues/12740
#RUN python3 -m pip install --no-cache-dir jupyter_http_over_ws ipykernel==5.1.1 nbformat==4.4.0 jedi==0.17.2
#RUN jupyter serverextension enable --py jupyter_http_over_ws

#RUN mkdir -p /tf/tensorflow-tutorials && chmod -R a+rwx /tf/
#RUN mkdir /.local && chmod a+rwx /.local
#RUN apt-get update && apt-get install -y --no-install-recommends wget git
#WORKDIR /tf/tensorflow-tutorials
#RUN wget https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/keras/classification.ipynb
#RUN wget https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/keras/overfit_and_underfit.ipynb
#RUN wget https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/keras/regression.ipynb
#RUN wget https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/keras/save_and_load.ipynb
#RUN wget https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/keras/text_classification.ipynb
#RUN wget https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/keras/text_classification_with_hub.ipynb
#COPY readme-for-jupyter.md README.md
#RUN apt-get autoremove -y && apt-get remove -y wget
#WORKDIR /tf


#RUN python3 -m ipykernel.kernelspec
# add ssh server
RUN apt update && apt install  openssh-server sudo -y
RUN useradd -rm -d /home/ubuntu -s /bin/bash -g root -G sudo -u 1000 rasoul
RUN  echo 'rasoul:rasoul' | chpasswd
RUN service ssh start
#RUN service ssh status
EXPOSE 5000
EXPOSE 22
# download the source code
ARG CACHEBUST=1
RUN git clone  https://testbed:oAon-z5sFxMbcXSAgKbr@gitlab.cttc.es/rnikbakht/DRLcache.git
WORKDIR DRLcache/
RUN git checkout master
RUN ls

#WORKDIR tests
#RUN pip install jupyterlab

#RUN python3 RL_cache_web_server.py
#CMD ["bash"]
CMD ["bash","-c","source /etc/bash.bashrc && python3 RL_cache_web_server.py"]
#CMD ["bash","-c","service ssh start","service ssh status", "source /etc/bash.bashrc && python3 RL_cache_web_server.py"]

#CMD ["bash", "-c", "source /etc/bash.bashrc && jupyter lab  --port=5000 --ip 0.0.0.0 --no-browser --allow-root"]
