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


FROM ubuntu:latest

ENV DEBIAN_FRONTEND=noninteractive
RUN apt update && apt install  openssh-server sudo -y
RUN useradd -ms /bin/bash rasoul
RUN  echo 'rasoul:rasoul' | chpasswd
RUN sudo usermod -aG sudo rasoul
RUN chown -R rasoul:sudo /home/rasoul
WORKDIR /home/rasoul
RUN service ssh start
#RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> \
#/etc/sudoers
#USER rasoul
#RUN sudo apt-get updat
RUN apt-get install -y \
    python3 \
    python3-pip\
    git
RUN /usr/bin/python3 -m pip install --upgrade pip

RUN python3 -m pip install nn_builder
RUN python3 -m pip install  gym scipy matplotlib flask
RUN python3 -m pip install torch



EXPOSE 5000
EXPOSE 22
# download the source code
ARG CACHEBUST=1
#RUN git clone  https://testbed:oAon-z5sFxMbcXSAgKbr@gitlab.cttc.es/rnikbakht/DRLcache.git
#WORKDIR DRLcache/
#RUN git checkout master
COPY . .
#RUN python3 RL_cache_web_server.py

#CMD ["bash"]
#CMD ["bash", "-c","python3 RL_cache_web_server.py"]
#CMD ["bash", "-c", "source /etc/bash.bashrc && jupyter notebook --notebook-dir=/tf --ip 0.0.0.0 --no-browser --allow-root"]