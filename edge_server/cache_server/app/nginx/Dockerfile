


FROM ubuntu:latest

ENV DEBIAN_FRONTEND=noninteractive

RUN apt update && apt install  openssh-server sudo -y
RUN apt-get install -y \
    python3 \
    python3-pip\
    git
RUN /usr/bin/python3 -m pip install --upgrade pip
RUN pip install nn_builder
RUN python3 -m pip install gym scipy matplotlib flask
RUN python3 -m pip install torch
#RUN useradd -rm -d /home/ubuntu -s /bin/bash -g root -G sudo -u 1000 test

#RUN  echo 'test:test' | chpasswd
#RUN  echo 'ubuntu:ubuntu' | chpasswd
RUN useradd -ms /bin/bash rasoul
RUN  echo 'rasoul:rasoul' | chpasswd
RUN sudo usermod -aG sudo rasoul
RUN chown -R rasoul:sudo /home/rasoul
WORKDIR /home/rasoul
RUN service ssh start


#RUN python3 -m pip --no-cache-dir install --upgrade \
#    "pip<20.3" \
#    setuptools \
#COPY bashrc /etc/bash.bashrc
#RUN chmod a+rwx /etc/bash.bashrc
EXPOSE 22
EXPOSE 8080
EXPOSE  443
EXPOSE 1935
EXPOSE 5000


CMD ["/usr/sbin/sshd","-D"]


