#FROM tiangolo/nginx-rtmp
#FROM murderousone/nginx-ffmpeg-rtmp:debian-latest
#RUN apk add nginx-extras
#RUN apk bash
FROM nginx:1.19-alpine
#FROM nginx
#RUN apk add bash
RUN mkdir -p /videos
WORKDIR /videos
#COPY . /videos
COPY nginx.conf /etc/nginx/conf.d/default.conf

#A
#COPY nginx-default.conf.template /etc/nginx/conf.d/default.conf.template
#
#COPY docker-entrypoint.sh .
#RUN chmod +x docker-entrypoint.sh
#ENTRYPOINT ["/docker-entrypoint.sh"]
#CMD ["nginx", "-g", "daemon off;"]

## Adding variable to nginx config
#RUN mkdir -p /etc/nginx/templates
#COPY nginx.conf.template /etc/nginx/templates/default.conf.template
#B




#COPY ./ /videos
#RUN /etc/nginx/nginx -t
#CMD ["/etc/nginx/sbin/nginx"]

#FROM ubuntu:18.04
#RUN apt update
#RUN apt install build-essential git -y
#RUN apt install libpcre3-dev libssl-dev zlib1g-dev -y
#RUN git clone https://github.com/arut/nginx-rtmp-module.git
#RUN git clone https://github.com/nginx/nginx.git
#RUN git clone https://github.com/RasoulNik/nginx-let-module.git
#WORKDIR nginx
#RUN ./auto/configure --add-module=../nginx-rtmp-module --add-module=../nginx-let-module --with-stream
#RUN make
#RUN make install
#
##RUN apt-get install aptitude -y
##RUN aptitude install nginx-mod-stream.x86_64 -y
##RUN apt-get install libnginx-mod-stream -y
#RUN apt-get install  nginx-common wget -y
#RUN wget http://archive.ubuntu.com/ubuntu/pool/main/n/nginx/libnginx-mod-stream_1.14.0-0ubuntu1.10_amd64.deb
#RUN dpkg -i libnginx-mod-stream_1.14.0-0ubuntu1.10_amd64.deb
#
#RUN mkdir -p /videos
##COPY ./nginx.conf /videos
## Create HLS playlist
##RUN apt-get install ffmpeg -y
##RUN git clone https://github.com/RasoulNik/HLS-Stream-Creator.git
##WORKDIR HLS-Stream-Creator
##COPY input.mp4 .
##RUN ./HLS-Stream-Creator.sh -i input.mp4 -s 10 #-b 512,1024,2048,4096
##chown -R root:roor /videos
#COPY nginx.conf /usr/local/nginx/conf
#RUN mkdir -p /opt/data/hls
#RUN /usr/local/nginx/sbin/nginx -t
#CMD ["/usr/local/nginx/sbin/nginx"]