##FROM ubuntu:latest
#FROM docker/compose
#RUN #apt update; apt install -y docker.io docker-compose git;
#RUN apk add  git bash
##RUN git clone  https://osm:PxyNtE73FkyjLJsao6BB@gitlab.cttc.es/rnikbakht/media_server.git

FROM ubuntu:18.04
RUN apt-get update
#RUN apt-get upgrade -y
RUN apt-get install -y python3 git
RUN apt-get install -y docker.io docker-compose
##RUN apt-get install -y pip
#RUN apt-get install -y curl
#RUN curl -L "https://github.com/docker/compose/releases/download/1.29.2/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
#RUN chmod +x /usr/local/bin/docker-compose
#WORKDIR /user/local/bin/
#RUN curl -L https://github.com/docker/compose/releases/download/1.29.2/docker-compose-`uname -s`-`uname -m` -o /compose/docker-compose
#RUN chmod +x /compose/docker-compose

#FROM alpine
#COPY --from=library/docker:latest /usr/local/bin/docker /usr/bin/docker
#COPY --from=docker/compose:latest /usr/local/bin/docker-compose /usr/bin/docker-compose

#RUN apk add python3
#RUN apk add py3-pip
#RUN pip3 install docker-compose
#RUN apk add  git bash

COPY ./ /media_server_uop
WORKDIR /media_server_uop
Expose 83
#RUN docker-compose
#CMD ["bash", "-c","docker-compose up"]
#CMD ["/usr/local/bin/docker-compose up"]