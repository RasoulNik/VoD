sudo docker run -it \
--name server2 \
-p 24:5000 \
-p 20022:22 \
-v /home/rasoul/ML:/home \
-v /home/rasoul/tmp:/tmp \
server:latest

