docker run -it \
--name pycharm_env_ml \
-p 422:22 \
-p 480:80 \
-p 48080:8080 \
-p 41935:1935 \
-p 4443:443 \
-p 45000:5000 \
-p 45001:5001 \
-p 45002:5002 \
-v /home/rasoul/PycharmProjects/DRLcache/DRLcache/server/:/home \
ubuntu_ml_local