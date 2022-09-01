# Docker-based implementation for video-on-demand streaming using RL-based edge caching
This directory implements video-on-demand streaming using RL-based edge caching. It has three major parts:

### A. Video-on-demand (VoD) service: 
We deployed VoD using two containers namely edge and remote servers. The
implementation uses Nginx as an entry point, Flask
as an application server, and HLS protocol for video playback over the Internet. The HLS playlist is pre-
processed using FFmpeg, which is an open-source tool for video and stream editing.
### B. RL-based caching:
An RL-based algorithm that can be trained using the
content popularity data. In this work, we have used
the discrete version of the soft actor-critic networks 
### Deployment: 
The current deployment use docker-compose for orchestration. For more detail about using the current repository, please refer to section Deployment. 
## A. Video-on-demand service
The HLS protocol consists of a playlist and video segments. The user client asks for a video playlist (video.m3u8)
though 5G link. If the content is cached (the playlist and video segments are available in the cache storage)
• The Nginx in the edge server receives the request and
serves the user. In addition, the Nginx mirrors the
user request to the ML algorithm.
If the content is not cached in the edge-server:
• The edge server re-routes the user request to the
remote server.
• Nginx in the remote server directly serves the users.
• In the background, the application server that in-
includes the RL-based caching algorithm updates the
cached content on the edge-server (download the
HLS playlist and video segments from the remote
server)
## B. RL-based caching:
Let's assume we have a total of M contents of which C contents can be kept in a cache storage. Formulating the cache server as an RL algorithm, we define the state and action space of the RL, the logic of the edge caching, observation, and the reward function as

### State:
	- Cached content IDs
	- The Total number of requests for each cached content in the given time frame or window.
  
### Action space: 
the DRL agent can either replace the selected cached content with the currently requested content or keep the cached contents the same.

### The logic of the edge caching:
	- If the content is cached, return the content (or its ID) and update the state.
	- If the content is not cached, then ask for the content from the server, and take an action.
### Observation: 
It is a request for a file index. In this project, we assume that the observations follow the Zipf distribution. It means that at any given time the content popularity has a specific structure that is imposed by Zipf distribution and its parameter. As an example, for the Zipf parameter of 1.25, a small percentage of the contents (5%) accounts for the majority of the traffic (80%). The proposed framework can be extended to real-world data once it is in the production stage.

## Deployment:
The current deployment is based on ubuntu 20..04, and it assumes you have docker and docker-compose installed on your machine.
### Steps:
- Create a network named "external-example" using the command:
```
docker network create external-example
```
- Got to the *remote_server* directory and run:
```
docker-compose build
docker-compose up
```
This will bring up the remote server.
- Go to the *edge_server* directory in the project root and run: 
```
docker-compose build
docker-compose up
```
This step might take 5-10 minutes, depending on your internet connection speed. It downloads all the required python libraries for running RL-based caching and builds a flask server for hosting it. Then, it launches the edge server and connects it to the remote server. 
### Video playback 
For the testing purpose two videos (big bunny and mountain) are preprocessed using FFmpeg and located in the edge (big bunny) and remote (mountain) servers. We assume that the user asks for a content ID, ranging from 1 to 10000. If the content is cached in the edge server (if the content ID is available in the current state of the RL-based caching), the edge server serves the user with the big bunny video. Otherwise, the edge server redirects the user to the remote server, where the user is served with mountain video. 

Open the VLC and insert the following link:

```
http://localhost/1.mp4.m3u8
```
Or if you have ffplay installed (it comes with FFmpeg) type:
```
ffplay http://localhost/1.mp4.m3u8
```
It asks for content ID 1. Since the training is done using Zipf distribution, the contents with lower IDs are more popular than contents with higher IDs (and content number 1 is the most popular one). As a result, you will see the big bunny video played from the edge server. If you ask for another content, say 700.mp4.m3u8, it is highly likely that this content is very unpopular and is not cached, so you will see the mountain video played from the remote server. 

Interestingly, if you ask again for the content 700.mp4.m3u8, you will get the big bunny from the edge server. Because the RL-based caching continuously listen to the user request and adapt itself. By asking gain for the same content, we make that content popular and impose a different popularity distribution, and as a resault, the video should be cached.   




