# Docker-based implementation for video-on-demand streaming using RL-based edge caching
This directory implements video-on-demand streaming using RL-based edge caching. It has two major parts:

### A. Video-on-demand (VoD) service: 
We deployed VoD using two containers namely edge and remote servers. The
implementation uses Nginx as an entry point, Flask
as an application server, and HLS protocol for video playback over the Internet. The HLS playlist is pre-
processed using FFmpeg, which is an open-source tool for video and stream editing.
### B. RL-based caching:
An RL-based algorithm that can be trained using the
content popularity data. In this work, we have used
the discrete version of the soft actor-critic networks 
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
