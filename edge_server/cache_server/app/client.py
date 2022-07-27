
import json
import os

import httpx
import time

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import pandas as pd
from scipy.io import savemat,loadmat
class cache_client():
    def __init__(self, cache_dimension=100, total_file_number=1000, zipf_param=0.3,server_IP="http://192.168.1.103",server_port="5000"):
        self.req_batch_num = 1000
        self.cache_dimension = cache_dimension
        self.reward_for_achieving_goal = 1.0
        self.step_reward_for_not_achieving_goal = 0
        self.total_file_number = total_file_number
        self.zipf_param = zipf_param
        self.server_ip = server_IP
        self.serve_port = server_port
        self.server_add = server_IP+":"+self.serve_port
        self.change_popular_contet_index = False
    def send_req_batch(self):
        with httpx.Client() as client:
            t1=time.time()
            cache_hit = []
            cache_hit_LFU = []
            elapsed_time =[]
            if self.change_popular_contet_index:
                bias = np.random.randint(0,self.total_file_number)
            else:
                bias = 0
            for i in range(self.req_batch_num):
                self.observation = self.zipf_truncated(self.zipf_param,self.total_file_number,1)
                self.observation = (self.observation+bias) % self.total_file_number

                r = client.get(self.server_add+"/cache_status/"+str(self.observation[0]))
                reply=json.loads(r.content.decode("utf-8"))
                print(self.observation, reply["Cache_status"])
                cache_hit.append(float(reply["Cache_status"]))
                cache_hit_LFU.append(float(reply["Cache_status_LFU"]))
                elapsed_time.append((r.elapsed.total_seconds()))
                # print(r.text)
            cache_hit_ratio = np.sum(cache_hit)/self.req_batch_num
            self.cache_hit = cache_hit
            self.cache_hit_LFU = cache_hit_LFU
            self.elapsed_time = elapsed_time
            t2= time.time()
            print("It took "+str(t2-t1)+"s to process "+ str(self.req_batch_num)+" requests")
            print("The cache hit ratio is "+str(cache_hit_ratio))
    def zipf_truncated(self,a,M,N):
        # Number of files
        # sampling size
        # distribution parameters
        # N = 7
        x = np.arange(1, M + 1)
        # a = 1.1
        weights = x ** (-a)
        weights /= weights.sum()
        bounded_zipf = stats.rv_discrete(name='bounded_zipf', values=(x, weights))
        samples = bounded_zipf.rvs(size=N)

        return samples
    def moving_averag(self,x):
        window_length = 100
        zero_pad = np.array([0.0]*window_length)
        x1 = np.append(zero_pad,np.array(x))
        df = pd.DataFrame(data=x1)
        y = df.rolling(window =window_length).mean()
        # plt.xlabel("file request")
        # plt.ylabel("Cache hit ratio")
        # plt.show()
        return y
    def save_result(self):
        pass

        # sample = bounded_zipf.rvs(size=10000)
        # plt.hist(sample, bins=np.arange(1, N + 2))
        # plt.show()

if __name__ == '__main__':
    # scenario one: constant traffic popularityç
    # cc_constant = cache_client()
    cc_constant = cache_client(server_IP="http://localhost", server_port="580")
    #cc_constant = cache_client(server_IP="http://10.10.100.24",server_port="5000")
    # cc_constant = cache_client(server_IP="http://172.16.100.20", server_port="5000")
    # cc_constant = cache_client(server_IP="http://172.112.40.147",server_port="5000")
    #cc_constant = cache_client(server_IP="http://10.10.10.64", server_port="8080")
    cc_constant.req_batch_num = 5000
    cc_constant.send_req_batch()
    cc_constant.moving_averag(cc_constant.cache_hit)
    # scenario two: introduce changes in traffic popularity
   # scenario one: constant traffic popularity
   #  cc_varying = cache_client()
    cc_varying = cache_client(server_IP="http://localhost", server_port="580")
    # cc_varying = cache_client(server_IP="http://172.16.100.20", server_port="5000")
    # cc_varying = cache_client(server_IP="http://172.112.40.147", server_port="5000")
    # cc_varying = cache_client(server_IP="http://10.10.10.64",server_port="8080")
    cc_varying.zipf_param = 1.5
    cc_varying.change_popular_contet_index = True
    cc_varying.req_batch_num = 5000
    cc_varying.send_req_batch()
plt.plot(cc_varying.moving_averag(cc_varying.cache_hit))
plt.plot(cc_varying.moving_averag(cc_varying.cache_hit_LFU))
plt.xlabel("File Request")
plt.ylabel("Cache hit ratio")
plt.title("ML vs LFU for a sudden change of the content popularity")
plt.legend(["ML","LFU"])
plt.ylim([0,1])
path = os.getcwd()+"/plots/ML_vs_LFU.svg"
plt.savefig(path)
plt.show()
data={"cache_dimension":cc_varying.cache_dimension,"total_file_number":cc_varying.total_file_number,
      "zipf_param1":cc_constant.zipf_param,"zipf_param2":cc_varying.zipf_param,
      "cache_hit_ML":cc_varying.moving_averag(cc_varying.cache_hit),"cache_hit_LFU":cc_varying.moving_averag(cc_varying.cache_hit_LFU)}
path = os.getcwd()+"/plots/ML_vs_LFU_shock.mat"
savemat(path,data)

