
import json
import httpx
import time
import numpy as np
import scipy.stats as stats
class cache_client():
    def __init__(self, cache_dimension=100, total_file_number=1000, zipf_param=1.5):
        self.req_batch_num = 200
        self.cache_dimension = cache_dimension
        self.reward_for_achieving_goal = 1.0
        self.step_reward_for_not_achieving_goal = 0
        self.total_file_number = total_file_number
        self.zipf_param = zipf_param
    def send_req_batch(self):
        with httpx.Client() as client:
            t1=time.time()
            cache_hit = []
            for i in range(self.req_batch_num):
                self.observation = self.zipf_truncated(self.zipf_param,self.total_file_number,1)
                r = client.get('http://127.0.0.1:24/'+str(self.observation[0]))
                reply=json.loads(r.content.decode("utf-8"))
                print(self.observation, reply["Cache_status"])
                cache_hit.append(int(reply["Cache_status"]))
                # print(r.text)
            cache_hit_ratio = np.sum(cache_hit)/self.req_batch_num
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
        # sample = bounded_zipf.rvs(size=10000)
        # plt.hist(sample, bins=np.arange(1, N + 2))
        # plt.show()
if __name__ == '__main__':
    cc = cache_client()
    cc.send_req_batch()