from json.decoder import JSONDecodeError
import gevent
from locust import HttpUser, task, between
from locust.env import Environment
import locust.stats
import pprint
from locust.stats import StatsCSVFileWriter, stats_printer, stats_history
# import random
import os
import json
import pandas
from pandas.core.frame import DataFrame
import numpy as np
import uuid
import random
import datetime

HOST = os.getenv("HOST", "http://172.16.100.17:8080")
USERS = os.getenv("USERS", int(1000))
SPAWN_RATE = os.getenv("SPAWN_RATE", int(1000))
DURATION = os.getenv("DURATION", int(100))
LIBRARY = os.getenv("LIBRARY", int(1e3))
EDGE_LIBRARY = os.getenv("EDGE_LIBRARY", int(10))
RESULTS_FILE = os.getenv("RESULTS_FILE", "results.csv")
#POLICY = os.getenv("POLICY", "None")
POLICY = os.getenv("POLICY", "w_lru")

class Requester(HttpUser):
    """
    Simple user requesting content from Cache
    """

    wait_time = between(0.5, 0.7)  # time between task executions
    unique_id = None
    result_df = None
    host = str(HOST)
    zipf_requests = None
    policy = POLICY
    timer = None

    def on_start(self):
        self.unique_id = str(uuid.uuid4())
        self.fill_zipf_array()
        self.timer = datetime.datetime.now()

    def fill_zipf_array(self):
        self.zipf_requests = (np.random.zipf(1.56, int(float(LIBRARY)))).tolist()
        # self.zipf_requests = (np.random.zipf(3.01, int(float(LIBRARY)))).tolist()

    @task()
    def generic_requests(self):
        """
        Generating random requests
        """
        
        #  Random requests
        # content_id = random.choice(range(0, int(float(LIBRARY))))

        # Zipf requests
        try:
            content_id = self.zipf_requests.pop()
        except IndexError as e:
            print("***>%s" % e)
            self.fill_zipf_array()
            return

        # shifting distribution at half DURATION
        _now = datetime.datetime.now()
        _SHIFT = 1000
        if int((_now - self.timer).seconds) >= (int(DURATION)/2):
            content_id += _SHIFT

        print("POLICY: %s, CONTENT: %s" % (self.policy, content_id))

        # requests types
        if self.policy == 'None':
            r = self.client.get(f"/request_no_cache/{content_id}", name="/request")
        else:
            r = self.client.get(f"/request/{self.policy}/{content_id}", name="/request")

        results = dict()
        try:
            results = dict(json.loads(r.json()))
        except:
            print("***>JSON error")
            print(r)
            return
        
        # using index comming from Server
        _index = list(dict(results['content']).keys())[0]
        results.setdefault("id", {_index: self.unique_id})
        print(results)

        self.result_df = DataFrame(results, columns=results.keys())
        _header = False
        if not os.path.isfile(RESULTS_FILE):
            _header = True
        self.result_df.to_csv(RESULTS_FILE, mode='a', header=_header)




##################
## LOCUST section
##################
_keys = sorted(dict(os.environ).keys())
for _k in _keys:
    print("-%s: %s" % (_k, dict(os.environ)[_k]))

# setup Environment and Runner
env = Environment(user_classes=[Requester])
env.create_local_runner()

# creating the csv writer
stats_path = os.path.join(os.getcwd(), "data")
csv_writer = StatsCSVFileWriter(environment=env,
                                base_filepath=stats_path,
                                full_history=True,
                                percentiles_to_report=[90.0, 95.0])

# locust.stats.CSV_STATS_INTERVAL_SEC = 1
# locust.stats.CSV_STATS_FLUSH_INTERVAL_SEC = 60

# start a WebUI instance
env.create_web_ui(host="127.0.0.1", port=8089,
                  stats_csv_writer=csv_writer)

# start csv writer
gevent.spawn(csv_writer)

# start a greenlet that periodically outputs the current stats
gevent.spawn(stats_printer(env.stats))

# start a greenlet that save current stats to history
gevent.spawn(stats_history, env.runner)

# start the test
env.runner.start(user_count=int(USERS), spawn_rate=int(SPAWN_RATE), wait=True)

# in DURATION seconds stop the runner
gevent.spawn_later(int(DURATION), lambda: env.runner.quit())

# wait for the greenlets
env.runner.greenlet.join()

# stop the web server for good measures
env.web_ui.stop()


# Retrieve CSV from InfluxDB
# my_db = DBClient(bucket=BUCKET,
#                     org=ORG,
#                     token=TOKEN,
#                     url=URL)

# # tables = my_db.retrieve_pandas()



