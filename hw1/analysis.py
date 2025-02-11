import json
import math
import random
import re
import requests
import statistics
from collections import Counter

url = 'https://raw.githubusercontent.com/clinc/oos-eval/master/data/data_full.json'
resp = requests.get(url)
data = json.loads(resp.text)

print("DATA SAMPLES:")
for d in data:
    print(d, len(data[d]))
    for i in range(5):
        print(data[d][i])

print("\nTOPICS:")
topic_set = set()
for v, k in data["val"]:
    topic_set.add(k)
print("total topics number", len(topic_set))
print("topics samples", random.choices(list(topic_set), k=10))

print("\nSTATS:")
# Number of the requests (dialogues/turns/sentences) for system
request_num = sum(len(data[d]) for d in data)
print("total requests number", request_num)

# Number of the words for system
word_list = []
request_len = []
separators = r"[^a-zA-Z0-9':]+"
for d in data:
    for request, topic in data[d]:
        request_split = re.split(separators, request)
        word_list.extend(request_split)
        request_len.append(len(request_split))

# Other stats
total_count = len(word_list)
print("total words number", total_count)
print("avg request length (in words)", round(statistics.mean(request_len), 2))
print("std request length (in words)", round(statistics.stdev(request_len), 2))

word_counter = Counter(word_list)
print("vocabulary size", len(word_counter.keys()))

# Shannon entropy
entropy = -1*sum((count/total_count)*math.log(count/total_count) for count in word_counter.values())
print("shannon entropy", round(entropy, 4))
