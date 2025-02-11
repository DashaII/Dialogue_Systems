There are two training sets in the chosen dataset and they are quite small, so to keep things simple I analyzed all the available datasets all together.

1. The name of the dataset. What kind of data it is (domain, modality)

Clinc-OOS (or CLINC150) is a dataset for task-oriented dialog systems presented in a text form.

2. Where you downloaded it from (include the original URL)

https://github.com/clinc/oos-eval

3. How it was collected

The authors used crowdsourcing to generate the dataset they asked crowd workers to either paraphrase "seed" phrases, or respond to scenarios (e.g. "pretend you need to book a flight, what would you say?"). The crowdsourcing was used to generate data for both in-scope and out-of-scope data.

4. What kind of dialogue system or dialogue system component it's designed for

This dataset is for evaluating the performance of intent classification systems in the presence of "out-of-scope" queries. OOS are the ones that do not fall into any of the system-supported intent classes.

5. What kind of annotation is present (if any at all), how was it obtained (human/automatic)

The classes (topics) of the requests to the dialogue system. It was obtained in a fixed human way, by these I mean that topics were defined from the beginning. 

6. What format is it stored in

JSON

7. What is the license

The dataset is free to use, but the authors should be cited

### STATS:

1. total requests number 23700
2. total words number 197527
3. avg request length (in words) 8.33
4. std request length (in words) 3.22
5. vocabulary size 7538
6. shannon entropy 5.9584


### Discussion:

There are few points I thought about while working with the dataset: 

1. The dataset is quite small, out of total 23700 entries just 1200 are out-of-scope entries. This doesn't seem enough for a proper training.
2. In some cases rephrasing seems slightly artificial, e.g. "what do you want me to refer to you as", "give me your name", "what are you called". It seems obvious that respondents were asked to formulate the questions differently and tried so hard that sometimes it doesn't sound natural or polite. 
3. Also, I didn't get from the description and context why some topics are considered to be OOS. Was it designed for a particular task-oriented system? I see it as a limitation for a wider range of systems.
4. On the other hand, the dataset is well-structured (JSON) and of a good quality from language point of view. 
