## NPA: Improving Large-scale Graph Neural Networks with
Non-parametric Attention

### Overview

This repo contains an example implementation of the Non-Parameter Attention (NPA) with SGC as a baseline. Equipped with NPA, SGC can achieve significant performance gains without losing scalability.


### Requirements
Environments: Win-64 or Linux-64, GPU:TITAN RTX. Other requirements can be seen in `requirements.txt`.

Note that we offer a pure python version and a python+cpp version to accelerate python loop. The details can be seen in `run.sh`.


### Run the Codes
To evalute the performance of SGC+NPA, please run the commands in `run.sh`.


### Test Results
| Method | Cora | Citeseer | Pubmed |
| :---: | :----: | :---: | :---: |
| SGC | 81.0 | 71.9 | 78.9 |
| SGC+NPA | 83.0 | 73.6 | 80.1 |


