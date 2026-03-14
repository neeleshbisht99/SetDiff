#!/bin/bash

### To run example
# python main.py --config configs/example.yaml
# nohup ./run.sh > out.log 2>&1 &

### python main.py --config configs/sweep_visdiffbench_purity1.0_seed0_knowledge_bank/22_easy.yaml


### To run evaluations on sweep_pairedimagesets
# nohup python sweeps/sweep_pairedimagesets.py > sweep_pairedimagesets.log 2>&1 &

### To run blip server
# nohup python serve/vlm_server_blip.py > blip_server.log 2>&1 &

### To run clip server
# nohup python serve/clip_server.py > clip_server.log 2>&1 &


#  group1: "Lace wedding dresses"
#  group2: "Satin wedding dresses"
# 137_hard.yaml
# python main.py --config configs/sweep_visdiffbench_purity1.0_seed0_knowledge_bank/137_hard.yaml