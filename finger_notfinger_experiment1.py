#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 25 10:43:45 2019

@author: rk
"""

# create the sample file
import os ,re
import json

filenames = os.listdir('bio/train')
sample_json = []

for i in filenames:
    sample_json.append(
        {
        'image_id': i,
        'label': re.sub(r"[^A-Za-z(),!?\'\`]", " ", i.split('.')[0]).strip()
        }
        )

with open('data.json', 'w') as outfile:
    json.dump(sample_json, outfile, indent=4, sort_keys=True)
    
    
    
from imageatm.components import DataPrep

dp = DataPrep(
    image_dir='bio/train',
    samples_file='data.json',
    job_dir='bio_allclass'
)

dp.run(resize=True)    

'''
The following samples were dropped:
- {'image_id': 'Thumbs.db', 'label': 'NotFinger'}
Class distribution after validation:
Finger: 10 (20.0%)
NotFinger: 40 (80.0%)
Class mapping:
{0: 'Finger', 1: 'NotFinger'}

****** Creating train/val/test sets ******

Split distribution: train: 0.70, val: 0.1, test: 0.2

Partial split distribution: train: 0.70, val: 0.10, test: 0.20

Train set:
0: 7 (20.0%)
1: 28 (80.0%)
Val set:
0: 1 (20.0%)
1: 4 (80.0%)
Test set:
0: 2 (20.0%)
1: 8 (80.0%)
'''


from imageatm.components import Training

trainer = Training(dp.image_dir, dp.job_dir, epochs_train_dense=6, epochs_train_all=1)

trainer.run()





from imageatm.components import Evaluation

e = Evaluation(image_dir=dp.image_dir, job_dir=dp.job_dir)

e.run()

for i  in range(4):
    c, w = e.get_correct_wrong_examples(label=i)
    e.visualize_images(c, show_heatmap=True)
    e.visualize_images(w, show_heatmap=True)
