import os
import json


train_json = []
test_json = []

filelist = json.load(open('dataset.json'))

for file in range(len(filelist)):
    if file % 5 == 0:
        test_json.append(filelist[file])
    else:
        train_json.append(filelist[file])

with open('train.json', "w") as file:
    json.dump(train_json, file)

with open('test.json', "w") as file:
    json.dump(test_json, file)