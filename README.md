# StrawberriiesDetection
StrawberriesDetection

Researcher of the Universidad Autonoma de Baja California initiates a study releated to strawberry diseases by utilizing Deep Learning techniques... (2025)

Use the next command to get access to this repository: !git clone https://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@github.com/edramos-lab/StrawberriiesDetection.git

This repository containes two files:
1) download.py
2) json2yolo.py

#Use
##download.py
Execute the first script in order to download the dataset to specific path, e.g. python download.py --pathToDownload <specific path where the dataset will be downloaded>

##json2yolo.py
Execute the second script to convert from json format to yolo as follows:  python json2yolo.py --inputPath <previous path where dataset is located + train / val or test folder>