#!/bin/bash
python --version
pip install --upgrade azure-cli
pip install --upgrade azureml-sdk
pip install -r requirements.txt
pip install numpy
pip install matplotlib
pip install scikit-learn
pip install pandas
pip install azureml-core
pip install --upgrade azureml-dataset-runtime