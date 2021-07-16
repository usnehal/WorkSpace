#!/bin/bash

set -e

cd /home/suphale/WorkSpace

jupyter nbconvert server.ipynb --to python
jupyter nbconvert client.ipynb --to python
jupyter nbconvert inference.ipynb --to python
jupyter nbconvert train.ipynb --to python

chmod 777 Server.py
chmod 777 Client.py
chmod 777 inference.py
chmod 777 train.py

