#!/bin/bash

set -e

cd /home/suphale/WorkSpace/exp

jupyter nbconvert Server.ipynb --to python
jupyter nbconvert Client.ipynb --to python
jupyter nbconvert inference.ipynb --to python

chmod 777 Server.py
chmod 777 Client.py

