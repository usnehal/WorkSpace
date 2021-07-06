set -e

cd ~/WorkSpace/communication
python3 ./Client.py -s 35.200.232.85 -t 1 -v 0
python3 ./Client.py -s 35.200.232.85 -t 2 -v 0
python3 ./Client.py -s 35.200.232.85 -t 3 -v 0

