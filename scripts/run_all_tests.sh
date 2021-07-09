set -e

if [ -z "$1" ]
then
  echo "no server ip"
else
  #IP=35.200.232.85
  IP=$1
fi

cd ~/WorkSpace/communication        
if [ -z "$IP" ]
then
python3 ./Client.py -t 1 -v 0       
python3 ./Client.py -t 2 -v 0       
python3 ./Client.py -t 3 -v 0       
else
python3 ./Client.py -s $IP -t 1 -v 0
python3 ./Client.py -s $IP -t 2 -v 0
python3 ./Client.py -s $IP -t 3 -v 0
fi

