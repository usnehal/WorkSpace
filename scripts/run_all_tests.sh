set -e
set -x

if [ -z "$1" ]
then
  echo "no server ip"
  SERVER=" "
else
  #IP=35.200.232.85
  IP=$1
  SERVER=" -s $IP"
fi

TESTS=" -m 50"
VERBOSE=" -v 0"

cd ~/WorkSpace/       
python3 ./client.py $SERVER -t 0 $VERBOSE $TESTS
python3 ./client.py $SERVER -t 1 $VERBOSE $TESTS
python3 ./client.py $SERVER -t 2 $VERBOSE $TESTS
python3 ./client.py $SERVER -t 3 $VERBOSE $TESTS
python3 ./client.py $SERVER -t 4 $VERBOSE $TESTS --split_layer 40
python3 ./client.py $SERVER -t 5 $VERBOSE $TESTS --split_layer 100

