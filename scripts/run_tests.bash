set -e
set -x

cd ~/WorkSpace/

echo "1 Mbps speed"
sudo wondershaper eth0 1000 1000
~/WorkSpace/scripts/run_all_tests_subset.bash 1_mbps

echo "3 Mbps speed"
sudo wondershaper eth0 3000 3000
~/WorkSpace/scripts/run_all_tests_subset.bash 3_mbps

echo "5 Mbps speed"
sudo wondershaper eth0 5000 5000
~/WorkSpace/scripts/run_all_tests_subset.bash 5_mbps

echo "7 Mbps speed"
sudo wondershaper eth0 7000 7000
~/WorkSpace/scripts/run_all_tests_subset.bash 7_mbps

echo "10 Mbps speed"
sudo wondershaper eth0 10000 10000
~/WorkSpace/scripts/run_all_tests_subset.bash 10_mbps

echo "15 Mbps speed"
sudo wondershaper eth0 15000 15000
~/WorkSpace/scripts/run_all_tests_subset.bash 15_mbps

echo "20 Mbps speed"
sudo wondershaper eth0 20000 20000
~/WorkSpace/scripts/run_all_tests_subset.bash 1_mbps

echo "30 Mbps speed"
sudo wondershaper eth0 30000 30000
~/WorkSpace/scripts/run_all_tests_subset.bash 30_mbps

echo "50 Mbps speed"
sudo wondershaper eth0 50000 1000
~/WorkSpace/scripts/run_all_tests_subset.bash 50_mbps

