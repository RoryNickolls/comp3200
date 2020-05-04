#!/bin/bash

source scripts/setup.sh

rm -rf log/async/*.log

parameter="10.0.0.102:8889"

replicas=(":8900" ":8901" ":8902" ":8903" ":8904" ":8905" ":8906" ":8907")
# replicas=(":8900")
# replicas=(":8900", ":8901")
# replicas=(":8900" ":8901" ":8902", ":8903")

# echo "Creating parameter server"
# $exe -algorithm=async -type=parameter -host=$parameter &

# sleep 2

echo "Creating model replicas"
for i in ${!replicas[@]}; do
    $exe -algorithm=async -type=model -parameter=$parameter &
done

wait