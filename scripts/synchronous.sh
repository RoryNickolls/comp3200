#!/bin/bash

source scripts/setup.sh

rm -rf log/sync/*.log

parameter=":8888"
clients=4

echo "Creating parameter server"
$exe -algorithm=sync -type=parameter -host=$parameter -clients=4 &

sleep 1

echo "Creating clients"
for i in $(seq 1 $clients); do
    $exe -algorithm=sync -type=client -parameter=$parameter &
done

wait