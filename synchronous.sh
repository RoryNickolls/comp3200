#!/bin/bash

source setup.sh

parameter=":8888"
clients=4

echo "Creating parameter server"
$exe -algorithm=sync -type=parameter -host=$parameter -clients=$clients &

sleep 1

echo "Creating clients"
for i in $(seq 1 $clients); do
    $exe -algorithm=sync -type=client -parameter=$parameter &
done

# $exe -algorithm=sync -type=client -parameter=$parameter &

wait