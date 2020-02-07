#!/bin/bash

trap "kill 0" EXIT

echo "Building project..."
go build -o ./build

exe=./build/comp3200

parameter=":8889"

replicas=(":8900" ":8901" ":8902" ":8903" ":8904" ":8905" ":8906" ":8907")
data=(":8890" ":8891" ":8892" ":8893" ":8894" ":8895" ":8896" ":8897")

joined_data=":8890,:8891,:8892,:8893,:8894,:8895,:8896,:8897"

echo "Creating data servers"
for a in ${data[@]}; do
    $exe -type=data -host=$a &
done

sleep 1

echo "Provisioning data servers"
$exe -type=provision -dataServers=$joined_data

echo "Creating parameter server"
$exe -type=parameter -host=$parameter &> log &

sleep 3

echo "Creating model replicas"
for i in ${!replicas[@]}; do
    $exe -type=model -host=${replicas[i]} -data=${data[i]} -parameter=$parameter &
done

wait