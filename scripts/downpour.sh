#!/bin/bash

source scripts/setup.sh

rm -rf log/downpour/*.log

parameter=":8889"

replicas=(":8900" ":8901" ":8902" ":8903" ":8904" ":8905" ":8906" ":8907")
data=(":8890" ":8891" ":8892" ":8893" ":8894" ":8895" ":8896" ":8897")
# replicas=(":8900" ":8901" ":8902" ":8903")
# data=(":8890" ":8891" ":8892" ":8893")
# joined_data=":8890,:8891,:8892,:8893"
joined_data=":8890,:8891,:8892,:8893,:8894,:8895,:8896,:8897"

echo "Creating data servers"
for a in ${data[@]}; do
    $exe -type=data -algorithm=downpour -host=$a &
done

sleep 1

echo "Provisioning data servers"
$exe -type=provision -dataServers=$joined_data

echo "Creating parameter server"
$exe -algorithm=downpour -type=parameter -host=$parameter &

sleep 3

echo "Creating model replicas"
for i in ${!replicas[@]}; do
    $exe -algorithm=downpour -type=model -data=${data[i]} -parameter=$parameter -fetch=5 -push=1 &
done

wait