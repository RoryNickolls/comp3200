#!/bin/bash

source scripts/setup.sh

parameter=":8889"

replicas=(":8900" ":8901" ":8902" ":8903" ":8904" ":8905" ":8906" ":8907")
data=(":8890" ":8891" ":8892" ":8893" ":8894" ":8895" ":8896" ":8897")
# replicas=(":8900" ":8901" ":8902" ":8903")
# data=(":8890" ":8891" ":8892" ":8893")
# joined_data=":8890,:8891,:8892,:8893"
joined_data=":8890,:8891,:8892,:8893,:8894,:8895,:8896,:8897"

echo "Creating data servers" >> $log
for a in ${data[@]}; do
    $exe -type=data -host=$a &>> $log &
done

sleep 1

echo "Provisioning data servers" >> $log
$exe -type=provision -dataServers=$joined_data

echo "Creating parameter server" >> $log
$exe -type=parameter -host=$parameter &>> $log &

sleep 3

echo "Creating model replicas" >> $log
for i in ${!replicas[@]}; do
    $exe -type=model -host=${replicas[i]} -data=${data[i]} -parameter=$parameter -fetch=30 -push=25 &>> $log &
done

wait