#!/bin/bash

source scripts/setup.sh

rm -rf log/downpour/*.log

parameter=":8889"

# replicas=(":8900" ":8901" ":8902" ":8903" ":8908" ":8909" ":8910" ":8911")
# data=(":8890" ":8891" ":8892" ":8893"  ":8908" ":8909" ":8910" ":8911")
# joined_data=":8890,:8891,:8892,:8893,:8908,:8909,:8910,:8911"
# replicas=(":8900")
# data=(":8890")
# joined_data=":8890"
# replicas=(":8900" ":8901")
# data=(":8890" ":8891")
# joined_data=":8890,:8891"
replicas=(":8900" ":8901" ":8902" ":8903")
data=(":8890" ":8891" ":8892" ":8893")
joined_data=":8890,:8891,:8892,:8893"

echo "Creating data servers"
for a in ${data[@]}; do
    $exe -type=data -algorithm=downpour -host=$a &
done

sleep 1

echo "Provisioning data servers"
$exe -type=provision -dataServers=$joined_data

echo "Creating parameter server"
$exe -algorithm=downpour -type=parameter -host=$parameter &

sleep 2

echo "Creating model replicas"
for i in ${!replicas[@]}; do
    $exe -algorithm=downpour -type=model -data=${data[i]} -parameter=$parameter -fetch=100 -push=20 &
done

wait