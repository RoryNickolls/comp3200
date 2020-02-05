#!/bin/bash
echo "Building project..."
go build -o ./build

exe=./build/comp3200

echo "Creating parameter server"
./build/comp3200 -type=parameter -host=:8889

sleep 1

# Data servers are on ports 8890-8899
echo "Creating data servers"
./build/comp3200 -type=data -host=:8890

sleep 1

# Model replicas are on ports 8900-8909
echo "Creating model replicas"
./build/comp3200 -type=model -host=:8900 -data=:8890 -parameter=:8889