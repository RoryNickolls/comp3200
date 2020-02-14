#!/bin/bash

trap "kill 0" EXIT

echo "Building project..."
go build -o ./build/comp3200

exe=./build/comp3200