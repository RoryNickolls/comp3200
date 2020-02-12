#!/bin/bash

trap "kill 0" EXIT

echo "Building project..."
go build -o ./build

exe=./build/comp3200