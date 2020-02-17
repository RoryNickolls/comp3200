#!/bin/bash

trap "kill 0" EXIT

echo "Building project..."
go build -o ./build/comp3200
echo "Build complete."

log=./output/log
> $log
echo "Output redirected to $log"

exe=./build/comp3200
