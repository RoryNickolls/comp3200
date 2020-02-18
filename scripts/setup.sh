#!/bin/bash

trap "kill 0" EXIT

echo "Building project..."
go build -o ./build/comp3200
echo "Build complete."

exe=./build/comp3200
