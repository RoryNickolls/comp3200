# comp3200
This repository contains the code for my final year university project titled 'Comparing distributed and non-distributed approaches to training neural networks'. It is a neural network written from scratch in Go, designed to be trained across a network with a variety of algorithms.

# Usage
The three bash scripts in the 'scripts/' folder are used to launch each algorithm:
- asynchronous.sh
- downpour.sh
- synchronous.sh

Each of these scripts contains configurable parameters and allows the user to set the address of each machine.

Local addresses can be written in the form ":\<port\>" and remote addresses should be written "\<ip\>:\<port\>".

Each script will output logs to the respective folder inside 'log/'

They will also call setup.sh which cleans and rebuilds the project if any changes were made
