package main

import (
	"comp3200/downpour"
	"comp3200/network"
	"flag"
	"strings"
)

func main() {
	var address string
	var nodeType string

	// Model replica parameters
	var dataAddress string
	var parameterAddress string
	var fetch int
	var push int

	var dataServers string

	flag.StringVar(&address, "host", "localhost:8888", "Host address")
	flag.StringVar(&nodeType, "type", "none", "Type of entity this is: parameter, model, data")

	flag.StringVar(&dataAddress, "data", "localhost:8888", "Address of the data server for this model")
	flag.StringVar(&parameterAddress, "parameter", "localhost:8888", "Address of the parameter server")
	flag.IntVar(&fetch, "fetch", 10, "Number of mini-batches to fetch at a time")
	flag.IntVar(&push, "push", 10, "Number of mini-batches to process before sending updates")

	flag.StringVar(&dataServers, "dataServers", "", "Comma-separated addresses of data servers to provision")

	flag.Parse()

	switch nodeType {
	case "parameter":
		network := network.NewNetwork().WithLayer(784, 300, "sigmoid").WithLayer(300, 100, "sigmoid").WithLayer(100, 10, "softmax").WithLearningRate(0.1)
		downpour.LaunchParameterServer(address, network)
		break
	case "model":
		downpour.LaunchModelReplica(address, dataAddress, parameterAddress, fetch, push)
		break
	case "data":
		downpour.LaunchDataServer(address)
		break
	case "provision":
		addresses := strings.Split(dataServers, ",")
		downpour.ProvisionData(addresses)
		break
	case "none":
		TrainStandardNetwork()
		break
	}
}
