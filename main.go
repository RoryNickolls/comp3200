package main

import (
	"comp3200/lib/downpour"
	"comp3200/lib/network"
	"comp3200/lib/synchronous"
	"flag"
	"strings"
)

func main() {
	var algorithm string

	// General parameters
	var address string
	var nodeType string
	var parameterAddress string

	// Downpour parameters
	var dataAddress string
	var fetch int
	var push int
	var dataServers string

	// Synchronous parameters
	var clients int

	// General
	flag.StringVar(&algorithm, "algorithm", "downpour", "Algorithm to use for training")
	flag.StringVar(&address, "host", "localhost:8888", "Host address")
	flag.StringVar(&nodeType, "type", "none", "Type of entity this is: parameter, model, data")
	flag.StringVar(&parameterAddress, "parameter", "localhost:8888", "Address of the parameter server")

	// Downpour specific
	flag.StringVar(&dataAddress, "data", "localhost:8888", "Address of the data server for this model")
	flag.IntVar(&fetch, "fetch", 10, "Number of mini-batches to fetch at a time")
	flag.IntVar(&push, "push", 10, "Number of mini-batches to process before sending updates")
	flag.StringVar(&dataServers, "dataServers", "", "Comma-separated addresses of data servers to provision")

	// Synchronous specific
	flag.IntVar(&clients, "clients", 2, "Number of clients expected to connect")

	flag.Parse()

	if algorithm == "standard" {
		network.TrainStandardNetwork()
	} else if algorithm == "downpour" {
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
			break
		}
	} else if algorithm == "sync" {
		switch nodeType {
		case "parameter":
			network := network.NewNetwork().WithLayer(784, 300, "sigmoid").WithLayer(300, 100, "sigmoid").WithLayer(100, 10, "softmax").WithLearningRate(0.1)
			synchronous.LaunchSynchronousParameterServer(address, clients, network)
			break
		case "client":
			synchronous.LaunchClient(parameterAddress)
			break
		}
	}
}
