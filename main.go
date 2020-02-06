package main

import (
	"flag"
	"strings"
)

func main() {
	var address string
	var nodeType string

	// Model replica parameters
	var dataAddress string
	var parameterAddress string

	var dataServers string

	flag.StringVar(&address, "host", "localhost:8888", "Host address")
	flag.StringVar(&nodeType, "type", "none", "Type of entity this is: parameter, model, data")

	flag.StringVar(&dataAddress, "data", "localhost:8888", "Address of the data server for this model")
	flag.StringVar(&parameterAddress, "parameter", "localhost:8888", "Address of the parameter server")

	flag.StringVar(&dataServers, "dataServers", "", "Comma-separated addresses of data servers to provision")

	flag.Parse()

	switch nodeType {
	case "parameter":
		LaunchParameterServer(address)
		break
	case "model":
		LaunchModelReplica(address, dataAddress, parameterAddress)
		break
	case "data":
		LaunchDataServer(address)
		break
	case "provision":
		addresses := strings.Split(dataServers, ",")
		ProvisionData(addresses)
		break
	case "none":
		TrainStandardNetwork()
		break
	}
}
