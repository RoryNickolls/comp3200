package main

import (
	"flag"
)

func main() {
	var address string
	var nodeType string
	
	// Model replica parameters
	var dataAddress string
	var parameterAddress string

	flag.StringVar(&address, "host", "localhost:8888", "Host address")
	flag.StringVar(&nodeType, "type", "none", "Type of entity this is: parameter, model, data")

	flag.StringVar(&dataAddress, "data", "localhost:8888", "Address of the data server for this model")
	flag.StringVar(&parameterAddress, "parameter", "localhost:8888", "Address of the parameter server")

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
	case "none":
		TrainStandardNetwork()
		break
	}
}
