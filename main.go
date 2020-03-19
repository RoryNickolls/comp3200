package main

import (
	"comp3200/lib"
	"comp3200/lib/downpour"
	"comp3200/lib/messenger"
	"comp3200/lib/network"
	"comp3200/lib/synchronous"
	"flag"
	"fmt"
	"log"
	"math/rand"
	"strings"
	"time"
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
	flag.StringVar(&dataAddress, "data", "", "Address of the data server for this model")
	flag.IntVar(&fetch, "fetch", 10, "Number of mini-batches to fetch at a time")
	flag.IntVar(&push, "push", 10, "Number of mini-batches to process before sending updates")
	flag.StringVar(&dataServers, "dataServers", "", "Comma-separated addresses of data servers to provision")

	// Synchronous specific
	flag.IntVar(&clients, "clients", 2, "Number of clients expected to connect")

	flag.Parse()

	if lib.LogMessages {
		messenger.StartLoggingMessages()
	}

	data := network.LoadData()
	model := network.NewNetwork().WithLayer(784, 300, "sigmoid").WithLayer(300, 100, "sigmoid").WithLayer(100, 10, "softmax").WithLearningRate(0.1)
	if algorithm == "standard" {
		network.TrainStandardNetwork()
	} else if algorithm == "check" {
		model.TrainAndUpdate(data.Train[:5000])
		for i := 0; i < 10; i++ {
			w, b := model.GradientCheck(data.Train[rand.Intn(len(data.Train))], 0.0000001)
			fmt.Println(w, b)
		}
	} else if algorithm == "downpour" {
		switch nodeType {
		case "parameter":
			lib.SetupLog("downpour/parameter")
			go ContinuousParameterEvaluation(model, data.Test)
			downpour.LaunchParameterServer(address, model, false)
			break
		case "model":
			lib.SetupLog("downpour/model")
			go ContinuousModelEvaluation()
			downpour.LaunchModelReplica(dataAddress, parameterAddress, 50, fetch, push)
			break
		case "data":
			lib.SetupLog("downpour/data")
			downpour.LaunchDataServer(address)
			break
		case "provision":
			lib.SetupLog("downpour/provisioner")
			addresses := strings.Split(dataServers, ",")
			downpour.ProvisionData(addresses)
			break
		case "none":
			break
		}
	} else if algorithm == "sync" {
		switch nodeType {
		case "parameter":
			lib.SetupLog("sync/parameter")
			go ContinuousParameterEvaluation(model, data.Test)
			synchronous.LaunchSynchronousParameterServer(address, clients, model)
			break
		case "client":
			lib.SetupLog("sync/model")
			synchronous.LaunchClient(parameterAddress)
			break
		}
	} else if algorithm == "async" {
		switch nodeType {
		case "parameter":
			lib.SetupLog("async/parameter")
			go ContinuousParameterEvaluation(model, data.Test)
			downpour.LaunchParameterServer(address, model, true)
			break
		case "model":
			lib.SetupLog("async/model")
			go ContinuousModelEvaluation()
			downpour.LaunchModelReplica("", parameterAddress, 20, 1, 1)
		}
	}
}

// Wait for 1 minute
var wait int = 1

func ContinuousParameterEvaluation(nn *network.Network, testData []network.Record) {
	count := 0
	for {
		loss, accuracy := nn.Evaluate(testData)
		curTime := count * wait
		rx := messenger.Received()
		tx := messenger.Sent()
		log.Printf("%d,%f,%f,%d,%d\n", curTime, loss, accuracy, rx, tx)
		count++
		time.Sleep(time.Duration(wait) * time.Minute)
	}
}

func ContinuousModelEvaluation() {
	count := 0
	for {
		curTime := count * wait
		rx := messenger.Received()
		tx := messenger.Sent()
		log.Printf("%d,%d,%d\n", curTime, rx, tx)
		count++
		time.Sleep(time.Duration(wait) * time.Minute)
	}
}
