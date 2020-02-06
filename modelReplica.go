package main

import (
	"fmt"

	"gonum.org/v1/gonum/mat"
)

type ModelReplica struct {
	model *Network
	data  Data
}

func LaunchModelReplica(address string, dataAddress string, parameterAddress string) {

	mr := ModelReplica{}
	mr.model = NewNetwork().WithLayer(784, 300, "sigmoid").WithLayer(300, 100, "sigmoid").WithLayer(100, 10, "softmax")

	dataMsg := Connect(dataAddress)
	paramMsg := Connect(parameterAddress)

	for {
		dataMsg.SendMessage("REQ")
		mr.getData(dataMsg)
		miniBatches := mr.data.GetMiniBatches(100)

		fmt.Println("Received data from data server")

		// Perform training on data
		// Update model before each mini-batch, send deltas to parameter server after
		for i := 0; i < len(miniBatches); i++ {
			mr.receiveParameters(paramMsg)
			weightDeltas, biasDeltas := mr.model.Train(miniBatches[i])
			mr.sendDeltas(paramMsg, weightDeltas, biasDeltas)
		}
	}
}

func (mr *ModelReplica) getData(messenger Messenger) {
	fmt.Println("Waiting for data")

	var data Data
	messenger.ReceiveInterface(&data)
	mr.data = data
}

func (mr *ModelReplica) receiveParameters(messenger Messenger) {
	fmt.Println("Requesting parameters")

	// Send request to parameter server
	messenger.SendMessage("REQ")

	// Retrieve weights and biases for each layer from parameter server
	var weights []mat.Dense
	var biases []mat.VecDense

	messenger.ReceiveInterface(&weights)
	messenger.ReceiveInterface(&biases)

	mr.model.SetParameters(weights, biases)
	fmt.Println("Model updated with most recent parameters")
}

func (mr *ModelReplica) sendDeltas(messenger Messenger, weights []mat.Dense, biases []mat.VecDense) {

	// send weight and bias deltas to parameter server
	messenger.SendMessage("UPD")
	messenger.SendInterface(weights)
	messenger.SendInterface(biases)
	fmt.Println("Sent updated deltas")
}
