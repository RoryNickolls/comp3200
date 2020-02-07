package main

import (
	"fmt"
	"strconv"

	"gonum.org/v1/gonum/mat"
)

type ModelReplica struct {
	model *Network
	data  Data
	fetch int
	push  int
}

func LaunchModelReplica(address string, dataAddress string, parameterAddress string, fetch int, push int) {

	mr := ModelReplica{fetch: fetch, push: push}
	dataMsg := Connect(dataAddress)
	paramMsg := Connect(parameterAddress)

	fmt.Println("Reqesting model configuration...")
	paramMsg.SendMessage("MDL")
	var networkConfig NetworkConfig
	paramMsg.ReceiveInterface(&networkConfig)
	mr.model = NewNetworkFromConfig(networkConfig)
	fmt.Println("Received model configuration")

	for {

		fmt.Println("Requesting mini-batches...")
		dataMsg.SendMessage("REQ " + strconv.Itoa(fetch))

		var miniBatches [][]Record
		dataMsg.ReceiveInterface(&miniBatches)
		fmt.Println("Received mini-batches")

		// Perform training on data
		// Update model before each mini-batch, send deltas to parameter server after

		fmt.Println("Training...")
		for i := 0; i < len(miniBatches); i++ {
			mr.receiveParameters(paramMsg)
			weightDeltas, biasDeltas := mr.model.Train(miniBatches[i])
			mr.sendDeltas(paramMsg, weightDeltas, biasDeltas)
		}
		fmt.Println("Finished training")
	}
}

func (mr *ModelReplica) getData(messenger Messenger) {
	var data Data
	messenger.ReceiveInterface(&data)
	mr.data = data
}

func (mr *ModelReplica) receiveParameters(messenger Messenger) {
	// Send request to parameter server
	messenger.SendMessage("REQ")

	// Retrieve weights and biases for each layer from parameter server
	var weights []mat.Dense
	var biases []mat.VecDense

	messenger.ReceiveInterface(&weights)
	messenger.ReceiveInterface(&biases)

	mr.model.SetParameters(weights, biases)
}

func (mr *ModelReplica) sendDeltas(messenger Messenger, weights []mat.Dense, biases []mat.VecDense) {
	// send weight and bias deltas to parameter server
	messenger.SendMessage("UPD")
	messenger.SendInterface(weights)
	messenger.SendInterface(biases)
}
