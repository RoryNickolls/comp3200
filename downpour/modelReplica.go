package downpour

import (
	"comp3200/messenger"
	"comp3200/network"
	"fmt"
	"strconv"

	"gonum.org/v1/gonum/mat"
)

type ModelReplica struct {
	model *network.Network
	data  network.Data
	fetch int
	push  int
}

func LaunchModelReplica(address string, dataAddress string, parameterAddress string, fetch int, push int) {

	mr := ModelReplica{fetch: fetch, push: push}
	dataMsg := messenger.Connect(dataAddress)
	paramMsg := messenger.Connect(parameterAddress)

	fmt.Println("Reqesting model configuration...")
	paramMsg.SendMessage("MDL")
	var networkConfig network.NetworkConfig
	paramMsg.ReceiveInterface(&networkConfig)
	mr.model = network.NewNetworkFromConfig(networkConfig)
	fmt.Println("Received model configuration")

	for {

		fmt.Println("Requesting mini-batches...")
		dataMsg.SendMessage("REQ " + strconv.Itoa(fetch))

		var miniBatches [][]network.Record
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

func (mr *ModelReplica) getData(msg messenger.Messenger) {
	var data network.Data
	msg.ReceiveInterface(&data)
	mr.data = data
}

func (mr *ModelReplica) receiveParameters(msg messenger.Messenger) {
	// Send request to parameter server
	msg.SendMessage("REQ")

	// Retrieve weights and biases for each layer from parameter server
	var weights []mat.Dense
	var biases []mat.VecDense

	msg.ReceiveInterface(&weights)
	msg.ReceiveInterface(&biases)

	mr.model.SetParameters(weights, biases)
}

func (mr *ModelReplica) sendDeltas(msg messenger.Messenger, weights []mat.Dense, biases []mat.VecDense) {
	// send weight and bias deltas to parameter server
	msg.SendMessage("UPD")
	msg.SendInterface(weights)
	msg.SendInterface(biases)
}
