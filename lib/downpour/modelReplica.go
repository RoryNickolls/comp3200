package downpour

import (
	"comp3200/lib/messenger"
	"comp3200/lib/network"
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

	paramMsg.SendMessage("MDL")
	var networkConfig network.NetworkConfig
	paramMsg.ReceiveInterface(&networkConfig)
	mr.model = network.NewNetworkFromConfig(networkConfig)
	fmt.Println("Received model configuration")

	for {

		// fmt.Println("Requesting mini-batches...")
		dataMsg.SendMessage("REQ " + strconv.Itoa(fetch))

		var miniBatches [][]network.Record
		dataMsg.ReceiveInterface(&miniBatches)
		// fmt.Println("Received mini-batches")

		storedDeltas := 0
		request := 0

		weights, biases := mr.model.ZeroedParameters()


		// Perform training on data
		// Update model before each mini-batch, send deltas to parameter server after

		// fmt.Println("Training...")
		for i := 0; i < len(miniBatches); i++ {

			// Only make a request after fetch minibatches
			request--
			if request <= 0 {
				mr.receiveParameters(paramMsg)
				request = push
			}

			w, b := mr.model.TrainAndUpdate(miniBatches[i])

			for j := 0; j < len(w); j++ {
				weights[j].Add(&weights[j], &w[j])
				biases[j].AddVec(&biases[j], &b[j])
			}
			storedDeltas++

			// Only update after push minibatches
			if storedDeltas % push == 0 {
				mr.sendDeltas(paramMsg, weights, biases)
				weights, biases = mr.model.ZeroedParameters()
				storedDeltas = 0
			}
		}
		// fmt.Println("Finished training")
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
