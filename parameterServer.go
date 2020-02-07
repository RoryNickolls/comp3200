package main

import (
	"fmt"
	"net"

	"gonum.org/v1/gonum/mat"
)

type ParameterServer struct {
	model *Network
}

var data *Data

func LaunchParameterServer(address string) {

	data = loadData()

	fmt.Println("Launching parameter server")
	ps := ParameterServer{}
	ps.model = NewNetwork().WithLayer(784, 300, "sigmoid").WithLayer(300, 100, "sigmoid").WithLayer(100, 10, "softmax").WithLearningRate(0.1)

	l, err := net.Listen("tcp4", address)
	if err != nil {
		fmt.Println("ERR:", err)
		return
	}

	for {
		conn, err := l.Accept()
		if err != nil {
			fmt.Println("ERR:", err)
			return
		}
		go ps.handleConnection(NewMessenger(conn))
	}
}

func (ps *ParameterServer) handleConnection(messenger Messenger) {
	fmt.Println("New model replica connected")
	for {
		var cmd string
		messenger.ReceiveMessage(&cmd)

		switch cmd {
		// Requesting parameters
		case "REQ":
			ps.handleParameterRequest(messenger)
			break
			// Updating parameters
		case "UPD":
			ps.handleParameterUpdate(messenger)
			break
		case "MDL":
			ps.handleModelRequest(messenger)
			break
		}
	}
}

func (ps *ParameterServer) handleParameterRequest(messenger Messenger) {
	//fmt.Println("Received request for parameters")
	weights, biases := ps.model.Parameters()

	// send current state of weights and biases
	messenger.SendInterface(weights)
	messenger.SendInterface(biases)
}

func (ps *ParameterServer) handleModelRequest(messenger Messenger) {
	messenger.SendInterface(ps.model.config)
}

var updates int

func (ps *ParameterServer) handleParameterUpdate(messenger Messenger) {

	// receive deltas for weights and biases
	var weightDeltas []mat.Dense
	var biasDeltas []mat.VecDense

	messenger.ReceiveInterface(&weightDeltas)
	messenger.ReceiveInterface(&biasDeltas)

	// update master model with deltas
	ps.model.UpdateWithDeltas(weightDeltas, biasDeltas)
	updates++

	if updates % 600 == 0 {
		loss, accuracy := ps.model.Evaluate(data.Test)
		fmt.Printf("%.4f,%.4f\n", loss, accuracy)
	}


}
