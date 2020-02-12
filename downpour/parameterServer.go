package downpour

import (
	"comp3200/messenger"
	"comp3200/network"
	"fmt"
	"net"

	"gonum.org/v1/gonum/mat"
)

type ParameterServer struct {
	model *network.Network
}

var data *network.Data

func LaunchParameterServer(address string) {

	data = network.LoadData()

	fmt.Println("Launching parameter server")
	ps := ParameterServer{}
	ps.model = network.NewNetwork().WithLayer(784, 300, "sigmoid").WithLayer(300, 100, "sigmoid").WithLayer(100, 10, "softmax").WithLearningRate(0.1)

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
		go ps.handleConnection(messenger.NewMessenger(conn))
	}
}

func (ps *ParameterServer) handleConnection(msg messenger.Messenger) {
	fmt.Println("New model replica connected")
	for {
		var cmd string
		msg.ReceiveMessage(&cmd)

		switch cmd {
		// Requesting parameters
		case "REQ":
			ps.handleParameterRequest(msg)
			break
		case "UPD":
			ps.handleParameterUpdate(msg)
			break
		case "MDL":
			ps.handleModelRequest(msg)
			break
		}
	}
}

func (ps *ParameterServer) handleParameterRequest(msg messenger.Messenger) {
	//fmt.Println("Received request for parameters")
	weights, biases := ps.model.Parameters()

	// send current state of weights and biases
	msg.SendInterface(weights)
	msg.SendInterface(biases)
}

func (ps *ParameterServer) handleModelRequest(msg messenger.Messenger) {
	msg.SendInterface(ps.model.Config)
}

var updates int

func (ps *ParameterServer) handleParameterUpdate(msg messenger.Messenger) {

	// receive deltas for weights and biases
	var weightDeltas []mat.Dense
	var biasDeltas []mat.VecDense

	msg.ReceiveInterface(&weightDeltas)
	msg.ReceiveInterface(&biasDeltas)

	// update master model with deltas
	ps.model.UpdateWithDeltas(weightDeltas, biasDeltas)
	updates++

	if updates%600 == 0 {
		loss, accuracy := ps.model.Evaluate(data.Test)
		fmt.Printf("%.4f,%.4f\n", loss, accuracy)
	}

}
