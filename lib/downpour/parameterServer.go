package downpour

import (
	"comp3200/lib"
	"comp3200/lib/messenger"
	"comp3200/lib/network"
	"log"
	"net"

	"gonum.org/v1/gonum/mat"
)

type ParameterServer struct {
	model *network.Network
}

var data *network.Data

func LaunchParameterServer(address string, model *network.Network, isAsync bool) {
	if isAsync {
		lib.SetupLog("async/parameter")
	} else {
		lib.SetupLog("downpour/parameter")
	}

	data = network.LoadData()

	log.Println("Launching parameter server")
	ps := ParameterServer{model}

	l, err := net.Listen("tcp4", address)
	if err != nil {
		log.Println("ERR:", err)
		return
	}

	for {
		conn, err := l.Accept()
		if err != nil {
			log.Println("ERR:", err)
			return
		}
		go ps.handleConnection(messenger.NewMessenger(conn))
	}
}

func (ps *ParameterServer) handleConnection(msg messenger.Messenger) {
	log.Println("New model replica connected")
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
}
