package synchronous

import (
	"comp3200/lib/messenger"
	"comp3200/lib/network"
	"fmt"
	"net"
	"sync"

	"gonum.org/v1/gonum/mat"
)

type SynchronousParameterServer struct {
	model            *network.Network
	connectedClients []messenger.Messenger
	clients          int
	accumWeight      []mat.Dense
	accumBias        []mat.VecDense
	accumMutex       sync.Mutex
}

var data *network.Data

func LaunchSynchronousParameterServer(address string, clients int, model *network.Network) {

	data = network.LoadData()

	fmt.Println("Launching parameter server")
	ps := SynchronousParameterServer{model: model, clients: clients}
	ps.newAccumulators()

	l, err := net.Listen("tcp4", address)
	if err != nil {
		fmt.Println("ERR:", err)
		return
	}

	connected := 0

	// Only accept the number of clients specified
	for {
		conn, err := l.Accept()
		if err != nil {
			fmt.Println("ERR:", err)
			return
		}
		msg := messenger.NewMessenger(conn)
		go ps.handleConnection(msg)
		ps.connectedClients = append(ps.connectedClients, msg)
		connected++
	}
}

func (ps *SynchronousParameterServer) newAccumulators() {
	weights, biases := ps.model.Parameters()
	for i := 0; i < len(weights); i++ {
		weights[i].Zero()
		biases[i].Zero()
	}
	ps.accumWeight, ps.accumBias = weights, biases
}

func (ps *SynchronousParameterServer) handleConnection(msg messenger.Messenger) {
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

func (ps *SynchronousParameterServer) handleParameterRequest(msg messenger.Messenger) {
	//fmt.Println("Received request for parameters")
	weights, biases := ps.model.Parameters()

	// send current state of weights and biases
	msg.SendInterface(weights)
	msg.SendInterface(biases)
}

func (ps *SynchronousParameterServer) handleModelRequest(msg messenger.Messenger) {
	msg.SendInterface(ps.model.Config)
}

var updates int
var updateMutex sync.Mutex

var syncUpdates int

func (ps *SynchronousParameterServer) handleParameterUpdate(msg messenger.Messenger) {

	// receive deltas for weights and biases
	var weightDeltas []mat.Dense
	var biasDeltas []mat.VecDense

	msg.ReceiveInterface(&weightDeltas)
	msg.ReceiveInterface(&biasDeltas)

	ps.accumMutex.Lock()
	for i := 0; i < len(weightDeltas); i++ {
		ps.accumWeight[i].Add(&ps.accumWeight[i], &weightDeltas[i])
		ps.accumBias[i].AddVec(&ps.accumBias[i], &biasDeltas[i])
	}
	ps.accumMutex.Unlock()

	updateMutex.Lock()
	updates++
	updateMutex.Unlock()

	syncUpdates++

	// If we have received an update from every client
	if updates >= ps.clients {

		// Update the model, evaluate it and send the continue (CON) signal
		ps.model.UpdateWithDeltas(ps.accumWeight, ps.accumBias)

		ps.newAccumulators()

		updateMutex.Lock()
		updates = 0
		updateMutex.Unlock()

		if syncUpdates%1000 == 0 {
			loss, accuracy := ps.model.Evaluate(data.Test)
			fmt.Printf("%.4f,%.4f\n", loss, accuracy)
		}

		// Send CON signal to all clients
		for _, m := range ps.connectedClients {
			m.SendMessage("CON")
		}
	}
}
