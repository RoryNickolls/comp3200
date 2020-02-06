package main

import (
	"fmt"
	"math/rand"
	"net"
)

type DataServer struct {
	data *Data
}

func (ds *DataServer) servePartition(messenger Messenger) {
	var randTrain []Record
	tempData := Data{}
	for i := 0; i < 10000; i++ {
		randTrain = append(randTrain, ds.data.Train[int(rand.Float64()*float64(len(ds.data.Train)))])
	}
	tempData.Train = randTrain
	fmt.Println("Serving data")
	messenger.SendInterface(tempData)
}

func LaunchDataServer(address string) {
	l, err := net.Listen("tcp4", address)

	if err != nil {
		fmt.Println("ERR:", err)
	}

	ds := DataServer{}

	// Initially receive all data
	fmt.Println("Waiting to be assigned data partition...")
	var data Data
	conn, err := l.Accept()
	messenger := NewMessenger(conn)
	messenger.ReceiveInterface(&data)
	ds.data = &data
	fmt.Println("Assigned data partition")

	// Wait for a model replica to connect
	fmt.Println("Waiting for model replica...")
	conn, _ = l.Accept()
	messenger = NewMessenger(conn)
	for {
		// Wait for a partition request
		ds.waitForRequest(messenger)

		// Serve request
		ds.servePartition(messenger)
	}
}

func (ds *DataServer) waitForRequest(messenger Messenger) {
	fmt.Println("Waiting for data request")
	var msg string
	messenger.ReceiveMessage(&msg)

	// If message reads REQ then exit and serve the partition
	if msg == "REQ" {
		fmt.Println("Received data request")
	}
}
