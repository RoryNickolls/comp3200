package main

import (
	"encoding/gob"
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

	ds := DataServer{loadData()}

	// Initially receive all data
	fmt.Println("Waiting to be assigned data partition")
	//ds.receiveData(l)

	// Wait for a model replica to connect
	conn, _ := l.Accept()
	messenger := NewMessenger(conn)
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

func (ds *DataServer) receiveData(l net.Listener) {
	conn, err := l.Accept()
	if err != nil {
		fmt.Println("ERR:", err)
	}

	gob.NewDecoder(conn).Decode(ds.data)
	fmt.Println("Received", ds.data)
}
