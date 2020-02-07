package main

import (
	"fmt"
	"math/rand"
	"net"
	"strconv"
	"strings"
)

type DataServer struct {
	miniBatches [][]Record
	index       int
}

func (ds *DataServer) serveMiniBatches(messenger Messenger, n int) {
	var batches [][]Record
	count := 0
	for count < n {
		index := ds.index + count
		if index >= len(ds.miniBatches) {
			index = 0
			rand.Shuffle(len(ds.miniBatches), func(i, j int) { ds.miniBatches[i], ds.miniBatches[j] = ds.miniBatches[j], ds.miniBatches[i] })
		}
		batches = append(batches, ds.miniBatches[index])
		count++
	}

	// Otherwise serve the minibatches
	fmt.Println("Serving data")
	messenger.SendInterface(batches)

	ds.index += count
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

	ds.miniBatches = data.GetMiniBatches(50)
	fmt.Println("Assigned data partition")

	// Wait for a model replica to connect
	fmt.Println("Waiting for model replica...")
	conn, _ = l.Accept()
	messenger = NewMessenger(conn)
	for {
		// Wait for a partition request telling us how many minibatches to send
		n := ds.waitForRequest(messenger)

		if n > 0 {

			// Serve request
			ds.serveMiniBatches(messenger, n)
		}
	}
}

func (ds *DataServer) waitForRequest(messenger Messenger) int {
	fmt.Println("Waiting for data request")
	var msg string
	messenger.ReceiveMessage(&msg)

	parts := strings.Split(msg, " ")

	// If message reads REQ then exit and serve the partition
	if parts[0] == "REQ" {
		count, _ := strconv.ParseInt(parts[1], 10, 32)
		fmt.Println("Received data request for", count, "batches")
		return int(count)
	}

	return 0
}
