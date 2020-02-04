package main

import (
	"encoding/gob"
	"fmt"
	"net"
	"bufio"
	"strings"
)

type DataServer struct {
	data *Data
}

func (ds *DataServer) servePartition(conn net.Conn) {
	fmt.Println("Serving data")
	gob.NewEncoder(conn).Encode(ds.data)
}

func LaunchDataServer(address string) {
	l, err := net.Listen("tcp4", address)

	if err != nil {
		fmt.Println("ERR:", err)
	}

	tempData := loadData()
	tempData.Train = tempData.Train[:10]
	tempData.Test = tempData.Test[:1]
	ds := DataServer{tempData}

	// Initially receive all data
	fmt.Println("Waiting to be assigned data partition")
	//ds.receiveData(l)
	for {
		// Wait for a partition request
		conn := ds.waitForRequest(l)

		// Serve request asynchronously
		go ds.servePartition(conn)
	}
}

func (ds *DataServer) waitForRequest(l net.Listener) net.Conn {
	for {
		conn, err := l.Accept()
		if err != nil {
			fmt.Println("ERR:", err)
		}

		// Read a message from the incoming connection
		msg, err := bufio.NewReader(conn).ReadString('\n')
		if err != nil {
			fmt.Println("ERR:", err)
		}

		// If message reads REQ then exit and serve the partition
		if strings.TrimSpace(msg) == "REQ" {
			fmt.Println("Received data request")
			return conn
		}
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
