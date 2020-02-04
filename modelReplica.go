package main

import (
	"net"
	"fmt"
	"bufio"
	"encoding/gob"
)

type ModelReplica struct {
	data Data
}

func LaunchModelReplica(address string, dataAddress string, parameterAddress string) {
 
	mr := ModelReplica{}

	conn, err := net.Dial("tcp4", dataAddress)
	if err != nil {
		fmt.Println("ERR:", err)
	}

	c := make(chan Data)
	go mr.getData(conn, c)

	writer := bufio.NewWriter(conn)
	writer.WriteString("REQ\n")
	writer.Flush()

	// Wait to receive data back
	<- c
	fmt.Println("Received data from data server")

	// now perform the training, and update parameter server!!
}

func (mr *ModelReplica) getData(conn net.Conn, c chan Data) {
	fmt.Println("Waiting for data")
	var data Data
	err := gob.NewDecoder(conn).Decode(&data)
	if err != nil {
		fmt.Println("ERR:", err)
	}
	mr.data = data
	c <- data
}