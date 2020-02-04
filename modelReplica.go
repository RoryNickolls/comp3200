package main

import (
	"net"
	"fmt"
	"bufio"
	"encoding/gob"
)

func LaunchModelReplica(address string, dataAddress string) {
	conn, err := net.Dial("tcp4", dataAddress)
	if err != nil {
		fmt.Println("ERR:", err)
	}

	c := make(chan Data)
	go receiveData(conn, c)

	writer := bufio.NewWriter(conn)
	writer.WriteString("REQ\n")
	writer.Flush()

	// Wait to receive data back
	<- c
	fmt.Println("Received data from data server")
}

func receiveData(conn net.Conn, c chan Data) {
	fmt.Println("Waiting for data")
	var data Data
	err := gob.NewDecoder(conn).Decode(&data)
	if err != nil {
		fmt.Println("ERR:", err)
	}
	c <- data
}