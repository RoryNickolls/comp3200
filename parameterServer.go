package main

import (
	"fmt"
	"net"
	"bufio"
	"strings"
)

func LaunchParameterServer(address string) {

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

		msg, err := bufio.NewReader(conn).ReadString('\n')
		if err != nil {
			fmt.Println("ERR:", err)
			return
		}

		switch strings.TrimSpace(msg) {
			// Requesting parameters
		case "REQ":
			go handleParameterRequest(conn)
			break
			// Updating parameters
		case "UPD":
			go handleParameterUpdate(conn)
			break
		}
	}
}

func handleParameterRequest(conn net.Conn) {
	// send current state of weights and biases
}

func handleParameterUpdate(conn net.Conn) {
	// receive deltas for weights and biases
	// update master model
}