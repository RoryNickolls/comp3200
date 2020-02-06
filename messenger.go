package main

import (
	"encoding/gob"
	"fmt"
	"net"
	"strings"
)

type Messenger struct {
	enc *gob.Encoder
	dec *gob.Decoder
}

func Connect(address string) Messenger {
	conn, err := net.Dial("tcp4", address)
	if err != nil {
		fmt.Println("Error creating messenger. Could not connect to ", address, err)
		panic(1)
	}
	return NewMessenger(conn)
}

func NewMessenger(conn net.Conn) Messenger {
	return Messenger{gob.NewEncoder(conn), gob.NewDecoder(conn)}
}

func (m *Messenger) ReceiveInterface(v interface{}) {
	err := m.dec.Decode(v)
	if err != nil {
		fmt.Println("Error receiving interface", err)
		panic(err)
	}
}

func (m *Messenger) ReceiveMessage(cmd *string) {
	var temp string
	
	err := m.dec.Decode(&temp)
	if err != nil {
		fmt.Println("Error receiving command", err)
		panic(err)
	}
	*cmd = strings.TrimSpace(temp)
}

func (m *Messenger) SendInterface(v interface{}) {
	err := m.enc.Encode(v)
	if err != nil {
		fmt.Println("Error sending interface", err)
		panic(err)
	}
}

func (m *Messenger) SendMessage(msg string) {
	err := m.enc.Encode(msg)
	if err != nil {
		fmt.Println("Error sending command", err)
		panic(err)
	}
}
