package messenger

import (
	"comp3200/lib"
	"encoding/gob"
	"fmt"
	"log"
	"net"
	"reflect"
	"strings"
	"time"
)

// Messenger is a struct that represents a two-way connection between system entities
type Messenger struct {
	enc *gob.Encoder
	dec *gob.Decoder
}

var received int
var sent int

// Connect connects this messenger to another messenger on an IPv4 address
func Connect(address string) Messenger {
	conn, err := net.Dial("tcp4", address)
	if err != nil {
		fmt.Println("Error creating messenger. Could not connect to ", address, err)
		panic(1)
	}
	return NewMessenger(conn)
}

// NewMessenger creates a new messenger using a TCP connection
func NewMessenger(conn net.Conn) Messenger {
	return Messenger{gob.NewEncoder(conn), gob.NewDecoder(conn)}
}

// ReceiveInterface instructs this messenger that it will receive a serialized interface
func (m *Messenger) ReceiveInterface(v interface{}) {
	SimulateLatency()
	err := m.dec.Decode(v)
	if err != nil {
		fmt.Println("Error receiving interface", err)
		panic(err)
	}
	logReceiveMessage(reflect.TypeOf(v).String())
}

// ReceiveMessage instructs this messenger that it will receive a command string
func (m *Messenger) ReceiveMessage(cmd *string) {
	var temp string
	SimulateLatency()
	err := m.dec.Decode(&temp)
	if err != nil {
		fmt.Println("Error receiving command", err)
		panic(err)
	}
	*cmd = strings.TrimSpace(temp)

	received++
	logReceiveMessage(*cmd)
}

// SendInterface instructs this messenger to send an interface
func (m *Messenger) SendInterface(v interface{}) {
	err := m.enc.Encode(v)
	if err != nil {
		fmt.Println("Error sending interface", err)
		panic(err)
	}
	logSendMessage(reflect.TypeOf(v).String())
}

// SendMessage instructs this messenger to send a command
func (m *Messenger) SendMessage(msg string) {
	err := m.enc.Encode(msg)
	if err != nil {
		fmt.Println("Error sending command", err)
		panic(err)
	}
	sent++
	logSendMessage(msg)
}

// SimulateLatency causes the current thread to sleep for the amount specified by latency
func SimulateLatency() {
	if lib.SimulatingLatency {
		time.Sleep(lib.Latency * time.Millisecond)
	}
}

var loggingMessages bool

func logSendMessage(msg string) {
	if loggingMessages {
		log.Println("tx", msg)
	}
}

func logReceiveMessage(msg string) {
	if loggingMessages {
		log.Println("rx", msg)
	}
}

// StartLoggingMessages enables message logging
func StartLoggingMessages() {
	loggingMessages = true
}

// TakeReceived returns the number of received messages and resets it
func TakeReceived() int {
	temp := received
	received = 0
	return temp
}

// TakeSent returns the number of sent messages and resets it
func TakeSent() int {
	temp := sent
	sent = 0
	return temp
}

// Received returns the number of received messages
func Received() int {
	return received
}

// Sent returns the number of sent messages
func Sent() int {
	return sent
}
