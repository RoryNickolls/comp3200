package messenger

import (
	"encoding/gob"
	"fmt"
	"log"
	"net"
	"reflect"
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
	logReceiveMessage(reflect.TypeOf(v).String())
}

func (m *Messenger) ReceiveMessage(cmd *string) {
	var temp string

	err := m.dec.Decode(&temp)
	if err != nil {
		fmt.Println("Error receiving command", err)
		panic(err)
	}
	*cmd = strings.TrimSpace(temp)

	logReceiveMessage(*cmd)
}

func (m *Messenger) SendInterface(v interface{}) {
	err := m.enc.Encode(v)
	if err != nil {
		fmt.Println("Error sending interface", err)
		panic(err)
	}
	logSendMessage(reflect.TypeOf(v).String())
}

func (m *Messenger) SendMessage(msg string) {
	err := m.enc.Encode(msg)
	if err != nil {
		fmt.Println("Error sending command", err)
		panic(err)
	}
	logSendMessage(msg)
}

var loggingMessages bool

func logSendMessage(msg string) {
	if loggingMessages {
		log.Println("Sent", msg)
	}
}

func logReceiveMessage(msg string) {
	if loggingMessages {
		log.Println("Received", msg)
	}
}

func StartLoggingMessages() {
	loggingMessages = true
}
