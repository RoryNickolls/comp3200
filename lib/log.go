package lib

import (
	"fmt"
	"log"
	"os"
	"strconv"
)

// SetupLog sets up the log package to output to a particular file for this process
func SetupLog(name string) {
	filename := "log/" + name + strconv.Itoa(os.Getpid()) + ".log"
	file, err := os.Create(filename)
	if err != nil {
		fmt.Println("Error setting up log file")
	}
	log.SetOutput(file)
}
