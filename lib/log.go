package lib

import (
	"fmt"
	"log"
	"os"
	"strconv"
)

func SetupLog(name string) {
	filename := "log/" + name + strconv.Itoa(os.Getpid()) + ".log"
	file, err := os.Create(filename)
	if err != nil {
		fmt.Println("Error setting up log file")
	}
	log.SetOutput(file)
}
