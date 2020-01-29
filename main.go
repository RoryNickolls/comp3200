package main

import (
	"fmt"
)

func main() {
	data := loadData()
	fmt.Println("Training network with", len(data.train), "training instances and", len(data.test), "testing instances")

	network := NewNetwork().WithLayer(784, 300, "sigmoid").WithLayer(300, 100, "sigmoid").WithLayer(100, 10, "softmax")
	epochs := 50
	eta := 0.01
	for i := 0; i < epochs; i++ {
		network.Train(data.train, eta)

		loss, correct := network.Evaluate(data.test)
		fmt.Println("Epoch", i, "Loss", loss, "Correct", correct)
	}
}
