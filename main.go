package main

import (
	"fmt"
)

func main() {
	data := loadData()
	fmt.Println("Training network with", len(data.train), "training instances and", len(data.test), "testing instances")

	network := NewNetwork().WithLayer(784, 300, "sigmoid").WithLayer(300, 100, "sigmoid").WithLayer(100, 10, "softmax")
	epochs := 1000
	eta := 0.01
	batchSize := 100
	for i := 0; i < epochs; i++ {

		miniBatches := data.getMiniBatches(batchSize)
		for j := 0; j < len(miniBatches); j++ {
			network.Train(miniBatches[j], eta)
		}
		loss, correct := network.Evaluate(data.test)

		fmt.Printf("Epoch %.2d Loss %.4f Accuracy %.4f\n", i, loss, correct)
	}
}
