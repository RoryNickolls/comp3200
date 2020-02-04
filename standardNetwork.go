package main

import (
	"fmt"
)

func Train() {
	data := loadData()
	fmt.Println("Training network with", len(data.Train), "Training instances and", len(data.Test), "testing instances")

	network := NewNetwork().WithLayer(784, 300, "sigmoid").WithLayer(300, 100, "sigmoid").WithLayer(100, 10, "softmax")
	epochs := 1000
	eta := 0.01
	batchSize := 100

	loss, correct := network.Evaluate(data.Test)
	fmt.Printf("Epoch 00 Loss %.4f Accuracy %.4f\n", loss, correct)
	for i := 0; i < epochs; i++ {
		miniBatches := data.GetMiniBatches(batchSize)
		for j := 0; j < len(miniBatches); j++ {
			network.Train(miniBatches[j], eta)
		}
		loss, correct := network.Evaluate(data.Test)

		fmt.Printf("Epoch %.2d Loss %.4f Accuracy %.4f\n", i+1, loss, correct)
	}
}