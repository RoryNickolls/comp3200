package main

import (
	"comp3200/network"
	"fmt"
)

func TrainStandardNetwork() {
	data := network.LoadData()
	fmt.Println("Training network with", len(data.Train), "Training instances and", len(data.Test), "testing instances")

	nn := network.NewNetwork().WithLayer(784, 300, "sigmoid").WithLayer(300, 100, "sigmoid").WithLayer(100, 10, "softmax").WithLearningRate(0.1)
	epochs := 1000
	batchSize := 30

	loss, correct := nn.Evaluate(data.Test)
	fmt.Printf("Epoch 00 Loss %.4f Accuracy %.4f\n", loss, correct)
	for i := 0; i < epochs; i++ {
		miniBatches := data.GetMiniBatches(batchSize)
		for j := 0; j < len(miniBatches); j++ {
			nn.TrainAndUpdate(miniBatches[j])
		}
		loss, correct := nn.Evaluate(data.Test)

		fmt.Printf("ep %.2d l %.4f a %.4f\n", i+1, loss, correct)
	}
}
