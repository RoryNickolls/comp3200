package network

import (
	"fmt"
)

func TrainStandardNetwork() {
	data := LoadData()
	fmt.Println("Training network with", len(data.Train), "Training instances and", len(data.Test), "testing instances")

	// Create model
	nn := NewNetwork().WithLayer(784, 300, "sigmoid").WithLayer(300, 100, "sigmoid").WithLayer(100, 10, "softmax").WithLearningRate(0.1)

	epochs := 1000
	batchSize := 10

	// Initial evaluation of (random) model
	loss, accuracy := nn.Evaluate(data.Test)
	printResult(0, loss, accuracy)

	for i := 0; i < epochs; i++ {

		// Train over all mini-batches in each epoch
		miniBatches := data.GetMiniBatches(batchSize)
		for j := 0; j < len(miniBatches); j++ {
			nn.TrainAndUpdate(miniBatches[j])
		}

		// Evaluate the network at the end of each epoch
		loss, accuracy := nn.Evaluate(data.Test)
		printResult(i+1, loss, accuracy)

		// Decay learning rate
		nn.Config.LearningRate *= 0.95
	}
}

func printResult(epoch int, loss float64, accuracy float64) {
	fmt.Printf("%.2d,%.4f,%.4f\n", epoch, loss, accuracy)
}
