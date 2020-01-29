package main

import (
	"fmt"
	"math"

	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat/distuv"
)

type Record struct {
	data  *mat.VecDense
	label int
}

func NewRecord(data *mat.VecDense, label int) Record {
	return Record{data, label}
}

func target(label int) *mat.VecDense {
	target := mat.NewVecDense(10, nil)
	target.SetVec(label, 1.0)
	return target
}

type Data struct {
	train []Record
	test  []Record
}

func (d *Data) getTrainPartitions(n int) []Record {
	return d.train
}

type Network struct {
	layers []Layer
}

func NewNetwork() *Network {
	return &Network{}
}

func (nn *Network) WithLayer(layer Layer) *Network {
	nn.layers = append(nn.layers, layer)
	return nn
}

func (nn *Network) OutputLayer() *Layer {
	return &nn.layers[len(nn.layers)-1]
}

func (nn *Network) Predict(input *mat.VecDense) *mat.VecDense {
	nn.layers[0].feed(input)
	for j := 1; j < len(nn.layers); j++ {
		nn.layers[j].feed(nn.layers[j-1].output)
	}
	return nn.OutputLayer().output
}

func softmax(vec *mat.VecDense) *mat.VecDense {
	new := mat.NewVecDense(vec.Len(), nil)

	sum := 0.0
	for i := 0; i < vec.Len(); i++ {
		sum += math.Exp(vec.AtVec(i))
	}

	for i := 0; i < vec.Len(); i++ {
		new.SetVec(i, math.Exp(vec.AtVec(i))/sum)
	}

	return new
}

func (nn *Network) Evaluate(testData []Record) (float64, int) {
	correct := 0

	// Calculate average MSE
	err := 0.0
	for _, record := range testData {
		prediction := nn.Predict(record.data)
		//fmt.Println(prediction)
		expected := target(record.label)
		//fmt.Println(expected)

		max := 0
		for i := 0; i < prediction.Len(); i++ {
			if prediction.AtVec(i) > prediction.AtVec(max) {
				max = i
			}
		}

		if expected.AtVec(max) == 1.0 {
			correct++
		}

		prediction.SubVec(expected, prediction)
		for i := 0; i < prediction.Len(); i++ {
			prediction.SetVec(i, math.Pow(prediction.AtVec(i), 2))
		}

		// Add MSE to total
		err += mat.Sum(prediction) / float64(prediction.Len())

	}

	// Average error over whole train set
	return err / float64(len(testData)), correct
}

type Layer struct {
	in                 int
	out                int
	activationFunction string
	weights            *mat.Dense
	biases             *mat.VecDense
	activation         *mat.VecDense
	output             *mat.VecDense
}

func NewLayer(in int, out int, activationFunction string) Layer {
	weights := initialiseWeights(out, in)
	biases := mat.NewVecDense(out, nil)
	for i := 0; i < out; i++ {
		biases.SetVec(i, 1.0)
	}
	activation := mat.NewVecDense(out, nil)
	output := mat.NewVecDense(out, nil)
	return Layer{in, out, activationFunction, weights, activation, output, biases}
}

func (layer *Layer) feed(input *mat.VecDense) {
	activation := mat.NewVecDense(layer.out, nil)
	activation.MulVec(layer.weights, input)
	//activation.AddVec(activation, layer.biases)
	layer.activation = activation

	if layer.activationFunction == "sigmoid" {
		layer.output = applyVec(layer.activation, sig)
	} else if layer.activationFunction == "softmax" {
		layer.output = softmax(layer.activation) 
	}
}

func initialiseWeights(rows, cols int) *mat.Dense {
	elements := rows * cols

	var data []float64 = make([]float64, elements)
	for i := 0; i < elements; i++ {
		data[i] = distuv.UnitNormal.Rand()
	}

	return mat.NewDense(rows, cols, data)
}

func sig(v float64) float64 {
	return 1.0 / (1.0 + math.Exp(-v))
}

func sigPrime(v float64) float64 {
	return v * (1.0 - v)
}

func cost(activation, target float64) float64 {
	return math.Pow(activation-target, 2.0)
}

func deltaCost(activation, target float64) float64 {
	return 2.0 * (target - activation)
}

func applyVec(vec *mat.VecDense, f func(float64) float64) *mat.VecDense {
	newVec := mat.NewVecDense(vec.Len(), nil)
	for i := 0; i < newVec.Len(); i++ {
		newVec.SetVec(i, f(vec.At(i, 0)))
	}

	return newVec
}

func main() {
	data := loadData()
	fmt.Println(len(data.train), len(data.test))

	network := NewNetwork().WithLayer(NewLayer(784, 300, "sigmoid")).WithLayer(NewLayer(300, 100, "sigmoid")).WithLayer(NewLayer(100, 10, "softmax"))
	epochs := 50
	eta := 0.01
	for i := 0; i < epochs; i++ {
		loss, correct := network.Evaluate(data.test)
		for r := 0; r < 10000; r++ {
			record := data.train[r]

			// Forward propagation
			network.Predict(record.data)

			// Get target prediction
			target := target(record.label)

			// Initialise error deltas
			var deltas []*mat.Dense
			for j := 0; j < len(network.layers); j++ {
				deltas = append(deltas, mat.NewDense(network.layers[j].out, network.layers[j].in, nil))
			}

			dEdI := mat.NewVecDense(network.OutputLayer().out, nil)

			// Find delta for each neuron
			for j := len(network.layers) - 1; j > 0; j-- {
				layer := network.layers[j]
				prevLayer := network.layers[j-1]
				delta := mat.NewDense(layer.out, layer.in, nil)

				// This is the output layer
				if j == len(network.layers)-1 {

					// K is index of each neuron in this layer
					for k := 0; k < layer.out; k++ {

						// Cost with respect to output
						dEdO := deltaCost(layer.output.AtVec(k), target.AtVec(k))

						// Output with respect to input
						dOdI := sigPrime(layer.output.AtVec(k))

						// Save some values for the next layer
						dEdI.SetVec(k, dEdO*dOdI)

						// L is index of each neuron connected behind this one
						for l := 0; l < layer.in; l++ {

							// Input with respect to weight
							dIdW := prevLayer.output.AtVec(l)

							// Combine derivatives using chain rule to get cost function with respect to weight
							dEdW := dEdO * dOdI * dIdW

							delta.Set(k, l, dEdW)
						}
					}

				} else {

					nextLayer := network.layers[j+1]
					dEdINew := mat.NewVecDense(layer.out, nil)
					// This is a hidden layer

					// K is index of each neuron in this layer
					for k := 0; k < layer.out; k++ {

						// Change to a sum of errors because we have multiple output neurons
						dEdO := 0.0
						for m := 0; m < nextLayer.out; m++ {

							// Retrieve saved value and multiply by weight to backpropagate the error
							dEdO += dEdI.AtVec(m) * nextLayer.weights.At(m, k)
						}

						// rest is the same
						dOdI := sigPrime(layer.output.AtVec(k))

						dEdINew.SetVec(k, dEdO*dOdI)

						// L is index of each neuron connected behind this one
						for l := 0; l < layer.in; l++ {

							dIdW := prevLayer.output.AtVec(l)
							dEdW := dEdO * dOdI * dIdW
							delta.Set(k, l, dEdW)
						}
					}

					dEdI = dEdINew
				}

				deltas[j] = delta
			}

			// Update each layers weights with deltas
			for j := 0; j < len(network.layers); j++ {
				layer := network.layers[j]
				deltas[j].Scale(eta, deltas[j])
				layer.weights.Add(layer.weights, deltas[j])
			}
		}

		fmt.Println("Epoch", i, "Loss", loss, "Correct", correct)
	}
}
