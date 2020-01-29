package main

import (
	"math"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat/distuv"
)

type Network struct {
	layers []layer
}

func NewNetwork() *Network {
	return &Network{}
}

func (nn *Network) WithLayer(in int, out int, activation string) *Network {
	nn.layers = append(nn.layers, newLayer(in, out, activation))
	return nn
}

func (nn *Network) OutputLayer() *layer {
	return &nn.layers[len(nn.layers)-1]
}

func (nn *Network) Predict(input *mat.VecDense) *mat.VecDense {
	nn.layers[0].feed(input)
	for j := 1; j < len(nn.layers); j++ {
		nn.layers[j].feed(nn.layers[j-1].output)
	}
	return nn.OutputLayer().output
}

func (nn *Network) Train(trainData []Record, eta float64) {
	for r := 0; r < len(trainData); r++ {
		record := trainData[r]

		// Forward propagation
		nn.Predict(record.data)

		// Get target prediction
		target := record.expected

		// Initialise error deltas
		var deltas []*mat.Dense
		for j := 0; j < len(nn.layers); j++ {
			deltas = append(deltas, mat.NewDense(nn.layers[j].out, nn.layers[j].in, nil))
		}

		dEdI := mat.NewVecDense(nn.OutputLayer().out, nil)

		// Find delta for each neuron
		for j := len(nn.layers) - 1; j > 0; j-- {
			layer := nn.layers[j]
			prevLayer := nn.layers[j-1]
			delta := mat.NewDense(layer.out, layer.in, nil)

			// This is the output layer
			if j == len(nn.layers)-1 {

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

				nextLayer := nn.layers[j+1]
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
		for j := 0; j < len(nn.layers); j++ {
			layer := nn.layers[j]
			deltas[j].Scale(eta, deltas[j])
			layer.weights.Add(layer.weights, deltas[j])
		}
	}
}

func (nn *Network) Evaluate(testData []Record) (float64, int) {
	correct := 0

	// Calculate average MSE
	err := 0.0
	for _, record := range testData {
		prediction := nn.Predict(record.data)
		expected := record.expected
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

type layer struct {
	in                 int
	out                int
	activationFunction string
	weights            *mat.Dense
	biases             *mat.VecDense
	activation         *mat.VecDense
	output             *mat.VecDense
}

func newLayer(in int, out int, activationFunction string) layer {
	weights := initialiseWeights(out, in)
	biases := mat.NewVecDense(out, nil)
	for i := 0; i < out; i++ {
		biases.SetVec(i, 1.0)
	}
	activation := mat.NewVecDense(out, nil)
	output := mat.NewVecDense(out, nil)
	return layer{in, out, activationFunction, weights, activation, output, biases}
}

func (layer *layer) feed(input *mat.VecDense) {
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
