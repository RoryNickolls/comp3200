package main

import (
	"fmt"
	"math"
	"math/rand"
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
		nn.layers[j].feed(nn.layers[j-1].activation)
	}
	return nn.OutputLayer().activation
}

func (nn *Network) Evaluate(testData []Record) float64 {

	// Calculate average MSE
	err := 0.0
	for _, record := range testData {
		prediction := nn.Predict(record.data)
		expected := target(record.label)

		prediction.SubVec(expected, prediction)
		for i := 0; i < prediction.Len(); i++ {
			prediction.SetVec(i, math.Pow(prediction.AtVec(i), 2))
		}

		// Average error per class
		err += mat.Sum(prediction) / float64(prediction.Len())

	}

	// Average error over whole train set
	return err / float64(len(testData))
}

type Layer struct {
	in         int
	out        int
	weights    *mat.Dense
	output     *mat.VecDense
	activation *mat.VecDense
}

func NewLayer(in int, out int) Layer {
	weights := makeMatrix(out, in)
	output := mat.NewVecDense(out, nil)
	activation := mat.NewVecDense(out, nil)
	return Layer{in, out, weights, output, activation}
}

func (layer *Layer) feed(input *mat.VecDense) {
	output := mat.NewVecDense(layer.out, nil)
	output.MulVec(layer.weights, input)
	layer.output = output
	layer.activation = applyVec(layer.output, sig)
}

func makeMatrix(rows, cols int) *mat.Dense {
	rand.Seed(0)
	elements := rows * cols

	normal := distuv.Normal { Mu: 0.0, Sigma: 1, }

	var data []float64 = make([]float64, elements)
	for i := 0; i < elements; i++ {
		data[i] = normal.Rand()
	}

	return mat.NewDense(rows, cols, data)
}

func sig(v float64) float64 {
	return 1.0 / (1.0 + math.Exp(-v))
}

func sigPrime(v float64) float64 {
	return v * (1 - v)
}

func cost(output, target float64) float64 {
	return math.Pow(target-output, 2)
}

func deltaCost(output, target float64) float64 {
	return 2 * (target - output)
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

	network := NewNetwork().WithLayer(NewLayer(784, 300)).WithLayer(NewLayer(300, 100)).WithLayer(NewLayer(100, 10))
	epochs := 50
	eta := 0.01
	for i := 0; i < epochs; i++ {
		fmt.Println("Epoch", i, "Loss", network.Evaluate(data.test))
		for r := 0; r < len(data.train); r++ {
			record := data.train[r]

			// Forward propagation
			network.Predict(record.data)

			// Get target prediction
			target := target(record.label)

			// Initialise delta vectors for each layers' out neurons
			var deltas []*mat.VecDense
			for j := 0; j < len(network.layers); j++ {
				deltas = append(deltas, mat.NewVecDense(network.layers[j].out, nil))
			}

			// Find delta for each neuron
			for j := len(network.layers) - 1; j >= 0; j-- {
				layer := network.layers[j]

				err := mat.NewVecDense(layer.out, nil)
				// This is the output layer
				if j == len(network.layers)-1 {

					// Calculate cost of output layer
					for k := 0; k < layer.out; k++ {
						err.SetVec(k, cost(layer.activation.AtVec(k), target.AtVec(k)))
					}
				} else {
					// Otherwise this is a hidden layer
					nextLayer := network.layers[j+1]

					// Calculate matrix of backwards propagated errors
					errorMat := mat.NewDense(layer.out, 1, nil)
					errorMat.Mul(nextLayer.weights.T(), deltas[j+1])
					for k := 0; k < layer.out; k++ {
						err.SetVec(k, errorMat.At(k, 0))
					}
				}
				
				for k := 0; k < layer.out; k++ {
					deltas[j].SetVec(k, err.AtVec(k)*sigPrime(err.AtVec(k)))
				}
			}

			// Update each weight according to deltas

			// For each layer
			for j := 0; j < len(network.layers); j++ {
				layer := network.layers[j]

				// input := record.data
				// if j != 0 {
				// 	input = network.layers[j-1].activation
				// }

				// For each neuron in that layer
				for k := 0; k < deltas[j].Len(); k++ {
					for l := 0; l < layer.out; l++ {
						oldWeight := layer.weights.At(k, l)
						update := eta * deltas[j].AtVec(k)
						layer.weights.Set(k, l, oldWeight-update)
					}
				}
			}
		}
	}
}
