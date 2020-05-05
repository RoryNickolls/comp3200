package network

import (
	"fmt"
	"sync"

	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat/distuv"
)

// NetworkConfig is a struct that represents the parameters used by a neural network for efficient synchronisation
type NetworkConfig struct {
	LearningRate float64
	LayerConfigs []LayerConfig
}

// LayerConfig is a struct that represents a single layer configuration of the neural network
type LayerConfig struct {
	In         int
	Out        int
	Activation string
}

// Network is a struct that represents the model of a neural network
type Network struct {
	Config NetworkConfig
	layers []layer
	mutex  sync.Mutex
}

// NewNetwork creates a new neural network
func NewNetwork() *Network {
	return &Network{Config: NetworkConfig{LearningRate: 0.01}}
}

// NewNetworkFromConfig creates a new neural network using a supplied config
func NewNetworkFromConfig(config NetworkConfig) *Network {
	network := NewNetwork()
	for _, layerConfig := range config.LayerConfigs {
		network = network.WithLayer(layerConfig.In, layerConfig.Out, layerConfig.Activation)
	}
	network = network.WithLearningRate(config.LearningRate)
	return network
}

// WithLayer is a chain method for building a network and its config
func (nn *Network) WithLayer(in int, out int, activation string) *Network {
	layerConfig := LayerConfig{in, out, activation}
	nn.layers = append(nn.layers, newLayerFromConfig(layerConfig))
	nn.Config.LayerConfigs = append(nn.Config.LayerConfigs, layerConfig)
	return nn
}

// WithLearningRate is a chain method for setting the learning rate of a network
func (nn *Network) WithLearningRate(eta float64) *Network {
	nn.Config.LearningRate = eta
	return nn
}

// SetParameters overrides the parameters of each layer in this neural network
func (nn *Network) SetParameters(weights []mat.Dense, biases []mat.VecDense) {
	nn.mutex.Lock()
	if len(weights) != len(nn.layers) || len(biases) != len(nn.layers) {
		fmt.Println("Error setting network weights. Not enough weight matrices supplied.")
		return
	}

	for i := 0; i < len(nn.layers); i++ {
		nn.layers[i].weights = &weights[i]
		nn.layers[i].biases = &biases[i]
	}
	nn.mutex.Unlock()
}

// Parameters returns the parameters for each layer of this neural network
func (nn *Network) Parameters() ([]mat.Dense, []mat.VecDense) {
	nn.mutex.Lock()

	var weights []mat.Dense
	var biases []mat.VecDense

	for i := 0; i < len(nn.layers); i++ {
		var w mat.Dense
		w.CloneFrom(nn.layers[i].weights)

		var b mat.VecDense
		b.CloneVec(nn.layers[i].biases)

		weights = append(weights, w)
		biases = append(biases, b)
	}

	nn.mutex.Unlock()

	return weights, biases
}

// ZeroedParameters returns parameter matrices and vectors of the correct dimensions but filled with zero
func (nn *Network) ZeroedParameters() ([]mat.Dense, []mat.VecDense) {
	weights, biases := nn.Parameters()
	for i := 0; i < len(weights); i++ {
		weights[i].Zero()
		biases[i].Zero()
	}

	return weights, biases
}

func (nn *Network) outputLayer() *layer {
	return &nn.layers[len(nn.layers)-1]
}

// Predict feeds an input forward through the network and returns its output
func (nn *Network) Predict(input *mat.VecDense) *mat.VecDense {
	nn.layers[0].feed(input)
	for j := 1; j < len(nn.layers); j++ {
		nn.layers[j].feed(nn.layers[j-1].output)
	}
	return nn.outputLayer().output
}

// UpdateWithDeltas shifts all the parameters by the supplied amounts scaled by the learning rate
func (nn *Network) UpdateWithDeltas(weightDeltas []mat.Dense, biasDeltas []mat.VecDense) {
	nn.mutex.Lock()
	// Update each layers weights with deltas
	for j := 0; j < len(nn.layers); j++ {
		layer := nn.layers[j]

		var w mat.Dense
		w.CloneFrom(&weightDeltas[j])

		var b mat.VecDense
		b.CloneVec(&biasDeltas[j])

		w.Scale(nn.Config.LearningRate, &w)
		b.ScaleVec(nn.Config.LearningRate, &b)

		layer.weights.Sub(layer.weights, &w)
		layer.biases.SubVec(layer.biases, &b)
	}
	nn.mutex.Unlock()
}

// Train returns the gradients that this network should be updated with based on the supplied list of training data records
func (nn *Network) Train(trainData []Record) ([]mat.Dense, []mat.VecDense) {

	// Initialise error deltas
	var weightDeltas []mat.Dense
	var biasDeltas []mat.VecDense
	for j := 0; j < len(nn.layers); j++ {
		weightDeltas = append(weightDeltas, *mat.NewDense(nn.layers[j].out, nn.layers[j].in, nil))
		biasDeltas = append(biasDeltas, *mat.NewVecDense(nn.layers[j].out, nil))
	}

	for r := 0; r < len(trainData); r++ {
		record := trainData[r]

		// Forward propagation
		prediction := nn.Predict(&record.Data)

		// Get target prediction
		target := record.Expected

		dEdI := mat.NewVecDense(nn.outputLayer().out, nil)

		// Find delta for each neuron
		for j := len(nn.layers) - 1; j >= 0; j-- {
			layer := nn.layers[j]

			deltaW := mat.NewDense(layer.out, layer.in, nil)
			deltaB := mat.NewVecDense(layer.out, nil)

			// This is the output layer
			if j == len(nn.layers)-1 {

				// K is index of each neuron in this layer
				for k := 0; k < layer.out; k++ {
					// Cost with respect to output
					dEdO := prediction.AtVec(k) - target.AtVec(k)

					// Save some values for the next layer
					dEdI.SetVec(k, dEdO)

					// L is index of each neuron connected behind this one
					for l := 0; l < layer.in; l++ {

						// Input with respect to weight
						dIdW := nn.layers[j-1].output.AtVec(l)

						// Combine derivatives using chain rule to get cost function with respect to weight
						dEdW := dEdO * dIdW

						deltaW.Set(k, l, dEdW)
					}

					//dEdB := cross.AtVec(k)
					dEdB := dEdO
					deltaB.SetVec(k, dEdB)
				}

			} else {
				// This is a hidden layer

				nextLayer := nn.layers[j+1]
				dEdINew := mat.NewVecDense(layer.out, nil)

				// K is index of each neuron in this layer
				for k := 0; k < layer.out; k++ {
					output := layer.output.AtVec(k)

					// Change to a sum of errors because we have multiple output neurons
					dEdO := 0.0
					for m := 0; m < nextLayer.out; m++ {

						// Retrieve saved value and multiply by weight to backpropagate the error
						dEdO += dEdI.AtVec(m) * nextLayer.weights.At(m, k)
					}

					// rest is the same
					dOdI := SigPrime(output)

					dEdINew.SetVec(k, dEdO*dOdI)

					// L is index of each neuron connected behind this one
					for l := 0; l < layer.in; l++ {
						dIdW := record.Data.AtVec(l)
						if j > 0 {
							dIdW = nn.layers[j-1].output.AtVec(l)
						}
						dEdW := dEdO * dOdI * dIdW
						deltaW.Set(k, l, dEdW)
					}

					dEdB := dEdO * dOdI
					deltaB.SetVec(k, dEdB)
				}

				dEdI = dEdINew
			}

			weightDeltas[j].Add(&weightDeltas[j], deltaW)
			biasDeltas[j].AddVec(&biasDeltas[j], deltaB)
		}
	}

	for j := 0; j < len(nn.layers); j++ {
		weightDeltas[j].Scale(1.0/float64(len(trainData)), &weightDeltas[j])
		biasDeltas[j].ScaleVec(1.0/float64(len(trainData)), &biasDeltas[j])
	}

	return weightDeltas, biasDeltas
}

// TrainAndUpdate trains this network on supplied data and then updates it using the learning rate
func (nn *Network) TrainAndUpdate(trainData []Record) ([]mat.Dense, []mat.VecDense) {
	weightDeltas, biasDeltas := nn.Train(trainData)
	nn.UpdateWithDeltas(weightDeltas, biasDeltas)
	return weightDeltas, biasDeltas
}

// Evaluate returns the average error and average correct predictions of a supplied test set
func (nn *Network) Evaluate(testData []Record) (float64, float64) {
	correct := 0

	// Calculate average MSE
	err := 0.0
	for _, record := range testData {
		prediction := nn.Predict(&record.Data)
		expected := &record.Expected

		cross := CrossEntropy(prediction, expected)
		max := 0
		for i := 0; i < prediction.Len(); i++ {
			if prediction.AtVec(i) > prediction.AtVec(max) {
				max = i
			}
		}

		if expected.AtVec(max) == 1.0 {
			correct++
		}

		err += cross
	}

	// Average error over whole train set
	return err / float64(len(testData)), float64(correct) / float64(len(testData))
}

// GradientCheck trains on a single record and returns the difference in analytical and numerical gradients for the weights and biases
func (nn *Network) GradientCheck(record Record, eps float64) (float64, float64) {
	analyticalWeights, analyticalBiases := nn.Train([]Record{record})
	weights, biases := nn.Parameters()
	weightSum := 0.0
	biasSum := 0.0
	weightCount := 0
	biasCount := 0
	for i := 0; i < len(weights); i++ {
		r, c := weights[i].Dims()
		approxWeights := mat.NewDense(r, c, nil)

		var w mat.Dense
		w.CloneFrom(&weights[i])
		for j := 0; j < r; j++ {
			for k := 0; k < c; k++ {

				v := weights[i].At(j, k)

				// Calculate cost by replacing weight on this layer with w - eps
				w.Set(j, k, v-eps)
				newWeights := append(append(weights[:i], w), weights[i+1:]...)
				nn.SetParameters(newWeights, biases)
				costMinus := CrossEntropy(nn.Predict(&record.Data), &record.Expected)

				// Calculate cost by replacing weight on this layer with w + eps
				w.Set(j, k, v+eps)
				newWeights = append(append(weights[:i], w), weights[i+1:]...)
				nn.SetParameters(newWeights, biases)
				costPlus := CrossEntropy(nn.Predict(&record.Data), &record.Expected)

				// Reset the value and neural network parameters
				w.Set(j, k, v)
				nn.SetParameters(weights, biases)

				// Calculate approximate derivative
				approx := (costPlus - costMinus) / (2 * eps)
				approxWeights.Set(j, k, approx)
			}
		}

		var diffW mat.Dense
		diffW.Sub(&analyticalWeights[i], approxWeights)
		num := Norm(FlattenMatrix(&diffW))
		denom := Norm(FlattenMatrix(approxWeights)) + Norm(FlattenMatrix(&analyticalWeights[i]))
		diff := num / denom
		weightSum += diff
		weightCount++

		var b mat.VecDense
		b.CloneVec(&biases[i])
		approxBiases := mat.NewVecDense(b.Len(), nil)
		for j := 0; j < b.Len(); j++ {
			v := b.AtVec(j)

			// Calculate cost by replacing bias with b - eps
			b.SetVec(j, v-eps)
			newBiases := append(append(biases[:i], b), biases[i+1:]...)
			nn.SetParameters(weights, newBiases)
			costMinus := CrossEntropy(nn.Predict(&record.Data), &record.Expected)

			// Calculate cost by replacing bias with b + eps
			b.SetVec(j, v+eps)
			newBiases = append(append(biases[:i], b), biases[i+1:]...)
			nn.SetParameters(weights, newBiases)
			costPlus := CrossEntropy(nn.Predict(&record.Data), &record.Expected)

			// Reset the value and neural network parameters
			b.SetVec(j, v)
			nn.SetParameters(weights, biases)

			// Calculate approximate derivative
			approx := (costPlus - costMinus) / (2 * eps)
			approxBiases.SetVec(j, approx)
		}

		var diffB mat.VecDense
		diffB.SubVec(&analyticalBiases[i], approxBiases)
		num = Norm(&diffB)
		denom = Norm(approxBiases) + Norm(&analyticalBiases[i])
		diff = num / denom
		biasSum += diff
		biasCount++
	}

	return weightSum / float64(weightCount), biasSum / float64(biasCount)
}

// layer is a struct that represents a single layer of the neural network
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

func newLayerFromConfig(config LayerConfig) layer {
	return newLayer(config.In, config.Out, config.Activation)
}

func (layer *layer) feed(input *mat.VecDense) {
	activation := mat.NewVecDense(layer.out, nil)
	activation.MulVec(layer.weights, input)
	activation.AddVec(activation, layer.biases)
	layer.activation = activation

	if layer.activationFunction == "sigmoid" {
		layer.output = ApplyVec(layer.activation, Sig)
	} else if layer.activationFunction == "softmax" {
		layer.output = Softmax(layer.activation)
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
