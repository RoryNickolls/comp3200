package network

import (
	"fmt"
	"math/rand"
	"sync"
	"time"

	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat/distuv"
)

type NetworkConfig struct {
	LearningRate float64
	LayerConfigs []LayerConfig
}

type LayerConfig struct {
	In         int
	Out        int
	Activation string
}

type Network struct {
	Config NetworkConfig
	layers []layer
	mutex  sync.Mutex
}

func NewNetwork() *Network {
	return &Network{Config: NetworkConfig{LearningRate: 0.01}}
}

func NewNetworkFromConfig(config NetworkConfig) *Network {
	network := NewNetwork()
	for _, layerConfig := range config.LayerConfigs {
		network = network.WithLayer(layerConfig.In, layerConfig.Out, layerConfig.Activation)
	}
	network = network.WithLearningRate(config.LearningRate)
	return network
}

func (nn *Network) WithLayer(in int, out int, activation string) *Network {
	layerConfig := LayerConfig{in, out, activation}
	nn.layers = append(nn.layers, newLayerFromConfig(layerConfig))
	nn.Config.LayerConfigs = append(nn.Config.LayerConfigs, layerConfig)
	return nn
}

func (nn *Network) WithLearningRate(eta float64) *Network {
	nn.Config.LearningRate = eta
	return nn
}

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

func (nn *Network) Parameters() ([]mat.Dense, []mat.VecDense) {
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

	return weights, biases
}

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

func (nn *Network) Predict(input *mat.VecDense) *mat.VecDense {
	nn.layers[0].feed(input)
	for j := 1; j < len(nn.layers); j++ {
		nn.layers[j].feed(nn.layers[j-1].output)
	}
	return nn.outputLayer().output
}

func (nn *Network) UpdateWithDeltas(weightDeltas []mat.Dense, biasDeltas []mat.VecDense) {
	nn.mutex.Lock()
	// Update each layers weights with deltas
	for j := 0; j < len(nn.layers); j++ {
		layer := nn.layers[j]
		weightDeltas[j].Scale(nn.Config.LearningRate, &weightDeltas[j])
		biasDeltas[j].ScaleVec(nn.Config.LearningRate, &biasDeltas[j])
		layer.weights.Add(layer.weights, &weightDeltas[j])
		layer.biases.AddVec(layer.biases, &biasDeltas[j])
	}
	nn.mutex.Unlock()
}

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
		nn.Predict(&record.Data)

		// Get target prediction
		target := record.Expected

		dEdI := mat.NewVecDense(nn.outputLayer().out, nil)

		// Find delta for each neuron
		for j := len(nn.layers) - 1; j > 0; j-- {
			layer := nn.layers[j]
			prevLayer := nn.layers[j-1]

			deltaW := mat.NewDense(layer.out, layer.in, nil)
			deltaB := mat.NewVecDense(layer.out, nil)

			// This is the output layer
			if j == len(nn.layers)-1 {

				cross := DeltaCrossEntropy(layer.output, &target)

				// K is index of each neuron in this layer
				for k := 0; k < layer.out; k++ {
					//output := layer.output.AtVec(k)

					// Cost with respect to output
					//dEdO := DeltaCost(output, target.AtVec(k))

					// Output with respect to input
					//dOdI := SigPrime(output)

					// Save some values for the next layer
					dEdI.SetVec(k, cross.AtVec(k))
					//dEdI.SetVec(k, dEdO*dOdI)

					// L is index of each neuron connected behind this one
					for l := 0; l < layer.in; l++ {

						// Input with respect to weight
						dIdW := prevLayer.output.AtVec(l)

						// Combine derivatives using chain rule to get cost function with respect to weight
						//dEdW := dEdO * dOdI * dIdW
						dEdW := cross.AtVec(k) * dIdW

						deltaW.Set(k, l, dEdW)
					}

					dEdB := cross.AtVec(k)
					//dEdB := dEdO * dEdI
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

						dIdW := prevLayer.output.AtVec(l)
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

func (nn *Network) TrainAndUpdate(trainData []Record) ([]mat.Dense, []mat.VecDense) {
	weightDeltas, biasDeltas := nn.Train(trainData)
	nn.UpdateWithDeltas(weightDeltas, biasDeltas)
	return weightDeltas, biasDeltas
}

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

func (nn *Network) ContinuousEvaluation(testData []Record, out chan float64) {
	for {
		loss, accuracy := nn.Evaluate(testData)
		out <- loss
		out <- accuracy
		time.Sleep(30000 * time.Millisecond)
	}
}

func (nn *Network) GradientCheck(eps float64) float64 {

	data := LoadData()
	nn.TrainAndUpdate(data.Train[:10000])

	record := data.Train[rand.Intn(len(data.Train))]
	analyticalWeights, analyticalBiases := nn.Train([]Record{record})

	weights, biases := nn.Parameters()

	sum := 0.0
	count := 0
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
		diffW.Scale(1.0/2.0*eps, &diffW)
		sum += Norm(FlattenMatrix(&diffW))
		count++

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
		diffB.ScaleVec(1.0/2.0*eps, &diffB)
		sum += Norm(&diffB)
		count++
	}

	return sum / float64(count)
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
