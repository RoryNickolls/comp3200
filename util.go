package main

import (
	"fmt"
	"math"

	"github.com/petar/GoMNIST"
	"gonum.org/v1/gonum/mat"
)

func rawImageToInputVector(img GoMNIST.RawImage) *mat.VecDense {

	pixels := len(img)

	var data []float64 = make([]float64, pixels)

	for i := 0; i < pixels; i++ {
		data[i] = float64(img[i]) / 255
	}

	return mat.NewVecDense(pixels, data)
}

func loadData() *Data {

	// Load data from files
	train, test, err := GoMNIST.Load("./data")
	if err != nil {
		fmt.Println("ERR:", err)
	}

	// Create training records
	var trainRecords []Record
	for i := 0; i < train.Count(); i++ {
		img, label := train.Get(i)
		trainRecords = append(trainRecords, NewRecord(*rawImageToInputVector(img), int(label)))
	}

	// Create testing records
	var testRecords []Record
	for i := 0; i < test.Count(); i++ {
		img, label := test.Get(i)
		testRecords = append(testRecords, NewRecord(*rawImageToInputVector(img), int(label)))
	}

	// Return data
	return &Data{trainRecords, testRecords}
}

func softmax(vec *mat.VecDense) *mat.VecDense {
	new := mat.NewVecDense(vec.Len(), nil)

	max := mat.Max(vec)

	sum := 0.0
	for i := 0; i < vec.Len(); i++ {
		sum += math.Exp(vec.AtVec(i) - max)
	}

	// NEED TO APPLY MAX-NORMALIZATION HERE

	for i := 0; i < vec.Len(); i++ {
		new.SetVec(i, math.Exp(vec.AtVec(i) - max)/sum)
	}

	return new
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
