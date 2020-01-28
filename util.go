package main

import (
	"github.com/petar/GoMNIST"
	"gonum.org/v1/gonum/mat"
	"fmt"
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
		trainRecords = append(trainRecords, NewRecord(rawImageToInputVector(img), int(label)))
	}

	// Create testing records
	var testRecords []Record
	for i := 0; i < test.Count(); i++ {
		img, label := test.Get(i)
		testRecords = append(testRecords, NewRecord(rawImageToInputVector(img), int(label)))
	}

	// Return data
	return &Data{ trainRecords, testRecords }
}