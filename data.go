package main

import (
	"fmt"
	"math/rand"
)

type Data struct {
	train []Record
	test  []Record
}

func (d *Data) getMiniBatches(batchSize int) [][]Record {

	if len(d.train)%batchSize != 0 {
		fmt.Println("WARNING: Training data cannot be split evenly into batches of size", batchSize)
	}

	// Shuffle training data
	rand.Shuffle(len(d.train), func(i, j int) { d.train[i], d.train[j] = d.train[j], d.train[i] })

	var miniBatches [][]Record
	for i := 0; i < len(d.train); i += batchSize {
		miniBatches = append(miniBatches, d.train[i:i+batchSize])
	}
	return miniBatches
}
