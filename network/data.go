package network

import (
	"fmt"
	"math/rand"
)

type Data struct {
	Train []Record
	Test  []Record
}

func (d *Data) GetMiniBatches(batchSize int) [][]Record {

	if len(d.Train)%batchSize != 0 {
		fmt.Println("WARNING: Training data cannot be split evenly into batches of size", batchSize)
	}

	// Shuffle Training data
	rand.Shuffle(len(d.Train), func(i, j int) { d.Train[i], d.Train[j] = d.Train[j], d.Train[i] })

	var miniBatches [][]Record
	for i := 0; i < len(d.Train); i += batchSize {
		miniBatches = append(miniBatches, d.Train[i:i+batchSize])
	}
	return miniBatches
}

func (d *Data) Partition(n int) []Data {
	size := len(d.Train) / n

	var partitions []Data
	for i := 0; i < len(d.Train); i += size {
		partitions = append(partitions, Data{d.Train[i : i+size], nil})
	}

	return partitions
}
