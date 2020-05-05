package network

import "gonum.org/v1/gonum/mat"

// Record is a struct that holds input data and expected output for a single training / test instance
type Record struct {
	Data     mat.VecDense
	Expected mat.VecDense
}

// NewRecord creates a new record from an input and one-hot encodes the specified label
func NewRecord(data mat.VecDense, label int) Record {
	target := mat.NewVecDense(10, nil)
	target.SetVec(label, 1.0)

	return Record{data, *target}
}

// NewRecordRaw creates a new record from an input and raw output vector
func NewRecordRaw(data mat.VecDense, expected mat.VecDense) Record {
	return Record{data, expected}
}
