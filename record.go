package main

import "gonum.org/v1/gonum/mat"

type Record struct {
	data     *mat.VecDense
	expected *mat.VecDense
}

func NewRecord(data *mat.VecDense, label int) Record {
	target := mat.NewVecDense(10, nil)
	target.SetVec(label, 1.0)

	return Record{data, target}
}
