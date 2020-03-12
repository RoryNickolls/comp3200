package network

import "gonum.org/v1/gonum/mat"

type Record struct {
	Data     mat.VecDense
	Expected mat.VecDense
}

func NewRecord(data mat.VecDense, label int) Record {
	target := mat.NewVecDense(10, nil)
	target.SetVec(label, 1.0)

	return Record{data, *target}
}

func NewRecordRaw(data mat.VecDense, expected mat.VecDense) Record {
	return Record{data, expected}
}
