package main

type Data struct {
	train []Record
	test  []Record
}

func (d *Data) getTrainPartitions(n int) []Record {
	return d.train
}