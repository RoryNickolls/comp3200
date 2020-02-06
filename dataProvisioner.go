package main

func ProvisionData(addresses []string) {
	data := loadData()
	data.Test = nil

	partitionSize := len(data.Train) / len(addresses)
	for i := 0; i < len(addresses); i++ {
		start := i * partitionSize
		end := start + partitionSize
		partition := Data{Train: data.Train[start:end]}
		sendPartition(addresses[i], partition)
	}
	
}

func sendPartition(address string, partition Data) {
	messenger := Connect(address)
	messenger.SendInterface(partition)
}