package downpour

import (
	"comp3200/lib/messenger"
	"comp3200/lib/network"
)

func ProvisionData(addresses []string) {
	data := network.LoadData()
	data.Test = nil

	partitionSize := len(data.Train) / len(addresses)
	for i := 0; i < len(addresses); i++ {
		start := i * partitionSize
		end := start + partitionSize
		partition := network.Data{Train: data.Train[start:end]}
		sendPartition(addresses[i], partition)
	}

}

func sendPartition(address string, partition network.Data) {
	messenger := messenger.Connect(address)
	messenger.SendInterface(partition)
}
