package main

import (
	"github.com/esadakcam/neural-net/internal/dataset"
)

func main() {
	ds, err := dataset.InitDataset()
	if err != nil {
		panic(err)
	}

	dataset.PrintImage(ds.TrainData[31], ds.TrainLabel[31])
}
