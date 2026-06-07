package main

import (
	"fmt"

	"github.com/esadakcam/neural-net/internal/dataset"
	"github.com/esadakcam/neural-net/internal/nn"
)

func main() {
	ds, err := dataset.InitDataset()
	if err != nil {
		panic(err)
	}

	// dataset.PrintImage(ds.TrainData[31], ds.TrainLabel[31])
	network := nn.New()
	input := dataset.ImageToArray(&ds.TrainData[0])
	fmt.Println(input)
	output, err := network.ForwardPropagation(input)
	if err != nil {
		fmt.Println("error: ", err)
		return
	}
	fmt.Println(output)
}
