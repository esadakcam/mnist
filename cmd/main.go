package main

import (
	"fmt"

	"github.com/esadakcam/neural-net/internal/dataset"
)

func main() {
	_, err := dataset.InitDataset()
	fmt.Println(err)
}
