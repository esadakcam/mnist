package main

import (
	"flag"
	"fmt"
	"math/rand"

	"github.com/esadakcam/neural-net/internal/dataset"
	"github.com/esadakcam/neural-net/internal/nn"
)

func main() {
	epochs := flag.Int("epochs", 5, "number of training epochs")
	learningRate := flag.Float64("lr", 0.1, "stochastic gradient descent learning rate")
	trainLimit := flag.Int("train-limit", dataset.TRAIN_DATA_SIZE, "number of training examples to use")
	testLimit := flag.Int("test-limit", dataset.TEST_DATA_SIZE, "number of test examples to evaluate")
	loadPath := flag.String("load", "", "path to a JSON weights file to load before training")
	savePath := flag.String("save", "", "path to write JSON weights after training")
	printRandom := flag.Bool("print-random", false, "print a random test image and prediction after loading weights")
	flag.Parse()

	ds, err := dataset.InitDataset()
	if err != nil {
		panic(err)
	}

	network := nn.New()
	if *loadPath != "" {
		if err := network.LoadWeights(*loadPath); err != nil {
			panic(err)
		}
		fmt.Printf("loaded weights from %s\n", *loadPath)
	}
	if *printRandom && *loadPath == "" {
		panic("-print-random requires -load")
	}

	*trainLimit = clamp(*trainLimit, 1, dataset.TRAIN_DATA_SIZE)
	*testLimit = clamp(*testLimit, 1, dataset.TEST_DATA_SIZE)

	if *printRandom {
		if err := printRandomPrediction(network, ds); err != nil {
			panic(err)
		}
	}

	epochsToRun := *epochs
	if epochsToRun < 0 {
		epochsToRun = 0
	}

	for epoch := 1; epoch <= epochsToRun; epoch++ {
		order := rand.Perm(*trainLimit)
		var totalLoss float64

		for _, index := range order {
			input := dataset.ImageToArray(&ds.TrainData[index])
			loss, err := network.BackwardPropagation(input, ds.TrainLabel[index], *learningRate)
			if err != nil {
				panic(err)
			}
			totalLoss += loss
		}

		accuracy, err := evaluate(network, ds, *testLimit)
		if err != nil {
			panic(err)
		}

		fmt.Printf(
			"epoch %d/%d - loss: %.4f - test accuracy: %.2f%%\n",
			epoch,
			*epochs,
			totalLoss/float64(*trainLimit),
			accuracy*100,
		)
	}

	if *epochs <= 0 {
		accuracy, err := evaluate(network, ds, *testLimit)
		if err != nil {
			panic(err)
		}
		fmt.Printf("test accuracy: %.2f%%\n", accuracy*100)
	}

	if *savePath != "" {
		if err := network.SaveWeights(*savePath); err != nil {
			panic(err)
		}
		fmt.Printf("saved weights to %s\n", *savePath)
	}
}

func printRandomPrediction(network *nn.Network, ds *dataset.Dataset) error {
	index := rand.Intn(dataset.TEST_DATA_SIZE)
	image := ds.TestData[index]
	label := ds.TestLabel[index]
	input := dataset.ImageToArray(&image)

	prediction, err := network.Predict(input)
	if err != nil {
		return err
	}

	dataset.PrintImage(image, label)
	fmt.Printf("prediction: %d\n", prediction)
	return nil
}

func evaluate(network *nn.Network, ds *dataset.Dataset, testLimit int) (float64, error) {
	var correct int

	for i := range testLimit {
		input := dataset.ImageToArray(&ds.TestData[i])
		prediction, err := network.Predict(input)
		if err != nil {
			return 0, err
		}
		if prediction == ds.TestLabel[i] {
			correct++
		}
	}

	return float64(correct) / float64(testLimit), nil
}

func clamp(value int, min int, max int) int {
	if value < min {
		return min
	}
	if value > max {
		return max
	}
	return value
}
