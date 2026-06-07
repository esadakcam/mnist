package dataset

import (
	"errors"
	"fmt"
	"os"
)

const (
	TRAIN_DATA_FILENAME   = "train-images-idx3-ubyte.gz"
	TRAIN_LABELS_FILENAME = "train-labels-idx1-ubyte.gz"
	TRAIN_DATA_SIZE       = 60000

	TEST_DATA_FILENAME   = "t10k-images-idx3-ubyte.gz"
	TEST_LABELS_FILENAME = "t10k-labels-idx1-ubyte.gz"
	TEST_DATA_SIZE       = 10000

	IMAGE_SHAPE = 28
)

type (
	Image = [IMAGE_SHAPE][IMAGE_SHAPE]uint8
	Label = uint8
)

type Dataset struct {
	Path       string
	TrainData  [TRAIN_DATA_SIZE]Image
	TrainLabel [TRAIN_DATA_SIZE]Label
	TestData   [TEST_DATA_SIZE]Image
	TestLabel  [TEST_DATA_SIZE]Label
}

func InitDataset() (*Dataset, error) {
	cwd, err := os.Getwd()
	if err != nil {
		return nil, nil
	}
	path := fmt.Sprintf("%s/%s", cwd, "data")

	_, err = os.Stat(path)

	if err != nil && errors.Is(err, os.ErrNotExist) {
		return downloadDataset(path)
	}

	return initDataset(path)
}

func initDataset(path string) (*Dataset, error) {
	trainImages, err := readImages(fmt.Sprintf("%s/%s", path, TRAIN_DATA_FILENAME), IMAGE_MAGIC_NUMBER)
	if err != nil {
		return nil, err
	}

	if len(trainImages) != TRAIN_DATA_SIZE {
		return nil, fmt.Errorf("Expected size: %d, got: %d", TRAIN_DATA_SIZE, len(trainImages))
	}
	trainLabels, err := readLabels(fmt.Sprintf("%s/%s", path, TRAIN_LABELS_FILENAME), LABEL_MAGIC_NUMBER)
	if err != nil {
		return nil, err
	}
	if len(trainLabels) != TRAIN_DATA_SIZE {
		return nil, fmt.Errorf("Expected size: %d, got: %d", TRAIN_DATA_SIZE, len(trainLabels))
	}
	testImages, err := readImages(fmt.Sprintf("%s/%s", path, TEST_DATA_FILENAME), IMAGE_MAGIC_NUMBER)
	if err != nil {
		return nil, err
	}
	if len(testImages) != TEST_DATA_SIZE {
		return nil, fmt.Errorf("Expected size: %d, got: %d", TEST_DATA_SIZE, len(testImages))
	}
	testLabels, err := readLabels(fmt.Sprintf("%s/%s", path, TEST_LABELS_FILENAME), LABEL_MAGIC_NUMBER)
	if err != nil {
		return nil, err
	}
	if len(testLabels) != TEST_DATA_SIZE {
		return nil, fmt.Errorf("Expected size: %d, got: %d", TEST_DATA_SIZE, len(testLabels))
	}

	return &Dataset{
		Path:       path,
		TrainData:  [TRAIN_DATA_SIZE]Image(trainImages),
		TrainLabel: [TRAIN_DATA_SIZE]Label(trainLabels),
		TestData:   [TEST_DATA_SIZE]Image(testImages),
		TestLabel:  [TEST_DATA_SIZE]Label(testLabels),
	}, nil
}

func PrintImage(image Image, label Label) {
	fmt.Println(label)
	for r := range IMAGE_SHAPE {
		for c := range IMAGE_SHAPE {
			pixel := image[r][c]

			switch {
			case pixel > 200:
				fmt.Print("#")
			case pixel > 120:
				fmt.Print("+")
			case pixel > 50:
				fmt.Print(".")
			default:
				fmt.Print(" ")
			}
		}
		fmt.Println()
	}
}

func ImageToArray(i *Image) []float64 {
	output := make([]float64, IMAGE_SHAPE*IMAGE_SHAPE)
	index := 0
	for _, row := range i {
		for _, col := range row {
			output[index] = float64(col) / 255.0
			index++
		}
	}
	return output
}
