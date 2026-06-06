package dataset

import (
	"errors"
	"fmt"
	"os"
)

type Dataset struct {
	Path       string
	TrainData  any
	TrainLabel any
	TestData   any
	TestLabel  any
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
	return nil, nil
}
