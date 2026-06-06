package dataset

import (
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"time"
)

const (
	BASE_URL = "https://storage.googleapis.com/cvdf-datasets/mnist"
)

func downloadDataset(path string) (*Dataset, error) {
	err := os.Mkdir(path, os.ModePerm)
	if err != nil {
		return nil, err
	}

	err = download(fmt.Sprintf("%s/%s", BASE_URL, TRAIN_DATA_FILENAME), fmt.Sprintf("%s/%s", path, TRAIN_DATA_FILENAME))
	if err != nil {
		return nil, err
	}
	err = download(fmt.Sprintf("%s/%s", BASE_URL, TRAIN_LABELS_FILENAME), fmt.Sprintf("%s/%s", path, TRAIN_LABELS_FILENAME))
	if err != nil {
		return nil, err
	}
	err = download(fmt.Sprintf("%s/%s", BASE_URL, TEST_DATA_FILENAME), fmt.Sprintf("%s/%s", path, TEST_DATA_FILENAME))
	if err != nil {
		return nil, err
	}
	err = download(fmt.Sprintf("%s/%s", BASE_URL, TEST_LABELS_FILENAME), fmt.Sprintf("%s/%s", path, TEST_LABELS_FILENAME))
	if err != nil {
		return nil, err
	}

	return nil, nil
}

func download(url string, location string) error {
	client := http.Client{Timeout: 15 * time.Second}

	resp, err := client.Get(url)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		fmt.Printf("[download] url: %s returned %d", url, resp.StatusCode)
		return errors.New("Non 200 response")
	}

	buffer, err := io.ReadAll(resp.Body)
	if err != nil {
		fmt.Printf("[download] error reading body for %s", url)
		return err
	}

	err = os.WriteFile(location, buffer, os.ModePerm)
	if err != nil {
		return err
	}

	return nil
}
