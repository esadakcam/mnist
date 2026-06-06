package dataset

import (
	"compress/gzip"
	"encoding/binary"
	"fmt"
	"io"
	"os"
)

const (
	IMAGE_MAGIC_NUMBER = 2051
	LABEL_MAGIC_NUMBER = 2049
)

func readImages(path string, magicNumber uint32) ([]Image, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	greader, err := gzip.NewReader(file)
	if err != nil {
		return nil, err
	}

	defer greader.Close()

	var magic uint32
	var numImages uint32
	var rows uint32
	var cols uint32

	err = binary.Read(greader, binary.BigEndian, &magic)
	if err != nil {
		return nil, err
	}

	if magic != magicNumber {
		return nil, fmt.Errorf("Magic number mismatch. expected %d, read %d", magicNumber, magic)
	}

	if err := binary.Read(greader, binary.BigEndian, &numImages); err != nil {
		return nil, err
	}

	if err := binary.Read(greader, binary.BigEndian, &rows); err != nil {
		return nil, err
	}

	if err := binary.Read(greader, binary.BigEndian, &cols); err != nil {
		return nil, err
	}

	if rows != IMAGE_SHAPE || cols != IMAGE_SHAPE {
		return nil, fmt.Errorf("Image size mismatch expected: %d, recevied rows: %d, cols: %d", IMAGE_SHAPE, rows, cols)
	}

	images := make([]Image, numImages)

	for i := 0; i < int(numImages); i++ {
		image := Image{}

		for r := 0; r < int(rows); r++ {
			row := make([]byte, IMAGE_SHAPE)
			_, err := io.ReadFull(greader, row)
			if err != nil {
				return nil, err
			}
			if len(row) != IMAGE_SHAPE {
				return nil, fmt.Errorf("row size is invalid")
			}

			image[r] = [IMAGE_SHAPE]uint8(row[:IMAGE_SHAPE])
		}

		images[i] = image
	}

	return images, nil
}

func readLabels(path string, magicNumber uint32) ([]Label, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	greader, err := gzip.NewReader(file)
	if err != nil {
		return nil, err
	}
	defer greader.Close()

	var magic uint32
	var numLabels uint32

	err = binary.Read(greader, binary.BigEndian, &magic)
	if err != nil {
		return nil, err
	}
	if magic != magicNumber {
		return nil, fmt.Errorf("Magic number mismatch. expected %d, read %d", magicNumber, magic)
	}

	err = binary.Read(greader, binary.BigEndian, &numLabels)
	if err != nil {
		return nil, err
	}

	labels := make([]Label, numLabels)
	var label Label
	for l := range numLabels {
		err = binary.Read(greader, binary.BigEndian, &label)
		if err != nil {
			return nil, err
		}
		labels[l] = label
	}

	return labels, nil
}
