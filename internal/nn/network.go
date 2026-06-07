package nn

import (
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"os"
)

const (
	INPUT_SIZE  = 28 * 28
	OUTPUT_SIZE = 10
	HIDDEN_SIZE = 16
)

type Network struct {
	W1 [][]float64
	B1 []float64
	W2 [][]float64
	B2 []float64
}

type weightsFile struct {
	InputSize  int         `json:"input_size"`
	HiddenSize int         `json:"hidden_size"`
	OutputSize int         `json:"output_size"`
	W1         [][]float64 `json:"w1"`
	B1         []float64   `json:"b1"`
	W2         [][]float64 `json:"w2"`
	B2         []float64   `json:"b2"`
}

type forwardPass struct {
	input []float64
	a1    []float64
	yHat  []float64
}

func New() *Network {
	return &Network{
		W1: initMatrix(HIDDEN_SIZE, INPUT_SIZE),
		B1: make([]float64, HIDDEN_SIZE),
		W2: initMatrix(OUTPUT_SIZE, HIDDEN_SIZE),
		B2: make([]float64, OUTPUT_SIZE),
	}
}

func (n *Network) SaveWeights(path string) error {
	file, err := os.Create(path)
	if err != nil {
		return err
	}
	defer file.Close()

	encoder := json.NewEncoder(file)
	encoder.SetIndent("", "  ")

	return encoder.Encode(weightsFile{
		InputSize:  INPUT_SIZE,
		HiddenSize: HIDDEN_SIZE,
		OutputSize: OUTPUT_SIZE,
		W1:         n.W1,
		B1:         n.B1,
		W2:         n.W2,
		B2:         n.B2,
	})
}

func (n *Network) LoadWeights(path string) error {
	file, err := os.Open(path)
	if err != nil {
		return err
	}
	defer file.Close()

	var weights weightsFile
	if err := json.NewDecoder(file).Decode(&weights); err != nil {
		return err
	}

	if err := validateWeights(weights); err != nil {
		return err
	}

	n.W1 = weights.W1
	n.B1 = weights.B1
	n.W2 = weights.W2
	n.B2 = weights.B2

	return nil
}

func (n *Network) ForwardPropagation(input []float64) ([OUTPUT_SIZE]float64, error) {
	pass, err := n.forward(input)
	if err != nil {
		return [OUTPUT_SIZE]float64{}, err
	}

	return [OUTPUT_SIZE]float64(pass.yHat), nil
}

func (n *Network) BackwardPropagation(input []float64, label uint8, learningRate float64) (float64, error) {
	if label >= OUTPUT_SIZE {
		return 0, fmt.Errorf("Expected label between 0 and %d, got %d", OUTPUT_SIZE-1, label)
	}
	if learningRate <= 0 {
		return 0, fmt.Errorf("learning rate must be positive")
	}

	pass, err := n.forward(input)
	if err != nil {
		return 0, err
	}

	loss := -math.Log(math.Max(pass.yHat[label], 1e-15))

	dz2 := make([]float64, OUTPUT_SIZE)
	for i := range dz2 {
		dz2[i] = pass.yHat[i]
	}
	dz2[label] -= 1

	dz1 := make([]float64, HIDDEN_SIZE)
	for hidden := range HIDDEN_SIZE {
		var weightedError float64
		for output := range OUTPUT_SIZE {
			weightedError += n.W2[output][hidden] * dz2[output]
		}
		dz1[hidden] = weightedError * sigmoidDerivative(pass.a1[hidden])
	}

	for output := range OUTPUT_SIZE {
		for hidden := range HIDDEN_SIZE {
			n.W2[output][hidden] -= learningRate * dz2[output] * pass.a1[hidden]
		}
		n.B2[output] -= learningRate * dz2[output]
	}

	for hidden := range HIDDEN_SIZE {
		for inputIndex := range INPUT_SIZE {
			n.W1[hidden][inputIndex] -= learningRate * dz1[hidden] * pass.input[inputIndex]
		}
		n.B1[hidden] -= learningRate * dz1[hidden]
	}

	return loss, nil
}

func (n *Network) Predict(input []float64) (uint8, error) {
	output, err := n.ForwardPropagation(input)
	if err != nil {
		return 0, err
	}

	bestIndex := 0
	bestValue := output[0]
	for i := 1; i < OUTPUT_SIZE; i++ {
		if output[i] > bestValue {
			bestIndex = i
			bestValue = output[i]
		}
	}

	return uint8(bestIndex), nil
}

func (n *Network) forward(input []float64) (*forwardPass, error) {
	if len(input) != INPUT_SIZE {
		return nil, fmt.Errorf("Expected size: %d, got size %d", INPUT_SIZE, len(input))
	}

	w1x := matmul(n.W1, input)
	z1 := vectorsum(w1x, n.B1)
	a1 := sigmoid(z1)
	w2a1 := matmul(n.W2, a1)
	z2 := vectorsum(w2a1, n.B2)
	y_hat := softmax(z2)

	return &forwardPass{
		input: input,
		a1:    a1,
		yHat:  y_hat,
	}, nil
}

func randomFloat64(cols int) float64 {
	limit := math.Sqrt(6.0 / float64(cols+1))
	return rand.Float64()*2*limit - limit
}

func initMatrix(rows int, cols int) [][]float64 {
	w := make([][]float64, rows)
	for i := range rows {
		w[i] = make([]float64, cols)
		for j := range cols {
			w[i][j] = randomFloat64(cols)
		}
	}
	return w
}

func validateWeights(weights weightsFile) error {
	if weights.InputSize != INPUT_SIZE || weights.HiddenSize != HIDDEN_SIZE || weights.OutputSize != OUTPUT_SIZE {
		return fmt.Errorf(
			"weight architecture mismatch: expected %d-%d-%d, got %d-%d-%d",
			INPUT_SIZE,
			HIDDEN_SIZE,
			OUTPUT_SIZE,
			weights.InputSize,
			weights.HiddenSize,
			weights.OutputSize,
		)
	}
	if err := validateMatrix("w1", weights.W1, HIDDEN_SIZE, INPUT_SIZE); err != nil {
		return err
	}
	if err := validateVector("b1", weights.B1, HIDDEN_SIZE); err != nil {
		return err
	}
	if err := validateMatrix("w2", weights.W2, OUTPUT_SIZE, HIDDEN_SIZE); err != nil {
		return err
	}
	if err := validateVector("b2", weights.B2, OUTPUT_SIZE); err != nil {
		return err
	}
	return nil
}

func validateMatrix(name string, matrix [][]float64, rows int, cols int) error {
	if len(matrix) != rows {
		return fmt.Errorf("%s row count mismatch: expected %d, got %d", name, rows, len(matrix))
	}
	for rowIndex, row := range matrix {
		if len(row) != cols {
			return fmt.Errorf("%s column count mismatch at row %d: expected %d, got %d", name, rowIndex, cols, len(row))
		}
	}
	return nil
}

func validateVector(name string, vector []float64, size int) error {
	if len(vector) != size {
		return fmt.Errorf("%s size mismatch: expected %d, got %d", name, size, len(vector))
	}
	return nil
}

func matmul(w [][]float64, x []float64) []float64 {
	if len(w[0]) != len(x) {
		panic("Mismatched vector size in matmul")
	}
	output := make([]float64, len(w))
	var sum float64
	for rowIndex, row := range w {
		sum = 0
		for colIndex, col := range row {
			sum += col * x[colIndex]
		}
		output[rowIndex] = sum
	}
	return output
}

func vectorsum(v1 []float64, v2 []float64) []float64 {
	if len(v1) != len(v2) {
		panic("Mismatched vector size in vectorsum")
	}
	output := make([]float64, len(v1))
	for i := range v1 {
		output[i] = v1[i] + v2[i]
	}
	return output
}

func sigmoid(v []float64) []float64 {
	output := make([]float64, len(v))
	for i := range v {
		output[i] = 1 / (1 + math.Exp(-v[i]))
	}
	return output
}

func sigmoidDerivative(sigmoidValue float64) float64 {
	return sigmoidValue * (1 - sigmoidValue)
}

func softmax(v []float64) []float64 {
	if len(v) == 0 {
		return nil
	}

	maxVal := v[0]
	for _, v := range v {
		if v > maxVal {
			maxVal = v
		}
	}

	expVals := make([]float64, len(v))
	var sum float64

	for i, v := range v {
		expVal := math.Exp(v - maxVal)
		expVals[i] = expVal
		sum += expVal
	}

	for i := range expVals {
		expVals[i] /= sum
	}

	return expVals
}
