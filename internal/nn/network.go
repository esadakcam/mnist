package nn

import (
	"fmt"
	"math"
	"math/rand"
)

const (
	INPUT_SIZE  = 28 * 28
	OUTPUT_SIZE = 10
	HIDDEN_SIZE = 16
)

type Network struct {
	// W1 [HIDDEN_SIZE][INPUT_SIZE]float64
	// B1 [HIDDEN_SIZE]float64
	// W2 [HIDDEN_SIZE][HIDDEN_SIZE]float64
	// B2 [HIDDEN_SIZE]float64
	// W3 [OUTPUT_SIZE][HIDDEN_SIZE]float64
	// B3 [OUTPUT_SIZE]float64
	W1 [][]float64
	B1 []float64
	W2 [][]float64
	B2 []float64
}

func New() *Network {
	return &Network{
		W1: initMatrix(HIDDEN_SIZE, INPUT_SIZE),
		B1: initVector(HIDDEN_SIZE),
		W2: initMatrix(OUTPUT_SIZE, HIDDEN_SIZE),
		B2: initVector(OUTPUT_SIZE),
	}
}

func (n *Network) ForwardPropagation(input []float64) ([OUTPUT_SIZE]float64, error) {
	if len(input) != INPUT_SIZE {
		return [OUTPUT_SIZE]float64{}, fmt.Errorf("Expected size: %d, got size %d", INPUT_SIZE, len(input))
	}

	w1x := matmul(n.W1, input)
	z1 := vectorsum(w1x, n.B1)
	a1 := sigmoid(z1)
	w2a1 := matmul(n.W2, a1)
	z2 := vectorsum(w2a1, n.B2)
	y_hat := softmax(z2)

	return [OUTPUT_SIZE]float64(y_hat), nil
}

func randomFloat64() float64 {
	v := rand.Float64() * math.MaxFloat64
	if rand.Intn(2) == 0 {
		return -v / 10000
	}

	return v / 10000
}

func initMatrix(rows int, cols int) [][]float64 {
	w := make([][]float64, rows)
	for i := range rows {
		w[i] = make([]float64, cols)
		for j := range cols {
			w[i][j] = randomFloat64()
		}
	}
	return w
}

func initVector(size int) []float64 {
	b := make([]float64, size)
	for i := range size {
		b[i] = randomFloat64()
	}
	return b
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
			sum += col + x[colIndex]
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

func softmax(xs []float64) []float64 {
	maxVal := xs[0]
	for _, x := range xs {
		if x > maxVal {
			maxVal = x
		}
	}

	exps := make([]float64, len(xs))
	sum := 0.0

	for i, x := range xs {
		exps[i] = math.Exp(x - maxVal)
		sum += exps[i]
	}

	result := make([]float64, len(xs))
	for i, v := range exps {
		result[i] = v / sum
	}

	return result
}
