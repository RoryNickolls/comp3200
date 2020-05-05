package network

import (
	"fmt"
	"math"

	"github.com/petar/GoMNIST"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat"
)

func rawImageToInputVector(img GoMNIST.RawImage) *mat.VecDense {

	pixels := len(img)

	var data []float64 = make([]float64, pixels)

	for i := 0; i < pixels; i++ {
		data[i] = float64(img[i]) / 255
	}

	return mat.NewVecDense(pixels, data)
}

// LoadData uses GoMNIST to load MNIST data from the data/ folder
func LoadData() *Data {

	// Load data from files
	train, test, err := GoMNIST.Load("./data")
	if err != nil {
		fmt.Println("ERR:", err)
	}

	// Create training records
	var trainRecords []Record
	for i := 0; i < train.Count(); i++ {
		img, label := train.Get(i)
		trainRecords = append(trainRecords, NewRecord(*rawImageToInputVector(img), int(label)))
	}

	// Create testing records
	var testRecords []Record
	for i := 0; i < test.Count(); i++ {
		img, label := test.Get(i)
		testRecords = append(testRecords, NewRecord(*rawImageToInputVector(img), int(label)))
	}

	// Return data
	return &Data{trainRecords, testRecords}
}

// Softmax returns the softmax of an input vector
func Softmax(vec *mat.VecDense) *mat.VecDense {
	new := mat.NewVecDense(vec.Len(), nil)

	max := mat.Max(vec)

	sum := 0.0
	for i := 0; i < vec.Len(); i++ {
		sum += math.Exp(vec.AtVec(i) - max)
	}

	for i := 0; i < vec.Len(); i++ {
		new.SetVec(i, math.Exp(vec.AtVec(i)-max)/sum)
	}

	return new
}

// Sig calculates the sigmoid function for a given value
func Sig(v float64) float64 {
	return 1.0 / (1.0 + math.Exp(-v))
}

// SigPrime calculates the derivative of the sigmoid function for a given value
func SigPrime(v float64) float64 {
	return v * (1.0 - v)
}

// Cost calculates MSE cost for an output and target value
func Cost(activation, target float64) float64 {
	return math.Pow(activation-target, 2.0)
}

// CostVec calculates total MSE cost for an entire output and target vector
func CostVec(activation, target *mat.VecDense) float64 {
	sum := 0.0
	for i := 0; i < activation.Len(); i++ {
		sum += Cost(activation.AtVec(i), target.AtVec(i))
	}
	return sum
}

// DeltaCost returns the derivative of MSE cost
func DeltaCost(activation, target float64) float64 {
	return 2.0 * (target - activation)
}

// DeltaCrossEntropy returns the derivative of the cross-entropy cost function (unused as neural network used a specific optimisation)
func DeltaCrossEntropy(activation, target *mat.VecDense) *mat.VecDense {
	newVec := mat.NewVecDense(activation.Len(), nil)
	newVec.SubVec(target, activation)
	newVec.ScaleVec(1.0/float64(newVec.Len()), newVec)
	return newVec
}

// CrossEntropy calculates the cross-entropy value of an output and target vector
func CrossEntropy(activation, target *mat.VecDense) float64 {
	val := stat.CrossEntropy(target.RawVector().Data, activation.RawVector().Data)
	return val
}

// ApplyVec applies a function to each element of a vector
func ApplyVec(vec *mat.VecDense, f func(float64) float64) *mat.VecDense {
	newVec := mat.NewVecDense(vec.Len(), nil)
	for i := 0; i < newVec.Len(); i++ {
		newVec.SetVec(i, f(vec.At(i, 0)))
	}

	return newVec
}

// FlattenMatrix takes a matrix and flattens it into a single vector
// Used for flattening raw images into input vectors
func FlattenMatrix(matrix *mat.Dense) *mat.VecDense {
	r, c := matrix.Dims()
	length := r * c
	return mat.NewVecDense(length, matrix.RawMatrix().Data)
}

// Norm returns the norm of a vector
func Norm(vec *mat.VecDense) float64 {
	sum := 0.0
	for i := 0; i < vec.Len(); i++ {
		sum += math.Pow(vec.AtVec(i), 2)
	}
	return math.Sqrt(sum)
}
