# MNIST Neural Network in Go

To refresh my knowledge in deep learning and linear algebra, I built a neural network that classifies a handwritten digit from MNIST dataset.

## Train the model

```bash
go run ./cmd -epochs=20 -lr=0.05 -save weights.json
```

## Load the model and test a random image

```bash
go run ./cmd -load weights.json -epochs=0 -print-random
```

## Example output

```bash
$ go run ./cmd -load weights.json -epochs=0 -print-random


             .++#++
           +########+
         .####.   +##.
         ###.    .###.
        ###.     .####.
        ##+       +###+
        ##+        +##+
       .###.      +##+
        ####+...+###+
         ###########.
        .####++.++###+
       .###+       +##+
       +##.         ###
      .###.         ###.
      .###.         ##+
      .###.        .##+
       +##+      .+###.
        ###+   .+###+.
         +########+
          .####+.




prediction: 8
test accuracy: 94.44%
```
