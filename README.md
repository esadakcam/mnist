To train:

```bash
go run ./cmd -epochs=20 -lr=0.05 -save weights.json

```

To load and test randomly:

```bash
go run ./cmd -load weights.json -epochs=0 -print-random
```

Example output:

```bash
go run ./cmd -load weights.json -epochs=0 -print-random


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
