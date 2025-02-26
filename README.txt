Inside there is the Offical_code folder which contains the code used. The remining folder are scratch codes that were used to understand stuff, I also leave these but those are not commented and some examples might not work properly as intended, as these were experiments.

The notes provided are under the assumption that Neural Networks are understood by the reader and the following was already completed:

Reading/watching:

Upmem's user manual: https://sdk.upmem.com/2023.1.0/

Upmem's SDK with the emulator: https://sdk.upmem.com/

Genomics tutorial: https://www.youtube.com/watch?v=ZtpwhVJdgX8&list=PL_wYZF3FksQeKDMsqzUmYKZ1KO7q5Q_Z_&index=1&t=3095s

Introduction to PIM in UPMEM: https://www.youtube.com/watch?v=9TNdZJYbQcA&list=PL_wYZF3FksQeKDMsqzUmYKZ1KO7q5Q_Z_&index=2

CUDA MLP: https://www.youtube.com/watch?v=gAgZkdTF4KQ&list=PL_wYZF3FksQeKDMsqzUmYKZ1KO7q5Q_Z_&index=3

UPMEM lecture: Juan Gomez-Luna: https://www.youtube.com/watch?v=6dwV_RBjK2c

Another PIM lecture by Juan: https://www.youtube.com/watch?v=D8Hjy2iU9l4 and slides: https://people.inf.ethz.ch/omutlu/pub/PrIM-UPMEM-Tutorial-Analysis-Benchmarking-SAFARI-Live-Seminar-2021-07-12-talk.pdf

Undestanding of the State-of-the-art of PIM. Please watch the lectures provided on Onur Mutlu's youtube channel and Safari group's website:

https://safari.ethz.ch/courses/

https://www.youtube.com/@OnurMutluLectures

Also reading papers is important, I have provided Professor G. Falcao with a .zip with all the literature used to write our conference paper, which should also be read.

Inside each folder, a .txt file with notes is given.

Usually, the complex functions are documented with comments.

Pretty much all the PIM implementations follow the same structure: host.c, Binary.c (call of the PIM functions), something_kernels.c (implementation of the functions)

The kernel functions are the following:


exponential: macro for exponential. For further detail read our paper. note that math.h does not work
Ceil: ceil function. note that math.h does not work
min: min function. note that math.h does not work
max: max function. note that math.h does not work

kMartixByMatrixElementwise: multithreading element-wise matrix multiplication.
kMartixSubstractMatrix: multithreading element-wise matrix subtraction.
kSigmoid: applies sigmoid to a matrix. Multithreading is used
kSigmoid_d: applies sigmoid derivative to a matrix. Multithreading is used
kReLU: ReLu function. Multithreading is used
kDot: Matrix multiplication. Multithreading is used
kDot_m1_m2T: Matrix multiplication where matrix 2 is transposed. Multithreading is used.
kDot_m1T_m2: Matrix multiplication where matrix 1 is transposed. Multithreading is used.
kFit: trains an MLP
kTest: inference on MLP.

Functions that have a _inference: do the exact same but also allow for single-element inference. Should not be used.