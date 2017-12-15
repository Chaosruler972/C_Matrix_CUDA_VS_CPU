# C_Matrix_CUDA_VS_CPU

my way of testing CPU vs GPU on CUDA library

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[![IDE_CLION](https://img.shields.io/badge/IDE-Visual%20studio-green.svg)](https://www.visualstudio.com/)

## Install and run:
	Using visual Studio solution manager
	
## Requirements!
* CUDA enabled GPU with more than 1.2GB of VRAM available for max case of 10,000*10,000 matrix multipication calculation (all 3)

Matrix calculaion is normally o(n^3) to (at best) o(n^2.3) at best on single thread.

# Hardware:
	GTX 970M 6GB VRAM GDDR5
	i7 6700HQ (mobile) 800MHz until 3.5GHz 4c\8t (FSB 100 multi 8-35) 4x32KB L1-data (8-way) 4x 32KB L1-instruction (8-way) 4x256KB L2 (4-way) 1X6MB L3 (12-way) 
	DDR4 16GB Dual Channel CL14 NB frequency 2.6GHz DRAM frequency 1.2GHz FSB:DRAM 1:18 (XMP 2)
	O.S: Page size: 4096

# Matrixes dimension is 100*100, numbers are random between 0 and 1000
 Host (CPU Multithreadded!) calculated matrix multipication at size of 100X100 at 1.12e-06 Seconds
 Device (GPU) calculated matrix multipication at size of 100X100 at 0.0182179 Seconds


And both matrixes are  Equal, therefore correct


 The ratio between GPU and CPU (multi_threadded) (GPU/CPU Multi-threadded) is  16265.9
Press any key to continue . . .

# Matrixes dimension is 1000*1000, numbers are random between 0 and 1000
Host (CPU Multithreadded!) calculated matrix multipication at size of 1000X1000 at 8 Seconds
 Device (GPU) calculated matrix multipication at size of 1000X1000 at 0.709721 Seconds


And both matrixes are  Equal, therefore correct


 The ratio between GPU and CPU (multi_threadded) (GPU/CPU Multi-threadded) is  0.0887152
Press any key to continue . . .

 
# Matrixes dimension is 1500*1500, numbers are random between 0 and 1000
 Host (CPU Multithreadded!) calculated matrix multipication at size of 1500X1500 at 34 Seconds
 Device (GPU) calculated matrix multipication at size of 1500X1500 at 0.383153 Seconds


And both matrixes are  Unequal, therefore (most likely GPU) is incorrect


 The ratio between GPU and CPU (multi_threadded) (GPU/CPU Multi-threadded) is  0.0112692
Press any key to continue . . .

# Conclusions

	The bigger the input, the better the usage of CUDA cores
	on small inputs, usage of CUDA core is a hassle due to memory transfer
