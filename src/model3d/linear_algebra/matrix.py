from pypardiso import spsolve
import scipy.sparse as sp
import numpy as np

# Define a sparse matrix A (CSR format)
A = sp.csr_matrix([
    [4, 1, 0],
    [1, 3, -1],
    [0, -1, 2]
], dtype=np.float64)

# Define the right-hand side vector b
b = np.array([15, 10, 10], dtype=np.float64)

# Solve Ax = b using MKL Pardiso
x = spsolve(A, b)

# Print the solution
print("Solution x:", x)

import numpy as np
import mkl

# Enable maximum MKL threads
mkl.set_num_threads(mkl.get_max_threads())

# Define two matrices
A = np.random.rand(1000, 1000)
B = np.random.rand(1000, 1000)

# Matrix multiplication (MKL BLAS is used)
C = np.matmul(A, B)  # MKL-optimized

print("Matrix multiplication using oneMKL completed!")

import numpy as np
np.__config__.show()

import numpy as np
import time

# Large matrix multiplication test
size = 5000
A = np.random.rand(size, size)
B = np.random.rand(size, size)

start_time = time.time()
C = np.dot(A, B)  # MKL should accelerate this
end_time = time.time()

print(f"Matrix multiplication completed in {end_time - start_time:.4f} seconds")

import mkl
print("MKL Threads:", mkl.get_max_threads())