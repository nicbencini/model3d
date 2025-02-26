# Model3d

## Overview
This library provides a `Vector3d` class for performing 3D vector operations using NumPy. It supports fundamental vector calculations such as dot product, cross product, magnitude, normalization, and parallelism checks.

## Features
- **Basic Vector Operations**: Dot product, cross product, magnitude, and unit vector computation.
- **Component Access**: Access and modify `x`, `y`, and `z` components directly.
- **Predefined Unit Vectors**: Unit vectors along the `x`, `y`, and `z` axes.
- **Vector Comparisons**: Check equality and parallelism of vectors.
- **Orthogonalization**: Gram-Schmidt process for generating orthogonal vectors.

## Installation
```bash
pip install numpy  # Ensure NumPy is installed
```

## Usage
```python
import numpy as np
from vector3d import Vector3d

# Create a new vector
v1 = Vector3d([1, 2, 3])
v2 = Vector3d([4, 5, 6])

# Compute magnitude and unit vector
print(v1.magnitude())  # Output: Magnitude of v1
print(v1.unit())       # Output: Unit vector of v1

# Perform dot and cross products
dot = v1.dot_product(v2)
cross = v1.cross_product(v2)

print(dot)   # Output: Dot product of v1 and v2
print(cross) # Output: Cross product of v1 and v2

# Check parallelism
print(v1.is_parallel_to(v2))

# Access and modify components
print(v1.x, v1.y, v1.z)
v1.x = 10  # Update x component
print(v1)
```

## License
This project is licensed under the MIT License.

## Contributions
Contributions are welcome! Feel free to open issues and submit pull requests.
