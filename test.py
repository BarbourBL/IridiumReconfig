import numpy as np
from numpy import zeros


root_sol = np.random.rand(5, 12)
print(root_sol)


# Get second largest root
second_largest_roots = zeros((5, 5))
for i, root in enumerate(root_sol):
    values = root[[0, 4, 8]]
    sorted_indices = np.argsort(values)
    
    # Get the second largest value's original index
    original_index = [0, 4, 8][sorted_indices[-2]]
    second_largest = root[original_index]
    
    second_largest_roots[i, 0] = second_largest
    second_largest_roots[i, 1:4] = root[original_index:original_index+3]  # Corrected indexing
    second_largest_roots[i, -1] = root[-1]  # Last element remains the same

print(second_largest_roots)