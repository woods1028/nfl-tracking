#%%

#%%

import numpy as np
from lime import lime_tabular

# 1. First, create a function that transforms flattened data back to your hierarchical format
def prepare_hierarchical_format(flattened_data, input_shapes):
    """
    Transforms flattened feature vectors back into hierarchical format.
    
    Args:
        flattened_data: 2D array where each row is a flattened sample
        input_shapes: List of shapes for each hierarchical input
        
    Returns:
        List of tensors in the structure expected by your model
    """
    if len(flattened_data.shape) == 1:  # Handle single sample
        flattened_data = flattened_data.reshape(1, -1)
    
    # Unpack the shapes to calculate correct indices
    samples = flattened_data.shape[0]
    hierarchical_data = [np.zeros((samples,) + shape) for shape in input_shapes]
    
    start_idx = 0
    for i, shape in enumerate(input_shapes):
        # Calculate number of elements in this input tensor
        flat_size = np.prod(shape)
        
        # For each sample, reshape the corresponding slice and store
        for j in range(samples):
            hierarchical_data[i][j] = flattened_data[j, start_idx:start_idx+flat_size].reshape(shape)
        
        start_idx += flat_size
    
    # If only one sample, return structure for single prediction
    if samples == 1:
        return [x[0] for x in hierarchical_data]
    
    return hierarchical_data

# 2. Define your model wrapper function for LIME
def predict_fn(flattened_data):
    """
    Wrapper function for your model that accepts flattened data from LIME
    and returns predictions in the format LIME expects.
    """
    # Define the shapes of your hierarchical inputs based on your data
    input_shapes = [(50, 14), (51, 42), (51, 36), (3,)]
    
    # Transform flattened data back to hierarchical format
    hierarchical_data = prepare_hierarchical_format(flattened_data, input_shapes)
    
    # Get predictions (ensure output is 2D for classification problems)
    predictions = hierarchy_model_full(hierarchical_data).numpy()
    
    # LIME expects probability for each class, so for binary classification:
    if predictions.shape[-1] == 1:  # If single output (binary)
        return np.hstack([1-predictions, predictions])  # Return [P(0), P(1)]
    else:
        return predictions  # For multi-class, return as is

# 3. Prepare your data for LIME
# Flatten your background data to create training data for LIME
def flatten_dataset(hierarchical_dataset):
    """Flatten hierarchical data to 2D array for LIME"""
    # Get number of samples from first input tensor
    n_samples = hierarchical_dataset[0].shape[0]
    
    # Create flattened dataset
    flattened_data = []
    for i in range(n_samples):
        # Extract and flatten each sample
        flat_sample = np.concatenate([tensor[i].flatten() for tensor in hierarchical_dataset])
        flattened_data.append(flat_sample)
    
    return np.array(flattened_data)
