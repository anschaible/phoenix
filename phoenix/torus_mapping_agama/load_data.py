import numpy as np

# Path to your generated dataset
#data_path = './data_mapping/phoenix_torus_training_data.npz'
data_path = './data_mapping/phoenix_conditional_torus_data.npz'

try:
    # Load the compressed numpy file
    data = np.load(data_path)
    
    print(f"Successfully loaded: {data_path}")
    print("-" * 40)
    
    # List all the arrays (keys) stored in the file
    print(f"Available keys: {data.files}\n")
    
    # Extract the arrays
    actions = data['actions']
    angles = data['angles']
    phase_space = data['phase_space']
    
    # Print the shapes to confirm dimensions (e.g., N x 3 or N x 6)
    print("Data Shapes:")
    print(f"Actions (J):       {actions.shape}")
    print(f"Angles (theta):    {angles.shape}")
    print(f"Phase Space (x,v): {phase_space.shape}\n")
    
    # Print a quick sample of the first row to sanity-check the values
    print("Sample Output (Row 0):")
    print(f"Actions:     {actions[0]}")
    print(f"Angles:      {angles[0]}")
    print(f"Phase Space: {phase_space[0]}")

except FileNotFoundError:
    print(f"Error: Could not find the file at {data_path}. Check your path!")
except Exception as e:
    print(f"An unexpected error occurred: {e}")