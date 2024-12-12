# Hopfield Network for Color Image Memory: Synchronous vs. Asynchronous Updates

## Features
- **Synchronous and Asynchronous Updates**: Implements both update mechanisms in Hopfield networks to store and retrieve color images.
- **Hebbian Learning**: Constructs weight matrices using the Hebbian learning rule to encode color image patterns.
- **Energy Landscape Analysis**: Tracks and visualizes the network's energy dynamics during the recall process.
- **Image Quality Evaluation**: Assesses reconstruction quality using metrics such as Mean Squared Error (MSE), Peak Signal-to-Noise Ratio (PSNR)
- **Visualization**: Generates visualizations of reconstructed images and energy landscapes to compare the performance of update mechanisms.

## Requirements
- **Python**: Version 3.8 or higher
- **Libraries**:
  - [NumPy](https://numpy.org/) 
  - [PyTorch](https://pytorch.org/) 
  - [Pillow](https://python-pillow.org/) 
  - [Matplotlib](https://matplotlib.org/) 
  - [scikit-image](https://scikit-image.org/) 
  - [Jupyter Notebook](https://jupyter.org/) (optional, for interactive exploration)
 

## Usage
1. **Run the Training and Testing Script**:
    ```bash
    python hopfield_color.py --train_path train_pics/processed_numpy_dataset/kodak24.npy --test_path test_pics/processed_numpy_test/kodak24_random.npy --res_path res_pics/kodak24/random_block
    ```
    
2. **View the Results**:
    - Reconstructed images and energy landscape plots will be saved in the `res_pics/` directory.

