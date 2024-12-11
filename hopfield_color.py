import numpy as np
import torch
from PIL import Image
import os
import time
import matplotlib.pyplot as plt
import random
# from skimage.metrics import structural_similarity as ssim  

def rgb_to_3array(array):
    """
    Returns three 2D numpy arrays corresponding to the RGB channels.
    """
    array = np.array(array)
    R = array[:, :, 0]
    G = array[:, :, 1]
    B = array[:, :, 2]
    return R, G, B

def array_standardize(array: np.ndarray, threshold=145):
    """
    Converts a standard 2D numpy array (values from 0-1) to a {-1, 1} numpy array based on a threshold.
    """
    array = (array * 255).astype(np.int16)
    tmp = np.zeros(array.shape, dtype=np.int16)
    tmp[array >= threshold] = 1
    tmp[array < threshold] = -1
    return tmp

def standarded_array_to_image(array, output_folder=None, i=0, prefix=''):
    """
    Converts a standardized numpy array to an image.
    """
    if output_folder is not None and not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Folder '{output_folder}' created.")
    data = (array * 255).astype(np.uint8)
    img = Image.fromarray(data)
    if output_folder is not None:
        img.save(os.path.join(output_folder, f"{prefix}image_{i + 1}.png"))
    return img

def array_to_vec(array: np.ndarray):
    """
    Converts a 2D numpy array to a vector.
    """
    vec = array.reshape(-1)
    return vec

def vec_to_array(vec: np.ndarray, shape):
    """
    Converts a vector back to a 2D numpy array with the specified shape.
    """
    array = vec.reshape(shape).astype(np.int16)
    return array

def create_W(array):
    """
    Computes the weight matrix from a one-dimensional vector (using outer product).
    """
    w = torch.from_numpy(array)
    w = w.outer(w)
    # Typically, Hopfield nets set diagonal to zero to avoid trivial self-feedback
    w.fill_diagonal_(0)
    return w


def calculate_energy(w, y):
    """
    Calculate Hopfield energy.
    E = -0.5 * y^T W y
    """
    s_vec = torch.from_numpy(y.astype(np.float32))
    E = -0.5 * torch.dot(s_vec, torch.mv(w, s_vec))
    return E.item()

def synchronous_update(w, y_vec, theta=0.5, max_iter=100):
    """
    Synchronous update. All neurons are updated at once each iteration.
    Returns the final state vector and the list of energy values.
    """
    w = w.float()
    y = torch.from_numpy(y_vec.astype(np.float32))
    y = y.reshape(-1, 1)
    energy_list = []

    for i in range(max_iter):
        s_vec = y.reshape(-1)
        # Compute energy
        E = -0.5 * torch.dot(s_vec, torch.mv(w, s_vec))
        energy_list.append(E.item())

        # Update rule (synchronous)
        y_new = torch.mm(w, y) - theta
        y_new = torch.sign(y_new)
        y_new[y_new == 0] = 1  # Handle zero elements

        # Check for convergence
        if torch.equal(y_new, y):
            print(f"Synchronous update converged after {i+1} iterations.")
            break
        y = y_new

    # Final energy calculation
    s_vec = y.reshape(-1)
    E = -0.5 * torch.dot(s_vec, torch.mv(w, s_vec))
    energy_list.append(E.item())

    return y.numpy().reshape(-1), energy_list

def asynchronous_update(w, y_vec, theta=0.5, max_iter=12000):
    """
    Asynchronous update. Neurons are updated one at a time in random order.
    Returns the final state vector and the list of energy values after each update that changes the state.
    """
    w = w.float()
    y = y_vec.copy()
    n_neurons = len(y)
    energy_list = []
    changed = True
    iteration = 0

    # We attempt a certain number of updates (max_iter).
    # Each iteration updates a single randomly chosen neuron.
    # If we go through n_neurons updates without any change, we consider it converged.
    updates_since_change = 0

    # Record initial energy
    energy_list.append(calculate_energy(w, y))

    while iteration < max_iter and updates_since_change < n_neurons:
        i = random.randint(0, n_neurons - 1)
        input_i = torch.dot(w[i], torch.from_numpy(y.astype(np.float32))) - theta
        new_state = 1 if input_i.item() >= 0 else -1

        if new_state != y[i]:
            y[i] = new_state
            energy_list.append(calculate_energy(w, y))
            updates_since_change = 0
        else:
            updates_since_change += 1

        iteration += 1

    if updates_since_change >= n_neurons:
        print(f"Asynchronous update converged after {iteration} single-neuron updates.")
    else:
        print(f"Asynchronous update stopped after max_iter={max_iter} updates (not fully converged).")

    return y, energy_list

def calculate_mse(original, reconstructed):
    """
    Calculate the Mean Squared Error (MSE) between the original image and the reconstructed image.
    """
    return np.mean((original.astype(np.float32) - reconstructed.astype(np.float32)) ** 2)

def calculate_psnr(original, reconstructed, max_pixel=255.0):
    """
    Calculate the Peak Signal-to-Noise Ratio (PSNR) between the original image and the reconstructed image.
    """
    mse = calculate_mse(original, reconstructed)
    if mse == 0:
        return float('inf')
    return 10 * np.log10((max_pixel ** 2) / mse)


def output_array_to_rgb(R, G, B):
    """
    Combines the R, G, B channels into a single RGB image array.
    """
    R[R == 1] = 1
    R[R == -1] = 0
    G[G == 1] = 1
    G[G == -1] = 0
    B[B == 1] = 1
    B[B == -1] = 0
    rgb = np.stack((R, G, B), axis=-1)
    return rgb

def train_test(train_path, test_path, res_path, theta=0.5, threshold=60):
    # Read image data
    print("Importing images...")
    x = np.load(train_path)
    y = np.load(test_path)
    count = 0

    if not os.path.exists(res_path):
        os.makedirs(res_path)
        
    mse_sync_list = []
    psnr_sync_list = []
    mse_async_list = []
    psnr_async_list = []

    for pic, noise_pic in zip(x, y):
        print(f"\nProcessing Image {count+1}...")

        # Process RGB channels
        R, G, B = rgb_to_3array(pic)
        R_noise, G_noise, B_noise = rgb_to_3array(noise_pic)
        pic_shape = R.shape
        R, G, B = array_standardize(R, threshold=threshold), array_standardize(G, threshold=threshold), array_standardize(B, threshold=threshold)
        R_noise, G_noise, B_noise = array_standardize(R_noise, threshold=threshold), array_standardize(G_noise, threshold=threshold), array_standardize(B_noise, threshold=threshold)
        R_vec, G_vec, B_vec = array_to_vec(R), array_to_vec(G), array_to_vec(B)
        R_vec_noise, G_vec_noise, B_vec_noise = array_to_vec(R_noise), array_to_vec(G_noise), array_to_vec(B_noise)

        # Create weight matrices
        print("Creating weight matrices...")
        start_time = time.time()
        w_R, w_G, w_B = create_W(R_vec), create_W(G_vec), create_W(B_vec)
        end_time = time.time()
        print("Weight matrices created. Time taken: %fs" % (end_time - start_time))

        # Perform synchronous updates
        print("Starting synchronous updates...")
        R_vec_after_sync, energy_R_sync = synchronous_update(w=w_R, y_vec=R_vec_noise, theta=theta)
        G_vec_after_sync, energy_G_sync = synchronous_update(w=w_G, y_vec=G_vec_noise, theta=theta)
        B_vec_after_sync, energy_B_sync = synchronous_update(w=w_B, y_vec=B_vec_noise, theta=theta)
        print("Synchronous updates completed.")

        # Perform asynchronous updates (from the same noisy input)
        print("Starting asynchronous updates...")
        R_vec_after_async, energy_R_async = asynchronous_update(w=w_R, y_vec=R_vec_noise, theta=theta)
        G_vec_after_async, energy_G_async = asynchronous_update(w=w_G, y_vec=G_vec_noise, theta=theta)
        B_vec_after_async, energy_B_async = asynchronous_update(w=w_B, y_vec=B_vec_noise, theta=theta)
        print("Asynchronous updates completed.")

        # Convert vectors back to arrays for synchronous
        R_sync_arr = vec_to_array(R_vec_after_sync, shape=pic_shape)
        G_sync_arr = vec_to_array(G_vec_after_sync, shape=pic_shape)
        B_sync_arr = vec_to_array(B_vec_after_sync, shape=pic_shape)

        # Convert vectors back to arrays for asynchronous
        R_async_arr = vec_to_array(R_vec_after_async, shape=pic_shape)
        G_async_arr = vec_to_array(G_vec_after_async, shape=pic_shape)
        B_async_arr = vec_to_array(B_vec_after_async, shape=pic_shape)

        # Generate and save the reconstructed image for synchronous
        sync_img = standarded_array_to_image(output_array_to_rgb(R_sync_arr, G_sync_arr, B_sync_arr), 
                                             output_folder=res_path, i=count, prefix='sync_')
        # Generate and save the reconstructed image for asynchronous
        async_img = standarded_array_to_image(output_array_to_rgb(R_async_arr, G_async_arr, B_async_arr), 
                                              output_folder=res_path, i=count, prefix='async_')

        # Show images (optional)
        # sync_img.show()
        # async_img.show()
        
        # Convert the reconstructed image into a numpy array for the calculation of quality metrics.
        sync_img_np = np.array(sync_img)
        async_img_np = np.array(async_img)

        # Create a binary image of the original clear image.
        clean_rgb = output_array_to_rgb(R, G, B)
        clean_rgb_255 = (clean_rgb).astype(np.uint8) * 255

        # Calculate the quality metrics for synchronous updates.
        mse_sync = calculate_mse(clean_rgb_255, sync_img_np)
        psnr_sync = calculate_psnr(clean_rgb_255, sync_img_np)


        # Calculate the quality metrics for asynchronous updates.
        mse_async = calculate_mse(clean_rgb_255, async_img_np)
        psnr_async = calculate_psnr(clean_rgb_255, async_img_np)
        
        # Store the quality metrics.
        mse_sync_list.append(mse_sync)
        psnr_sync_list.append(psnr_sync)
        mse_async_list.append(mse_async)
        psnr_async_list.append(psnr_async)


        print(f"同步更新 MSE: {mse_sync:.2f}, PSNR: {psnr_sync:.2f} dB")
        print(f"异步更新 MSE: {mse_async:.2f}, PSNR: {psnr_async:.2f} dB")

        # # Plot the energy landscape for synchronous vs asynchronous
        # plt.figure(figsize=(10, 6))
        
        # # Synchronous energy: typically recorded per iteration
        # plt.plot(range(len(energy_R_sync)), energy_R_sync, label='Sync R', color='red', linestyle='--')
        # plt.plot(range(len(energy_G_sync)), energy_G_sync, label='Sync G', color='green', linestyle='--')
        # plt.plot(range(len(energy_B_sync)), energy_B_sync, label='Sync B', color='blue', linestyle='--')

        # # Asynchronous energy: recorded each time a neuron changes state
        # # We only plot 20 iterations
        # plt.plot(range(len(energy_R_async)), energy_R_async, label='Async R', color='red')
        # plt.plot(range(len(energy_G_async)), energy_G_async, label='Async G', color='green')
        # plt.plot(range(len(energy_B_async)), energy_B_async, label='Async B', color='blue')

        # plt.xlabel('Updates/Iterations')
        # plt.ylabel('Energy')
        # plt.title(f'Energy Landscape Comparison for Image {count+1}')
        # plt.legend()
        # plt.grid(True, linestyle='--', alpha=0.7)

        # Create a figure with two subplots (1 row, 2 columns)
        fig, axs = plt.subplots(1, 2, figsize=(20, 6), sharey=False)

        # ------------------ Synchronous Energy Plot ------------------
        # Synchronous energy: typically recorded per iteration
        axs[0].plot(range(len(energy_R_sync)), energy_R_sync, label='Sync R', color='red', linestyle='--')
        axs[0].plot(range(len(energy_G_sync)), energy_G_sync, label='Sync G', color='green', linestyle='--')
        axs[0].plot(range(len(energy_B_sync)), energy_B_sync, label='Sync B', color='blue', linestyle='--')

        axs[0].set_xlabel('Iterations')
        axs[0].set_ylabel('Energy')
        axs[0].set_title(f'Synchronous Energy Landscape for Image {count+1}')
        axs[0].legend()
        axs[0].grid(True, linestyle='--', alpha=0.7)

        # ------------------ Asynchronous Energy Plot ------------------
        # Asynchronous energy: recorded each time a neuron changes state
        # We only plot 20 iterations
        axs[1].plot(range(len(energy_R_async)), energy_R_async, label='Async R', color='red')
        axs[1].plot(range(len(energy_G_async)), energy_G_async, label='Async G', color='green')
        axs[1].plot(range(len(energy_B_async)), energy_B_async, label='Async B', color='blue')

        axs[1].set_xlabel('Updates')
        axs[1].set_ylabel('Energy')
        axs[1].set_title(f'Asynchronous Energy Landscape for Image {count+1}')
        axs[1].legend()
        axs[1].grid(True, linestyle='--', alpha=0.7)

        # Adjust layout for better spacing
        plt.tight_layout()
        # plt.show()

        energy_plot_path = os.path.join(res_path, f'energy_compare_{count+1}.png')
        plt.savefig(energy_plot_path, dpi=300)
        plt.close()
        print(f"Energy comparison plot saved to {energy_plot_path}")
        
        

        count += 1
    # Calculate and output the average quality metrics.
    avg_mse_sync = np.mean(mse_sync_list)
    avg_psnr_sync = np.mean(psnr_sync_list)
    avg_mse_async = np.mean(mse_async_list)
    avg_psnr_async = np.mean(psnr_async_list)

    print("\n== 性能指标 ==")
    print(f"同步更新 平均 MSE: {avg_mse_sync:.2f}")
    print(f"同步更新 平均 PSNR: {avg_psnr_sync:.2f} dB")
    print(f"异步更新 平均 MSE: {avg_mse_async:.2f}")
    print(f"异步更新 平均 PSNR: {avg_psnr_async:.2f} dB")

    # Create a bar chart comparing MSE.
    plt.figure(figsize=(8, 6))
    methods = ['Synchronous', 'Asynchronous']
    mse_values = [avg_mse_sync, avg_mse_async]
    plt.bar(methods, mse_values, color=['skyblue', 'salmon'])
    plt.ylabel('Mean MSE')
    plt.title('Synchronous and Asynchronous Mean MSE Compare')
    for i, v in enumerate(mse_values):
        plt.text(i, v + 0.5, f"{v:.2f}", ha='center', va='bottom')
    mse_plot_path = os.path.join(res_path, 'mse_comparison.png')
    plt.savefig(mse_plot_path, dpi=300)
    plt.close()
    print(f"MSE对比图已保存至 {mse_plot_path}")

    # Create a bar chart comparing PSNR.
    plt.figure(figsize=(8, 6))
    psnr_values = [avg_psnr_sync, avg_psnr_async]
    plt.bar(methods, psnr_values, color=['skyblue', 'salmon'])
    plt.ylabel('Mean PSNR (dB)')
    plt.title('Synchronous and Asynchronous Mean PSNR Compare')
    for i, v in enumerate(psnr_values):
        plt.text(i, v + 0.5, f"{v:.2f}", ha='center', va='bottom')
    psnr_plot_path = os.path.join(res_path, 'psnr_comparison.png')
    plt.savefig(psnr_plot_path, dpi=300)
    plt.close()
    print(f"PSNR对比图已保存至 {psnr_plot_path}")
    print("\nAll images processed.")


# Example usage (make sure paths are correct for your environment):
train_paths = './train_pics/processed_numpy_dataset/kodak24.npy'
test_paths = './test_pics/processed_numpy_test/kodak24_random.npy'
res_paths = './res_pics/kodak24/random_block'

if __name__ == "__main__":
    # Hopfield network starts!
    train_test(train_path=train_paths, test_path=test_paths, res_path=res_paths, theta=0.5, threshold=120)

