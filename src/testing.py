import numpy as np
import matplotlib.pyplot as plt

# Load seismic .npy file
data = np.load("F:/Amazon ML challenge/SPG/.venv/Finals_SPGxSLB/training_traces/training_traces/152_iline.npy")

# Inspect basic info
print("Type:", type(data))
print("Shape:", data.shape)
print("Data type:", data.dtype)

# Example: if it's a 2D seismic section (time × trace)
plt.imshow(data, cmap="gray", aspect="auto")
plt.colorbar(label="Amplitude")
plt.title("Seismic Section")
plt.xlabel("Trace")
plt.ylabel("Time sample")
plt.show()