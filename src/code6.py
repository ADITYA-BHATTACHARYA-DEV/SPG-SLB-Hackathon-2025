import numpy as np

# Use a raw string (r'') for the path to handle backslashes correctly
npz_file_name = r'F:\Amazon ML challenge\SPG\.venv\subject_seismic.npy'
csv_file_name = 'seismic_data_output.csv'  # The name of the new CSV file

arr=np.load(npz_file_name)
print(arr.shape)

# try:
#     # 1. Load the array from the .npz file
#     with np.load(npz_file_name) as data:
#         seismic_data = data['arr_0']
#
#     print(f"✅ Successfully loaded the array 'arr_0' from '{npz_file_name}'.")
#     print(f"Array Shape: {seismic_data.shape}")
#     print(f"Data Type: {seismic_data.dtype}")
#
#     # --- 2. Save the array to CSV format ---
#
#     # Choose a format string based on the data type:
#     if seismic_data.dtype.kind in ('i', 'u'):  # 'i' for integer, 'u' for unsigned integer
#         # Format for integers (e.g., no decimal points)
#         fmt = '%d'
#     else:
#         # Default format for floats (e.g., 6 decimal places)
#         fmt = '%.6f'
#
#     # Use numpy.savetxt to write the array to the CSV file
#     np.savetxt(
#         csv_file_name,  # Name of the output file
#         seismic_data,  # The NumPy array to save
#         delimiter=',',  # Use comma as the separator for CSV
#         fmt=fmt  # Format string for data elements
#     )
#
#     print(f"\n✅ Successfully saved the data to CSV file: '{csv_file_name}'")
#     print("You can find the output file in the same directory as this script.")
#
# except FileNotFoundError:
#     print(f"❌ Error: The file '{npz_file_name}' was not found. Please check the path.")
# except Exception as e:
#     print(f"❌ An error occurred during loading or saving: {e}")