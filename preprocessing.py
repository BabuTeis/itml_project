from sklearn.datasets import fetch_species_distributions
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
data = fetch_species_distributions()

import matplotlib.pyplot as plt

# all the data
species = data.train['species']
latitudes = data.train['dd lat']
longitudes = data.train['dd long']

# Plot only one of the 10 subgroups (for now at least) to visualize
new_species = [name for name in set(species) if name == b'bradypus_variegatus_0' or name == b'microryzomys_minutus_0']
print(new_species)
plt.figure(figsize=(10, 6))
for species_name in set(new_species):
    mask = species == species_name
    plt.scatter(longitudes[mask], latitudes[mask], label=species_name, s=10)

plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Species Distribution - Training Data")
plt.legend()
plt.grid(True)
plt.show()


import matplotlib.pyplot as plt

# Makes a map for specific coverage value that you can select and filter not given values out (-9999)
layer = data.coverages[3]
masked_layer = np.ma.masked_where(layer == -9999, layer)

plt.figure(figsize=(10, 6))
plt.imshow(masked_layer, cmap='viridis', interpolation='nearest')
plt.colorbar(label='Environmental Feature Value')
plt.title('Environmental Feature Layer 0')
plt.show()


#####################
### Preprocessing ###
#####################

"""
First, change to only use the two species examples that end with 0 (for now only maybe)
"""

species_train = data.train['species']
species_test = data.test['species']


exclusive_species = [b'microryzomys_minutus_0', b'bradypus_variegatus_0']

train_filter = np.isin(species_train, exclusive_species)
filtered_train = data.train[train_filter]

test_filter = np.isin(species_test, exclusive_species)
filtered_test = data.test[test_filter]

print("\nFiltered Training Data Example:")
print(filtered_train[:5])

print("\nFiltered Test Data Example:")
print(filtered_test[:5])


"""
Then deal with the missing data in coverages.
"""

coverages = data.coverages

# Mask missing data for coverages
masked_coverages = np.ma.masked_where(coverages == -9999, coverages)

print("Before:")
print(coverages[1, :, :])

print("After:")
print(masked_coverages[1, :, :])

from sklearn.preprocessing import MinMaxScaler

# scaler we use because data is not normally distributed!!
scaler = MinMaxScaler()

# template filled w -9999
normalized_coverages = np.full(masked_coverages.shape, -9999, dtype=float)

# Iterates through each layer
for i in range(14):
    layer = masked_coverages[i]

    # Only normalize valid (non-masked) data
    valid_mask = ~layer.mask
    valid_values = layer[valid_mask]  # Extract valid values

    # Scale valid values
    scaled_values = scaler.fit_transform(valid_values.reshape(-1, 1)).flatten()

    # Reintegrate into the layer with same dimensions
    normalized_layer = np.full(layer.shape, -9999, dtype=float)
    normalized_layer[valid_mask] = scaled_values

    # Save back to normalized_coverages
    normalized_coverages[i] = normalized_layer

print("\nNormalized Coverages Example:")
print(normalized_coverages[0, 500:505, 500:505])


from sklearn.preprocessing import LabelEncoder

# Encodes species as 0 or 1
le = LabelEncoder()
train_species = le.fit_transform(filtered_train['species'])
test_species = le.transform(filtered_test['species'])

print("\nSpecies Encoding Example:")
print("Original Species Names:", filtered_train['species'][80:90])
print("Encoded Species Labels:", train_species[80:90])
