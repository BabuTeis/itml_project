from sklearn.datasets import fetch_species_distributions
import numpy as np
import matplotlib.pyplot as plt

# loads the dataset
data = fetch_species_distributions()

# all the data
species = data.train['species']
latitudes = data.train['dd lat']
longitudes = data.train['dd long']

print(species[:5])

# Plot only one of the 10 subgroups (for now at least) to visualize (the one that ends with _0)
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
