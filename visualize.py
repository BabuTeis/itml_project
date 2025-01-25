import numpy as np
import matplotlib.pyplot as plt
from preprocessing import preprocess_data

def visual_plots(number: int) -> None:
    """
    Generates two visual plots:
    1. A scatter plot showing the distribution of two species (0 for bradypus_variegatus and 1 for microryzomys_minutus) in the training data.
    2. A heatmap of an environmental feature layer, which is selected by the user.

    Args:
        number (int): The index of the environmental feature layer.
    """
    # loads the dataset
    preprocessed_data = preprocess_data()

    train_data = preprocessed_data["train_data"]
    coverages = preprocessed_data["coverages"]

    # Extracts species, latitudes, and longitudes from the preprocessed data
    species = train_data["species"]
    latitudes = train_data["dd lat"]
    longitudes = train_data["dd long"]


    # Plot the preprocessed data of the two species
    new_species = [name for name in set(species) if name == 0 or name == 1]
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


    # Make a map for specific coverage value that you can select
    layer = number
    masked_layer = np.ma.masked_where(coverages[layer] == -9999, coverages[layer])


    plt.figure(figsize=(10, 6))
    plt.imshow(masked_layer, cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Environmental Feature Value')
    plt.title(f'Environmental Feature Layer {number}')
    plt.show()
