from sklearn.datasets import fetch_species_distributions
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

def simplify_species_name(species: bytes) -> bytes:
    """
    Convert b'microryzomys_minutus_0' -> b'microryzomys_minutus'

    Args:
        species (bytes): The encoded species name with suffix.

    Returns:
        bytes: The simplified species name without suffix.
    """
    name_str = species.decode('utf-8')
    base_str = name_str.rsplit('_', 1)[0]
    return base_str.encode('utf-8')  # or just return base_str (string)

def preprocess_data() -> dict:
    """
    Preprocess the species distribution dataset to:
    - Filter data for specific species.
    - Handle missing data in environmental coverage layers.
    - Normalize coverage values.
    - Encode species labels as 0 or 1.
    - Split test data into validation and test sets.

    Returns:
        dict: A dictionary which has preprocessed training, validation, test data and normalized coverage layers.
    """

    # Loads the dataset
    data = fetch_species_distributions()

    # Simplify species names
    species_train_simplified = np.array([simplify_species_name(s) for s in data.train['species']])
    species_test_simplified = np.array([simplify_species_name(s) for s in data.test['species']])

    # Filter for target species
    real_names = [b'microryzomys_minutus', b'bradypus_variegatus']

    train_filter = np.isin(species_train_simplified, real_names)
    filtered_train_raw = data.train[train_filter]

    test_filter = np.isin(species_test_simplified, real_names)
    filtered_test_raw = data.test[test_filter]

    filtered_train = species_train_simplified[train_filter]
    filtered_test = species_test_simplified[test_filter]


    # Handle missing data in coverages

    coverages = data.coverages

    # Mask missing data for coverages
    masked_coverages = np.ma.masked_where(coverages == -9999, coverages)

    # Normalize all valid coverage data
    # We use this scaler because data is not normally distributed
    scaler = MinMaxScaler()
    normalized_coverages = np.full(masked_coverages.shape, -9999, dtype=float)

    # Iterates through each layer
    for i in range(14):
        layer = masked_coverages[i]

        # Only normalize valid not masked data
        valid_mask = ~layer.mask
        valid_values = layer[valid_mask]

        # Scale valid values
        scaled_values = scaler.fit_transform(valid_values.reshape(-1, 1)).flatten()

        # Renew into the layer with same dimensions
        normalized_layer = normalized_coverages[i]
        normalized_layer[valid_mask] = scaled_values

        # Save back to the normalized coverages
        normalized_coverages[i] = normalized_layer

    masked_coverages = np.ma.masked_where(normalized_coverages == -9999, normalized_coverages)

    # Encode species as 0 or 1
    le = LabelEncoder()
    encoded_train_species = le.fit_transform(filtered_train)
    encoded_test_species = le.transform(filtered_test)

    # Make arrays for training, validation and test data
    encoded_train = np.array([
        (encoded_train_species[i], filtered_train_raw['dd long'][i], filtered_train_raw['dd lat'][i])
        for i in range(len(filtered_train))
    ], dtype=[('species', 'int'), ('dd long', '<f4'), ('dd lat', '<f4')])

    encoded_test = np.array([
        (encoded_test_species[i], filtered_test_raw['dd long'][i], filtered_test_raw['dd lat'][i])
        for i in range(len(filtered_test))
    ], dtype=[('species', 'int'), ('dd long', '<f4'), ('dd lat', '<f4')])


    # Split the test data into validation and test sets (50/50 split)
    num_test_samples = len(encoded_test)
    indices = np.arange(num_test_samples)
    val_size = num_test_samples // 2

    val_indices = indices[:val_size]
    final_test_indices = indices[val_size:]

    # Slice the encoded_test array
    encoded_val = encoded_test[val_indices]
    encoded_test = encoded_test[final_test_indices]


    return {
        "train_data": encoded_train,
        "val_data": encoded_val,
        "test_data": encoded_test,
        "coverages": masked_coverages,
    }
