from sklearn.datasets import fetch_species_distributions
import numpy as np
import matplotlib.pyplot as plt


def preprocess_data():
    """
    First, change to only use the two species examples that end with 0 (for now only maybe)
    """
    data = fetch_species_distributions()

    species_train = data.train['species']
    species_test = data.test['species']


    # 2. Simplify species names at the very beginning
    def simplify_species_name(species):
        """Convert b'microryzomys_minutus_0' -> b'microryzomys_minutus'"""
        name_str = species.decode('utf-8')
        base_str = name_str.rsplit('_', 1)[0]
        return base_str.encode('utf-8')  # or just return base_str (string)

    species_train_simplified = np.array([
        simplify_species_name(s) for s in data.train['species']
    ])
    species_test_simplified = np.array([
        simplify_species_name(s) for s in data.test['species']
    ])

    print(species_test_simplified[:5])

    print(species_train[80:90])
    print(species_train_simplified[80:90])

    # This part is kinda confusing ngl but it basically just removes the numbers at the
    # end of the species' names and groups them together

    exclusive_species = [b'microryzomys_minutus', b'bradypus_variegatus']

    train_filter = np.isin(species_train_simplified, exclusive_species)
    filtered_train_raw = data.train[train_filter]

    test_filter = np.isin(species_test_simplified, exclusive_species)
    filtered_test_raw = data.test[test_filter]

    filtered_train = species_train_simplified[train_filter]
    filtered_test = species_test_simplified[test_filter]

    print("\nFiltered Training Data Example:")
    print(filtered_train[:5])
    print(len(data.train['species']))
    print(len(filtered_train))

    print(len(data.test['species']))
    print(len(filtered_test))

    print("\nFiltered Test Data Example:")
    print(filtered_test[:5])


    """
    Then deal with the missing data in coverages (features).
    """

    coverages = data.coverages

    # Mask missing data for coverages
    masked_coverages = np.ma.masked_where(coverages == -9999, coverages)

    print("Before:")
    print(coverages[1, :, :])

    print("After:")
    print(masked_coverages[1, :, :])

    print(coverages.shape)
    print(masked_coverages.shape)


    """
    Normalize all the data
    """

    from sklearn.preprocessing import MinMaxScaler

    # scaler we use because data is not normally distributed!!
    scaler = MinMaxScaler()

    # template filled w -9999
    normalized_coverages = np.full(masked_coverages.shape, -9999, dtype=float)

    print(normalized_coverages[1, :, :])

    # Iterates through each layer
    for i in range(14):
        layer = masked_coverages[i]

        # Only normalize valid (non-masked) data
        valid_mask = ~layer.mask
        valid_values = layer[valid_mask]  # Extract valid values

        # Scale valid values
        scaled_values = scaler.fit_transform(valid_values.reshape(-1, 1)).flatten()

        # Reintegrate into the layer with same dimensions
        normalized_layer = normalized_coverages[i]
        normalized_layer[valid_mask] = scaled_values

        # Save back to normalized_coverages
        normalized_coverages[i] = normalized_layer

    print("\nNormalized Coverages Example:")
    print(normalized_coverages[0, 500:505, 500:505])

    masked_coverages = np.ma.masked_where(normalized_coverages == -9999, normalized_coverages)

    print("Before:")
    print(coverages[1, :, :])

    print("After:")
    print(masked_coverages[1, :, :])

    layer = masked_coverages[3]


    """
    Encode species to either 0 (b'bradypus_variegatus_0') or 1 (b'microryzomys_minutus_0') instead of their names
    """

    from sklearn.preprocessing import LabelEncoder


    # Encodes species as 0 or 1
    le = LabelEncoder()
    encoded_train_species = le.fit_transform(filtered_train)
    encoded_test_species = le.transform(filtered_test)

    # Replace the species names with their encoded labels in filtered_train and filtered_test
    encoded_train = np.array([
        (encoded_train_species[i], filtered_train_raw['dd long'][i], filtered_train_raw['dd lat'][i])
        for i in range(len(filtered_train))
    ], dtype=[('species', 'int'), ('dd long', '<f4'), ('dd lat', '<f4')])

    encoded_test = np.array([
        (encoded_test_species[i], filtered_test_raw['dd long'][i], filtered_test_raw['dd lat'][i])
        for i in range(len(filtered_test))
    ], dtype=[('species', 'int'), ('dd long', '<f4'), ('dd lat', '<f4')])

    # Print to verify
    print("\nFiltered Training Data (Encoded):")
    print(encoded_train[400:405])

    print("\nFiltered Test Data (Encoded):")
    print(encoded_test[400:405])

    print("Unique species in filtered_train (raw):", np.unique(filtered_train_raw['species']))
    print("Unique species in filtered_test (raw):", np.unique(filtered_test_raw['species']))

    print("Encoded train species unique values:", np.unique(encoded_train_species))
    print("Encoded test species unique values:", np.unique(encoded_test_species))


    """
    Making of valdation set from th test set
    """
    # Let's say we do a 50/50 split of the test set
    num_test_samples = len(encoded_test)
    indices = np.arange(num_test_samples)
    val_size = num_test_samples // 2    # integer division

    val_indices = indices[:val_size]
    final_test_indices = indices[val_size:]

    # Now slice the encoded_test array
    encoded_val = encoded_test[val_indices]
    encoded_final_test = encoded_test[final_test_indices]


    return {
        "train_data": encoded_train,
        "val_data": encoded_val,
        "test_data": encoded_final_test,
        "coverages": masked_coverages,
    }