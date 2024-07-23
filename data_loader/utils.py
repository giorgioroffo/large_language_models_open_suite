def print_dataset_details(dataset):
    """
    Print details about the dataset including the number of samples, shape,
    keys, and contents of the first sample, and how to access each field.

    Parameters:
    dataset (list of dict): The dataset to describe.

    """
    # Print the number of samples in the dataset
    print(f"Number of samples in the dataset: {len(dataset)}")

    # Print the unique number of categories in the dataset
    unique_categories = set(dataset['category'])
    print(f"Number of unique categories in the dataset: {len(unique_categories)}")

    # Print the shape of the dataset
    print(f"Shape of the dataset: {len(dataset), len(dataset[0].keys()) if dataset else 0}")

    # Print the keys of the first sample in the dataset to show the structure
    print(f"Keys in the first sample: {list(dataset[0].keys()) if dataset else 'No data available'}")

    # Print the first sample to show its contents
    if dataset:
        print("First sample in the dataset:")
        for key, value in dataset[0].items():
            print(f"{key}: {value}")
    else:
        print("No data available")

    # Accessing each field in the first sample
    if dataset:
        instruction = dataset[0]['instruction']
        context = dataset[0]['context']
        response = dataset[0]['response']
        category = dataset[0]['category']

        print("\nAccessing fields in the first sample:")
        print(f"Instruction: {instruction}")
        print(f"Context: {context}")
        print(f"Response: {response}")
        print(f"Category: {category}")
