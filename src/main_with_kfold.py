# import os
# import re
# import pandas as pd
# from sklearn.model_selection import train_test_split

# import jpype
# from jpype import JClass

# from BoostingLRWrapper import BoostingLRWrapper
# from utils import kendalls_tau

# def start_jvm():
#     # Start the JVM and set up the classpath
#     jvm_args = ["-Xmx1g"]  # Set maximum heap size for JVM
#     cp = ["./", "../lib/*", "./weka", "../datasets"]  # Set the classpath for Weka and other Java dependencies
#     jpype.startJVM(*jvm_args, classpath=cp, convertStrings=True)

# def stop_jvm():
#     jpype.shutdownJVM()

# def load_xarff(file_path):
#     """Custom loader to read XARFF files and convert them into a pandas DataFrame."""
#     with open(file_path, 'r') as file:
#         lines = file.readlines()

#     # Separate metadata and data
#     data_start_idx = lines.index('@data\n') + 1
#     metadata = lines[:data_start_idx]
#     data_lines = lines[data_start_idx:]

#     # Extract column names and types from metadata
#     column_names = []
#     attribute_info = {}
#     for line in metadata:
#         if line.startswith('@attribute'):
#             # Extract the column name and its type information
#             match = re.findall(r'@attribute\s+([\w\d_-]+)\s+(.*)', line)
#             if match:
#                 column_name, attribute_type = match[0]
#                 column_names.append(column_name)
#                 attribute_info[column_name] = attribute_type.strip()

#     # Parse the data lines into a list of lists
#     data = []
#     for line in data_lines:
#         # Split the line by commas and strip any extra spaces
#         split_line = re.split(r',\s*(?![^{}]*\})', line.strip())
#         # Handle the RANKING column by converting it to a list
#         if split_line[-1].startswith('{'):
#             ranking = split_line[-1][1:-1].split('>')
#             split_line[-1] = ranking
#         data.append(split_line)

#     # Create a DataFrame from the parsed data
#     df = pd.DataFrame(data, columns=column_names)

#     return df, attribute_info

# def save_to_xarff(df, file_path, relation_name="dataset", attribute_info=None):
#     """Save a Pandas DataFrame as a XARFF file."""
#     with open(file_path, 'w') as f:
#         # Write the relation name
#         f.write(f"@relation {relation_name}\n\n")

#         # Write the attribute names and types using the provided attribute_info
#         for column in df.columns:
#             attribute_type = attribute_info.get(column, 'STRING')
#             f.write(f"@attribute {column} {attribute_type}\n")

#         # Write the data
#         f.write("\n@data\n")
#         for _, row in df.iterrows():
#             row_data = []
#             for value in row:
#                 if isinstance(value, list):
#                     # Convert list back to ranking format
#                     value = '>'.join(value)
#                 row_data.append(value)
#             f.write(','.join(map(str, row_data)) + '\n')

# def load_dataset_as_Instances(file):
#     DataSource = JClass("weka.core.converters.ConverterUtils$DataSource")
#     data = DataSource.read(file)
#     data.setClassIndex(data.numAttributes() - 1)
#     return data

# def run_lrt(train_data, test_data):
#     total_kt = 0.0
#     lrt = JClass("weka.classifiers.labelranking.LRT")()
#     lrt.buildClassifier(train_data)
#     for i in range(test_data.numInstances()):
#         instance = test_data.instance(i)
#         prefs = model.boosting_lr.preferences(instance)
#         preds = lrt.distributionForInstance(instance)

#         total_kt += kendalls_tau(preds, prefs)

#     # Calculate and print the average Kendall's Tau
#     accuracy = total_kt / test_data.numInstances()
#     print(f"Test Accuracy (Average Kendall's Tau): {accuracy * 100:.2f}%")


# from sklearn.model_selection import KFold

# if __name__ == "__main__":

#     start_jvm()

#     dataset = "../datasets/sushi.xarff"

#     # Extract the base name (without extension) and directory
#     base_name = os.path.basename(dataset).replace(".xarff", "")
#     directory = os.path.dirname(dataset)

#     # Load the dataset and get attribute information
#     df, attribute_info = load_xarff(dataset)

#     # Initialize KFold with 5 splits (or choose any number of folds)
#     kf = KFold(n_splits=5, shuffle=True, random_state=42)

#     total_accuracy = 0
#     fold = 1

#     # Perform KFold cross-validation
#     for train_index, test_index in kf.split(df):
#         print(f"Running fold {fold}...")
#         fold += 1

#         # Split the data into training and test sets using indices
#         train_data = df.iloc[train_index]
#         test_data = df.iloc[test_index]

#         # Save the split datasets to XARFF files with the original attribute info
#         train_dataset = os.path.join(directory, f"{base_name}_train_fold{fold}.xarff")
#         test_dataset = os.path.join(directory, f"{base_name}_test_fold{fold}.xarff")
        
#         save_to_xarff(train_data, train_dataset, relation_name=f"{base_name}_train_fold{fold}", attribute_info=attribute_info)
#         save_to_xarff(test_data, test_dataset, relation_name=f"{base_name}_test_fold{fold}", attribute_info=attribute_info)

#         print(f"Training and test datasets for fold {fold} saved to XARFF files.")

#         # Load dataset as Instances
#         train_data = load_dataset_as_Instances(train_dataset)
#         test_data = load_dataset_as_Instances(test_dataset)

#         # Initialize the BoostingLR wrapper
#         model = BoostingLRWrapper(max_iterations=50)

#         # Train the model on the training dataset
#         model.fit(train_data)

#         # Predict on the test dataset
#         predictions = model.predict(test_data)

#         # Evaluate predictions
#         total_kt = 0.0
#         for i in range(test_data.numInstances()):
#             instance = test_data.instance(i)
#             prefs = model.boosting_lr.preferences(instance)
#             preds = predictions[i]

#             total_kt += kendalls_tau(preds, prefs)

#         # Calculate and print the average Kendall's Tau
#         accuracy = total_kt / test_data.numInstances()
#         print(f"Test Accuracy (Average Kendall's Tau) for fold {fold}: {accuracy * 100:.2f}%")
#         total_accuracy += accuracy

#     print(f"Total average accuracy across all folds: {(total_accuracy / kf.get_n_splits()) * 100:.2f}%")

#     # Shutdown the JVM
#     stop_jvm()


import os
import re
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

import jpype
from jpype import JClass

from BoostingLRWrapper import BoostingLRWrapper
from utils import kendalls_tau

def start_jvm():
    # Start the JVM and set up the classpath
    jvm_args = ["-Xmx1g"]  # Set maximum heap size for JVM
    cp = ["./", "../lib/*", "./weka", "../datasets"]  # Set the classpath for Weka and other Java dependencies
    jpype.startJVM(*jvm_args, classpath=cp, convertStrings=True)

def stop_jvm():
    jpype.shutdownJVM()

def load_xarff(file_path):
    """Custom loader to read XARFF files and convert them into a pandas DataFrame."""
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Separate metadata and data
    data_start_idx = lines.index('@data\n') + 1
    metadata = lines[:data_start_idx]
    data_lines = lines[data_start_idx:]

    # Extract column names and types from metadata
    column_names = []
    attribute_info = {}
    for line in metadata:
        if line.startswith('@attribute'):
            # Extract the column name and its type information
            match = re.findall(r'@attribute\s+([\w\d_-]+)\s+(.*)', line)
            if match:
                column_name, attribute_type = match[0]
                column_names.append(column_name)
                attribute_info[column_name] = attribute_type.strip()

    # Parse the data lines into a list of lists
    data = []
    for line in data_lines:
        # Split the line by commas and strip any extra spaces
        split_line = re.split(r',\s*(?![^{}]*\})', line.strip())
        # Handle the RANKING column by converting it to a list
        if split_line[-1].startswith('{'):
            ranking = split_line[-1][1:-1].split('>')
            split_line[-1] = ranking
        data.append(split_line)

    # Create a DataFrame from the parsed data
    df = pd.DataFrame(data, columns=column_names)

    return df, attribute_info

def save_to_xarff(df, file_path, relation_name="dataset", attribute_info=None):
    """Save a Pandas DataFrame as a XARFF file."""
    with open(file_path, 'w') as f:
        # Write the relation name
        f.write(f"@relation {relation_name}\n\n")

        # Write the attribute names and types using the provided attribute_info
        for column in df.columns:
            attribute_type = attribute_info.get(column, 'STRING')
            f.write(f"@attribute {column} {attribute_type}\n")

        # Write the data
        f.write("\n@data\n")
        for _, row in df.iterrows():
            row_data = []
            for value in row:
                if isinstance(value, list):
                    # Convert list back to ranking format
                    value = '>'.join(value)
                row_data.append(value)
            f.write(','.join(map(str, row_data)) + '\n')

def load_dataset_as_Instances(file):
    DataSource = JClass("weka.core.converters.ConverterUtils$DataSource")
    data = DataSource.read(file)
    data.setClassIndex(data.numAttributes() - 1)
    return data

def run_lrt(train_data, test_data):
    total_kt = 0.0
    lrt = JClass("weka.classifiers.labelranking.LRT")()
    lrt.buildClassifier(train_data)
    for i in range(test_data.numInstances()):
        instance = test_data.instance(i)
        prefs = model.boosting_lr.preferences(instance)
        preds = lrt.distributionForInstance(instance)

        total_kt += kendalls_tau(preds, prefs)

    # Calculate and print the average Kendall's Tau
    accuracy = total_kt / test_data.numInstances()
    print(f"Test Accuracy (Average Kendall's Tau): {accuracy * 100:.2f}%")

if __name__ == "__main__":

    start_jvm()

    dataset = "../datasets/sushi.xarff"

    # Extract the base name (without extension) and directory
    base_name = os.path.basename(dataset).replace(".xarff", "")
    directory = os.path.dirname(dataset)

    # Create the full paths and the desired string
    train_base_name = f"{base_name}_train"
    test_base_name = f"{base_name}_test"

    train_dataset = os.path.join(directory, f"{train_base_name}.xarff")
    test_dataset = os.path.join(directory, f"{test_base_name}.xarff")

    # Load the dataset and get attribute information
    df, attribute_info = load_xarff(dataset)


    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    total_kt_scores = []

    # Loop over the folds
    for fold, (train_index, test_index) in enumerate(kf.split(df)):
        print(f"Processing fold {fold + 1}...")

        # Split data into training and test sets for this fold
        train_data, test_data = df.iloc[train_index], df.iloc[test_index]

        # Save the split datasets into XARFF files
        train_dataset_file = f"train_fold_{fold}.xarff"
        test_dataset_file = f"test_fold_{fold}.xarff"

        save_to_xarff(train_data, train_dataset_file, relation_name=f"train_fold_{fold}", attribute_info=attribute_info)
        save_to_xarff(test_data, test_dataset_file, relation_name=f"test_fold_{fold}", attribute_info=attribute_info)

        # Load the dataset as Instances for Weka
        train_instances = load_dataset_as_Instances(train_dataset_file)
        test_instances = load_dataset_as_Instances(test_dataset_file)

        # Initialize and train the BoostingLR model
        model = BoostingLRWrapper(max_iterations=25)
        model.fit(train_instances)

        # Predict on the test set and evaluate using Kendall's Tau
        predictions = model.predict(test_instances)
        total_kt = 0.0

        for i in range(test_instances.numInstances()):
            instance = test_instances.instance(i)
            prefs = model.boosting_lr.preferences(instance)
            preds = predictions[i]

            total_kt += kendalls_tau(preds, prefs)

        # Calculate the average Kendall's Tau for this fold
        avg_kt = total_kt / test_instances.numInstances()
        total_kt_scores.append(avg_kt)
        print(f"Fold {fold + 1} Kendall's Tau: {avg_kt * 100:.2f}%")

    # Calculate the overall average Kendall's Tau across all folds
    mean_kt = np.mean(total_kt_scores)
    print(f"Overall Average Kendall's Tau over {kf.get_n_splits()} folds: {mean_kt * 100:.2f}%")

    stop_jvm()