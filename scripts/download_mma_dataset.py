import os
import subprocess
import sys
import zipfile

import pandas as pd
import numpy as np
from PIL import Image

# Kaggle API Authentication and Download:
# Set up Kaggle API credentials
kaggle_json_path = os.path.expanduser("~/.kaggle/kaggle.json")

if not os.path.isfile(kaggle_json_path):
    # TODO: change username and key
    # username = input("Kaggle username: ")
    # api_key = input("Kaggle API key: ")
    username = "ranweiss53"
    api_key = "27590f70d038fc0f2328107bda6cb521"

    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(kaggle_json_path), exist_ok=True)

    # Write the credentials to the kaggle.json file
    with open(kaggle_json_path, "w") as file:
        file.write(f'{{"username":"{username}","key":"{api_key}"}}')

    # Set file permissions to read and write for the owner only
    os.chmod(kaggle_json_path, 0o600)

# Importing kaggle will authenticate automatically
# import kaggle

# Command to authenticate and download the dataset - mma-facial-expression
api_command = "kaggle datasets download -d mahmoudima/mma-facial-expression"

# Execute the command
try:
    subprocess.run(api_command, shell=True, check=True)
except subprocess.CalledProcessError as e:
    print("An error occurred while downloading the dataset. Please double-check your Kaggle API key.")
    os.remove(kaggle_json_path)
    sys.exit(1)

# Extracting and Preparing Dataset:
print("Preparing dataset..")
with zipfile.ZipFile("mma-facial-expression.zip", "r") as zip_ref:
    zip_ref.extractall("mma")

output_folder_path = "../mma"

os.remove("mma-facial-expression.zip")

