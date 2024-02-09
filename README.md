# Welcome to RiskNet!
This is a downloadable Pip package where you can access and run an XGBoost pipeline.

To access the Risknet package, you can download it by running `pip install risknet`. You can find documentation here: https://pypi.org/project/risknet/1.0.17/
To access the Risknet code, fork the code from the Git repo.

# Folder/File Layout Layout
- src/risknet
   - `config`: holds `conf.yaml`
      - `conf.yaml`: helps with setup
      - `handlers.py`
   - `data`: empty folder where user can store FM dataset
   - `jobs`
      - `cloud_etl.py`: helps save files to cloud
   - `proc`: contains preprocessing steps like feature encoding, label prep, and train-test splits
      - `encoder.py`: feature engineering/encoding categoricals
      - `label_prep.py`: defines default, progress on loan
      - `reducer.py`: reduces features based on importance, also train/test/val splits
   -`run`: contains files for running pipeline
      - `main.py`: currently empty. WIP define pipeline as a function and call here
      - `model.py`: defines the model class + functions
      - `pipeline.py`: calls functions to execute the pipeline
   - `sys`: contains files to set up system environment and logging
      - `log.py`: sets up logger
      - `managers.py`: sets up the dask manager
      - `system.py`: defines creating and removing files via the `sys` package
   -`main.py`: logs start, stop time for running the program (including downloading packages from setup.cfg)
   - `tests`: store tests here
      -`test_stub.py`: currently only asserts True == True. No tests added yet.

# Running The Code
Currently, this code is hosted on testpypi, a Test version of the Python Packaging Index. You can see package documentation [here](https://test.pypi.org/project/risknet/).

As described in the website above, you can access the code by running `pip install -i https://test.pypi.org/simple/ risknet` on your local computer. Ensure pip is updated.
- Note: this code might not work on your computer as many dependencies of the package like pandas are not available for download via testpypi

To run a specific part of the code, use risknet.utils.{feature_name} as is standard practice when accessing Python packages.

# Reproducibility Information:
## Accessing Data
You may want to access our base dataset for reference purposes. Here's how you can do that:

In this study we will use the Freddie Mac Single-Family Loan Dataset to run code. Specifically, we will use the 2009_Q1 dataset.
1. Go to this link [here](https://www.freddiemac.com/research/datasets/sf-loanlevel-dataset). This will redirect you to the Freddie Mac site.
2. Scroll down until you reach a table that says "Loan-Level Dataset Files". Download the **standard quarterly** dataset.
3. Submit necessary education credentials including name, email, and reason for accessing the files. There should not be a payment step. The site will email you a username and password.
4. Reload the page and log in with your new email/password credentials.
5. Download the Quarter 1 data from the year 2009. You will receive a .zip file in your Downloads folder. 

If you unzip the file, you will see multiple files including a "date_time" file and a "data" file.
6. Save these files into the src/data folder in a local copy of this repository

## Dependencies
You can find a list of this package's dependencies inside the file called `setup.cfg`.

In summary, the downloads needed for this code are:
- numpy==1.26.1
- pandas==2.1.2
- dask[complete]==2023.10.1
- xgboost==2.0.1
- PyYAML
- types-PyYAML
- pyarrow
- fastparquet
- pytest
- pytest-cov
- mypy
- flake8

# Steps to Update Version on TestPyPi
To update the version:
1. Reset the code from the previous version (if necessary)
- `rm -rf dist build` to remove build folder
- manually remove the "risknet.egg-info" folder. This will change `src` to `src/risknet`.
2. Update setup.cfg's version number depending on if major, minor, or bug change
3. Rerun `python3 -m build` (you should get a new dist folder + egg folder in \src)
   - THIS SHOULD CREATE A NEW binary file where version is UPDATED
   - Make sure you're in the same directory as your setup.cfg when you run this command.
4. Rerun `python3 -m twine upload --repository testpypi dist/*`
   - Username: `__token__`
   - Password: [testpypi password starting with pypi]
   - If you did NOT update the version # before running `build` then you will get an error

# When Uploading to PyPi:
Repeat steps above with these important differences:
- Use `python3 -m twine upload dist/*` to upload to PyPi. You do not need to specify `--repository testpypi` when uploading to PyPi.
- Login username will be the same. However, remember to use PyPi's login password/API token, **not** TestPyPi's token for the password.

# Package Version History Documentation:
0.0.1: Ran into problems with installing pytest-cov

0.0.2: Got error:
`ERROR: Could not find a version that satisfies the requirement dask[complete] (from risknet) (from versions: none) ERROR: No matching distribution found for dask[complete]`

0.0.3: Tried moving dask into [options.extras_require] not install_requires. Got error for flake8

0.0.4: Commented out more packages.
Got error: `error: package directory 'lib3/yaml' does not exist` and `metadata-generation-failed`.

0.0.5: Moved types-YAML into options, still get `'lib3/yaml does not exist'` error :/

0.0.6: we're removing YAML as a test
Got error `ERROR: Could not find a version that satisfies the requirement typing (from risknet) (from versions: none) ERROR: No matching distribution found for typing`

0.0.10: reverting to version 0.0.6, checking if it works --success! But still error `No matching distribution found for typing`

0.0.11: what happens when we remove typing (since it's part of stdlib in python >3.5)?

0.0.12: commented out all packages added by EC

0.0.13: Got new error:
`ERROR: Cannot install pandas==1.3.4 and risknet==0.0.12 because these package versions have conflicting dependencies.

The conflict is caused by:
    risknet 0.0.12 depends on numpy
    pandas 1.3.4 depends on numpy>=1.17.3; platform_machine != "aarch64" and platform_machine != "arm64" and python_version < "3.10"`

Solution: will try setting python > 3.10

**SUCCESS! Version 0.0.13 can download from emily's (base)**
Caveats:
- Probably can only work because it has all dependencies already installed in the env (it threw an error when I tried to run it in risknet_test)
- Can only import `risknet.utils.label_prep` since `risknet.utils.encoder`, etc. have local imports to different .py files which Python can't read (??)
But Running `>>> import risknet.utils.label_prep as label_prep, >>> label_prep.label_proc(fm_root, data)` works!!

0.0.14: change setup.py to `if __name__ == "__main__: setup()`.

0.0.16: try compiling on base environment (python 3.12, pip 23.2)

1.0.17: Uploading to PyPi as risknet! SUCCESS!! :D"

# Running RiskNet with Docker
For users who prefer Docker, we provide a Docker image of RiskNet that encapsulates all dependencies and offers a straightforward way to run the pipeline.
Docker User Guide
For detailed instructions on using the Docker image, including configuration options and advanced usage, refer to our comprehensive Docker User Guide available [here](https://docs.google.com/document/d/1bzSW1dIQg1xbr0qvtNB7DhVBaFcTQRXwFjURhHDCV0o/edit?usp=sharing).
