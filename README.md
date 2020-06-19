# Thesis Project:
Repository for my M.Sc. thesis work on TIme Series Forecasting, using different algorithms and exogenous data.

## Table of contents
- [Thesis Project:](#thesis-project)
  - [Table of contents](#table-of-contents)
  - [Getting started](#getting-started)
    - [Prerequisites](#prerequisites)
    - [Installing the required packages](#installing-the-required-packages)

## Getting started

### Prerequisites
This project uses *pipenv* for dependency management and virtual environment management.

If you do not have pipenv installed you can install it using: 
- `brew install pipenv` 
- `pip install --user pipenv`

For other installation options check out [the pipenv installation docs](https://pipenv-fork.readthedocs.io/en/latest/install.html#installing-pipenv).

### Installing the required packages

!You can follow this procedure only after you have installed pipenv and cloned the repository!

Navigate to the root directory where the *Pipfile* is located. The *Pipfile* is where pipenv stores package information. To create a Python virtual environment and install all the packages simply run `pipenv install`.

After that process is finished you will see that a *Pipfile.lock* file has been created. The virtual environment with all the needed packages is now ready to use. You can access it by using `pipenv shell`. To exit the virtual environment simply use `exit`. 

Once the virtual environment is activated you can run any Python command you prefer. To access the notebooks, run the `jupyter-notebook` command and then the Jupyter window will open in your browser (by default in port 8888).

You can check examples and other commands you can use with pipenv by simply running `pipenv`.