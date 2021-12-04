# Apulu

Apulu is a project inspired by the project [Apollo](https://github.com/jnrkufuor/apollo). This project implements data pipeline, data cleaning, graph construction, and modeling with Graph Neural Network.

- [Apulu](#apulu)
  - [Getting Started](#getting-started)
  - [Project Configuration](#project-configuration)
  - [Usage](#usage)
    - [Data Pipeline](#data-pipeline)


## Getting Started

Start with  creating a python virtual environment by following commands

```bash
conda create -- name apulu python=3.8.5
```

Install the dependencies by

```bash
conda activate apulu
./env.sh
```

## Project Configuration

The project is driven by configuration file: a yaml file specified every parameters. You can refer `configs/example.yaml` for your own configuration file. Our report is mainly investigating dow and jones listing, so our report refers to `configs/dow_jones.yaml`.

## Usage

our project is used sequentially with `data pipeline`, `datasets`, `modeling`, ...

to run the entire project once, you may run following the steps in the `notebooks`: