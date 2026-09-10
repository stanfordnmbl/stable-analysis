This repository contains the code required to replicate the results from the following manuscript (in-submission):

> **StaBLE: digital metrics capture balance performance across a wide spectrum of balance tasks and abilities**
>
> Hannah Heigold, Parker S. Ruth, Julie Muccini, Dawit Lee, Sydney Barta, Ariana Rodrigues, Trevor Hastie, Kristen K. Steenerson, Scott L. Delp


This code has been tested on Windows and MacOS
## Code Installation
Install the [(mini)conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) Python environment manager and open a terminal.

1. Navigate to an empty working directory: `cd [path_to_empty_working_directory]`
2. Clone the GitHub repository: `git clone https://github.com/stanfordnmbl/stable-analysis.git`
3. Navigate to the directory: `cd stable-analysis`
4. Create environment: `conda create -n stable-analysis python=3.11`
5. Activate environment: `conda activate stable-analysis`
6. Install OpenSim: `conda install -c opensim-org opensim`
7. Install required packages: `python -m pip install -r requirements.txt`


## Data Access

The dataset is hosted on Zenodo: https://doi.org/10.5281/zenodo.22286165

Data can be downloaded through the browser, or through the command line as follows:

```
zenodo_get -d 10.5281/zenodo.22286165 -o zenodo_data
```

Move all of the files, except the dataset `README.md`, to the working directory.

Windows:
```
move zenodo_data\*.csv .
move zenodo_data\*.zip .
```

MacOS: 
```
mv zenodo_data/*.csv zenodo_data/*.zip .
```

At this point, your working directory should be organized as follows:

```
[working_directory]
└── stable-analysis
    ├── get_features
	├── zenodo_data
	├── .gitignore
    ├── compute_scores.ipynb
	├── df_ps_col_labels.csv
	├── feature_info.csv
	├── fig2.ipynb
	├── fig4.ipynb
	├── fig5.ipynb
	├── fig6.ipynb
	├── incomplete_trials.csv
	├── LICENSE
	├── opencap_data.zip
	├── plots.py
	├── README.md
	├── requirements.txt
	├── stable_demographics.csv
	├── stable_features.csv
	└── stable_scores.csv
		
```

Unzip the `opencap_data.zip` to the stable-analysis directory.

Windows:
```
tar -xf opencap_data.zip
```
MacOS:
```
unzip opencap_data.zip
```

## Juypter Notebooks
The code for the extracting the features, StaBLE scores, and figure generation is in Jupyter Notebooks. 
These can be run in Jupyter Lab (use command `jupyter lab` to launch) or in [Visual Studio Code](https://code.visualstudio.com/docs/datascience/jupyter-notebooks) with the Jupyter and Python extensions.

## Feature Generation
`get_features` contains the code used to segment and extract the features for each StaBLE task for the exemplar data in `opencap_data`.
Extract the task features for the exemplar participants using the`get_features.ipynb` notebook.
Run each cell in the notebook to generate a `.json` file (saved to `features`) containing the respective task's features for each participant.

Feature descriptions, and whether the goal is to minimize or maximize a given feature (sign), can be found in `feature_info.csv`.

Feature values for the full dataset can be found in `stable_features.csv`. 

## StaBLE Score Generation
`compute_scores.ipynb` computes the normalized features, task scores, and StaBLE score for all 180 participants. 

It reads in `stable_demographics.csv` and `stable_features.csv` and outputs `stable_scores.csv`. 
The `stable_scores.csv` file combines the input csv data (demographics and raw features) with the normalized features and scores. 

## Figure Generation
`fig[2,4,5,6].ipynb` reproduce the reported figures and statistical analyses. Statistics from the manuscript are displayed in the Jupyter Notebook and figures are saved to `figs`

## Citing This Work

We invite you to cite our [Zenodo dataset](https://doi.org/10.5281/zenodo.13788592).


> H. Heigold, et al., “Stanford Balance Level Evaluation (StaBLE) digital metrics of balance performance across a wide spectrum of balance tasks and abilities.” Zenodo, Sept. 3, 2026. doi:10.5281/zenodo.22286165

