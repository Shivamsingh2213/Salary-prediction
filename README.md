# Salary-prediction
salary prediction using machine learning
# Developer Salary Prediction

This project aims to predict annual salaries based on various factors such as country, education level, employment status, and years of professional coding experience. The project utilizes machine learning for predictions and provides interactive widgets for input. Additionally, it visualizes the results using various plots.

## Table of Contents

- [Overview](#overview)
- [Data Source](#data-source)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Visualizations](#visualizations)
- [License](#license)

## Overview

The goal of this project is to build a machine learning model to predict annual salaries based on several factors. It includes data preprocessing, model training, and interactive widgets for user inputs. The results are visualized using histograms, boxplots, and scatter plots.

## Data Source

The dataset used for this project is `survey_results_public.csv`, which contains survey data from developers.

## Requirements

- Python 3.x
- pandas
- numpy
- scikit-learn
- ipywidgets
- matplotlib
- seaborn
- plotly

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/salary-prediction.git
    cd salary-prediction
    ```

2. Install the required packages:
    ```bash
    pip install pandas numpy scikit-learn ipywidgets matplotlib seaborn plotly
    ```

## Usage

1. Ensure you have the `survey_results_public.csv` file in the project directory.
   or you can install the dataset directly from the https://survey.stackoverflow.co/
2. Run the Jupyter Notebook:
    ```bash
    jupyter notebook
    ```
3. Open the notebook and run the cells to load the data, train the model, and use the interactive widgets for predictions.

## Project Structure

- `survey_results_public.csv`: The dataset file.
- `salary_prediction.ipynb`: Jupyter Notebook containing the code for data preprocessing, model training, and visualizations.
- `README.md`: This readme file.

## Visualizations

The project includes the following visualizations:

1. **Salary Distribution**: Histogram of the salary distribution with a KDE plot.
2. **Salary by Country (Boxplot)**: Boxplot showing the salary distribution across different countries.
3. **Salary by Education Level**: Boxplot displaying salary distribution based on education levels.
4. **Years of Experience vs. Salary**: Scatter plot with a linear regression line showing the relationship between years of experience and salary.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

