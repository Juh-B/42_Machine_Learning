# --- Add CSV from update
# from google.colab import files
# uploaded = files.upload()

# Salary_dataset.csv(text/csv) - 664 bytes, last modified: 1/9/2023 - 100% done
# Saving Salary_dataset.csv to Salary_dataset.csv

# --- Add CSV from google drive
# from google.colab import drive
# drive.mount("/content/drive")
# file = 'drive/My Drive/Colab/Trasactions.csv'

# ------------ Ex 00 ------------

# You need to import the pandas and seaborn library

# ###
# 1. pandas:
# Para manipulação de dados tabulares (DataFrames)
# Internamente usa arrays NumPy para armazenar dados
# Facilita leitura/escrita, limpeza e análise de dados
# 2. seaborn:
# Biblioteca baseada em matplotlib para visualização estatística
# Usa arrays NumPy e pandas DataFrames como entrada
# ###

import pandas as pd
import seaborn as sns

# Read the data from the CSV file with read from pandas
data = pd.read_csv("Salary_dataset.csv")

# Configure the style
sns.set_theme(style="darkgrid", font_scale=0.8)

# Plot the data points with seaborn
sns.jointplot(data=data, x="YearsExperience", y="Salary", kind="scatter", color="#734CAD")

# ------------ Ex 01 ------------

# We're going to use this library and find out how it works on the Internet
from sklearn.linear_model import LinearRegression

# Separate the features (YearsExperience) from the target variable (Salary)
x = data[["YearsExperience"]]
y = data["Salary"]


# Create a linear regression model
# ###
# Linear Regression
# y= b0 + (b1 * x)

# y: variável que você quer prever (ex: salário)
# x: variável independente (ex: anos de experiência)
# b₀: intercepto (valor de y quando x = 0)
# b₁: coeficiente angular (quanto y aumenta a cada unidade de x)
# ###
regression = LinearRegression()

# Train the model
# ###
# print("Intercepto (b0):", regression.intercept_)
# print("Coeficiente (b1):", regression.coef_[0])
# ###
train = regression.fit(x, y)

# Predict the Salary values based on YearsExperience
predictions = regression.predict(x)
data["PredictedSalary"] = predictions

# Create a linear regression plot using lmplot
g = sns.lmplot(
    data=data,
    x="YearsExperience",
    y="Salary",
    scatter=True,
    scatter_kws={"s": 20, "color": "#734CAD"},
    line_kws={"color": "#734CAD"}
    )

# Set labels and title
g.set_axis_labels("Years of Experience", "Salary", fontsize=8)
g.fig.suptitle("Linear Regression: Years of Experience vs Salary", fontsize=10)
g.fig.subplots_adjust(top=0.92)

# ------------ Ex 02 ------------

# You should have these results with 10 years of expertise, so it's up to you to do your own tests
# Predicted salary for 10 years of experience : 119347.82718107398

# Years of experience of the person you want to predict the salary for
years_experience = 10

# Predict the salary
input_data = pd.DataFrame({"YearsExperience": [years_experience]})
predicted_salary = regression.predict(input_data)[0]
# years_experience = [[10]]
# predicted_salary = regression.predict([[years_experience]])[0]

# Display the predicted salary
print(f"Predicted salary for {years_experience} years of experience: {predicted_salary}")
