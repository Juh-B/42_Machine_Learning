# from google.colab import files
# uploaded = files.upload()
# session. Please rerun this cell to enable.
# Saving Salary_dataset.csv to Salary_dataset.csv

#This time pandas is forbidden, you have to use only numpy
import numpy as np

# ------------ Ex 00 ------------

# Read the data from the CSV file with read from Numpy
# ###
# loadtxt -> só analisa dados numericos,
# genfromtxt -> pega dados mistos (numericos, strings, etc)
# data = np.genfromtxt("Salary_dataset.csv", delimiter=',', names=True, dtype=None, encoding='utf-8')
# ###
data = np.loadtxt('sSalary_dataset.csv', delimiter=',', skiprows=1)

print(data)

# Separate the features (YearsExperience) from the target variable (Salary)
# ### 
# Variavel em maiuscula = constante
# ###
X = data[:, 1]
y = data[:, 2]

print(X, y)

#This time seaborn is forbidden, you have to find a library that works with numpy
# ###
# 2. matplotlib:
# Biblioteca para criar gráficos
# Aceita arrays NumPy diretamente para plotagem
# ###
import matplotlib.pyplot as plt

#You have to reproduce this graph
plt.show()

# ------------ Ex 01 ------------

# Let's create a function that displays the point line with the bar.
def visualize(theta, X, y):
    # y = β + βX + ϵ
    y_pred = theta[0] + theta[1] * X
    plt.plot(X, y_pred, color='purple')
    plt.scatter(X, y, color="#734CAD")
    plt.figure()

# Ok, let's test our function now, you should get a result comparable to this one

# np.zeros(2) -> Gera um array com 2 numeros zeros [0, 0]
theta = np.zeros(2)
visualize(theta, X, y)

# Create a function that multiplies each element of the matrix X by the slope of the model (theta[1]),
#followed by the addition of the intercept of the model (theta[0]), thus producing the predictions of the simple linear regression model.

def predict(X, theta):
    return theta[0] + theta[1] * X

# ###
# Gradiente Descendente -> gradient descent
# Atualiza o theta a cada interação para fazer com que a 
# previsão e o numero real se aproximem cada vez mais

# O gradiente descendente faz o seguinte repetidamente:
# 1. Calcula o erro da previsão atual
# 2. Vê o quanto esse erro muda em relação aos parâmetros (θ₀, θ₁)
# 3. Atualiza os parâmetros na direção que diminui o erro

# Fórmula da função de custo (MSE - Mean Squared Error):
# A função de custo (J) mais comum em regressão linear é:
#  𝐽(𝜃) = 1/2𝑚 m ∑ 𝑖=1 (ℎ𝜃(𝑥(**𝑖)) − 𝑦(**𝑖))**2
#  hθ(x) = θ0 + θ1x → a previsão do modelo
#  𝑦(**𝑖) → o valor real
#  m → número de exemplos (linhas no seu dataset)

# theta = theta - alpha * gradiente

# As derivadas (gradientes) parciais da função de custo (MSE) são:
# grad_0 = (1/m) * np.sum(error)
# grad_1 = (1/m) * np.sum(error * X)
# ###

def fit(X, y, theta, alpha, num_iters):
    # Initialize some useful variables
    m = X.shape[0]

    # Loop over the number of iterations
    for _ in range(num_iters):
        # Perform one iteration of gradient descent (i.e., update theta once)
        # y_pred = theta[0] + theta[1] * X
        # error = y_pred - y
        error = predict(X, theta) - y

        grad_0 = (1/m) * np.sum(error)
        grad_1 = (1/m) * np.sum(error * X)

        theta[0] = theta[0] - alpha * grad_0
        theta[1] = theta[1] - alpha * grad_1

    return theta


# To begin, we'll set alpha to 0.01 and num_iters to 1000

theta = np.zeros(2)
# finetuned_theta = fit(X, y, theta, 0.01, 1000)
finetuned_theta = fit(X, y, theta, 0.01, 0)
print(finetuned_theta)
#You should have a result similar to this one: [21912.58918422329, 9880.814004608217]

# Ok, let's test our function now, you should get a result comparable to this one
visualize(fit(X, y, theta, 0.01, 0), X, y)
visualize(fit(X, y, theta, 0.01, 1), X, y)
visualize(fit(X, y, theta, 0.01, 2), X, y)
visualize(fit(X, y, theta, 0.01, 3), X, y)
visualize(fit(X, y, theta, 0.01, 4), X, y)
visualize(fit(X, y, theta, 0.01, 1000), X, y)

def cost(X, y, theta):
    m = X.shape[0]
    # Calculate the difference between model predictions and actual target values
    diff_y = predict(X, theta) - y

    # Calculate the squared sum of the loss and scale it by 1/(2 * number of samples)
    # cost = (1 / (2 * m)) * np.sum(diff_y ** 2)
    cost = (1 / (2 * m)) * np.sum(np.square(diff_y))
    
    # Return the computed cost as a measure of model fit
    return cost


# Test it with theta = [0,0]. You should get approximately 3251553638.
cost_for_theta_zero = cost(X, y, [0, 0])
print(cost_for_theta_zero)

def fit_with_cost(X, y, theta, alpha, num_iters):
    m = X.shape[0]  # Number of training examples
    J_history = []  # List to store cost values at each iteration

    # Loop over the specified number of iterations
    for itr in range(num_iters):
        # Calculate the loss (difference between predictions and actual values)
        diff_y = predict(X, theta) - y

        # Update the temporary values of theta for both coefficients using the gradient descent formula
        grad_0 = (1/m) * np.sum(diff_y)
        grad_1 = (1/m) * np.sum(diff_y * X)

        # Update the theta values
        theta[0] = theta[0] - alpha * grad_0
        theta[1] = theta[1] - alpha * grad_1

        # Calculate and append the cost for the current theta values to the history list
        cost_value = cost(X, y, theta)
        J_history.append(cost_value)

        # Perform one iteration of gradient descent (update theta values)
        plt.plot(J_history)
        plt.xlabel("Iteration")
        plt.ylabel("Cost")
        plt.title("Cost over iterations")

    # Return the final theta values and the list of cost values over iterations
    return (theta, J_history)

# First, we initialize theta to zero
theta = np.zeros(2)

# Start the training using your new function
theta, J_history = fit_with_cost(X, y, theta, 0.001, 100)

# ------------ Ex 02 ------------

# Years of experience of the person you want to predict the salary for
years_experience = 10

# Predict the salary
predicted_salary = predict(years_experience, theta)

# Display the predicted salary
print("Predicted salary for {} years of experience {}".format(years_experience, predicted_salary))