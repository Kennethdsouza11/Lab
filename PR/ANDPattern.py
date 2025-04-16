from sklearn.linear_model import Perceptron

X = [[0, 0], [1, 1], [1, 0], [0, 1]]
y = [0, 1, 0, 0]  # AND pattern

model = Perceptron(max_iter = 1000, eta0 = 0.1, random_state = 42)
model.fit(X,y)

for input_data in X:
    output = model.predict([input_data])
    print(f"Input : {input_data} -> Output : {output[0]}")