from sklearn.neural_network import MLPClassifier

# or
X = [[0., 0.], [1., 1.],[0., 1.],[1., 0]]
y = [0, 1,1,1]
clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)
clf.fit(X, y)

out = clf.predict([[2., 2.], [-1., -2.]])
print(out)