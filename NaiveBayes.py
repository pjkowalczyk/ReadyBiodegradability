# Import Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB
import numpy as np

# assign predictor and target variables
X = np.array([[-3,7],[1,5], [1,2], [-2,0], [2,3], [-4,0], [-1,1], [1,1], [-2,2], [2,7], [-4,1], [-2,7]])
y = np.array([3, 3, 3, 3, 4, 3, 3, 4, 3, 4, 4, 4])

# Create a Gaussian Naive Bayes Classifier
model = GaussianNB()

# Train the model using the training sets 
model.fit(X, y)

# Predict output 
predicted= model.predict([[1,2],[3,4]])
print(predicted)
