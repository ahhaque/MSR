from numpy import *
from sklearn import linear_model

class Model(object):
	def __init__(self):
		self.model = None

	"""
	Initialize training of a new weighted linear regression model by choosing best parameters.
	Sets the trained model for this object.
	"""
	def trainUsingWeights(self, traindata, trainLabels, weightSrcData):
		self.model = linear_model.LinearRegression(fit_intercept=True, normalize=False, copy_X=True, n_jobs=1)
		self.model.fit(traindata, trainLabels, weightSrcData)

	"""
	Test the weighted SVM to predict labels of a given test data.
	Returns the result of prediction, and confidence behind the prediction
	"""
	def test(self, testdata):
		if len(testdata) == 1:
			testdata = np.reshape(testdata, (1, -1))
		predictions = self.model.predict(testdata)
		return predictions

	"""
	Get summary of the model
	"""
	def getModelSummary(self):
		summary = '************************* Model   S U M M A R Y ************************\n'
		summary += 'Coefficients: ' + self.model.coef_.tolist()
		summary += '\nIntercept: ' + self.model.intercept_
		return summary