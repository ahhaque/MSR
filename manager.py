from __future__ import print_function
from properties import Properties
from kliep import Kliep
from regressionModel import Model
from stream import Stream
from sklearn import svm, grid_search
import time, sys, datetime
import numpy as np
import random, math
import gaussianModel as gm
#from py4j.java_gateway import JavaGateway, GatewayParameters, CallbackServerParameters

class Manager(object):
	def __init__(self, sourceFile, targetFile):
		self.SDataBufferArr = None #2D array representation of self.SDataBuffer
		self.SDataLabels = None
		self.TDataBufferArr = None #2D array representation of self.TDataBuffer
		self.TDataLabels = None

		self.useKliepCVSigma = Properties.useKliepCVSigma

		self.kliep = None

		self.regModel = Model()

		self.initialWindowSize = int(Properties.INITIAL_DATA_SIZE)
		self.maxWindowSize = int(Properties.MAX_WINDOW_SIZE)

		self.enableForceUpdate = int(Properties.enableForceUpdate)
		self.forceUpdatePeriod = int(Properties.forceUpdatePeriod)

		"""
		- simulate source and target streams from corresponding files.
		"""
		print("Reading the Source Dataset")
		self.source = Stream(sourceFile, Properties.INITIAL_DATA_SIZE)
		print("Reading the Target Dataset")
		self.target = Stream(targetFile, Properties.INITIAL_DATA_SIZE)
		print("Finished Reading the Target Dataset")

		Properties.MAXVAR = self.source.initialData.shape[0]

	"""
	Write value (accuracy or confidence) to a file with DatasetName as an identifier.
	"""
	def __saveResult(self, acc, datasetName):
		with open(datasetName + '_' + Properties.OUTFILENAME, 'a') as f:
			f.write(str(acc) + "\n")
		f.close()

	"""
	The main method handling multistream classification using KLIEP.
	"""
	def startFusionRegression(self, datasetName, probFromSource):
		#save the timestamp
		globalStartTime = time.time()
		Properties.logger.info('Global Start Time: ' + datetime.datetime.fromtimestamp(globalStartTime).strftime('%Y-%m-%d %H:%M:%S'))
		#open files for saving accuracy and confidence
		fAcc = open(datasetName + '_' + Properties.OUTFILENAME, 'w')
		fConf = open(datasetName + '_confidence' + '_' + Properties.OUTFILENAME, 'w')
		#initialize gaussian models
		gmOld = gm.GaussianModel()
		gmUpdated = gm.GaussianModel()
		#variable to track forceupdate period
		idxLastUpdate = 0

		#Get data buffer
		self.SDataBufferArr = self.source.initialData
		self.SDataLabels = self.source.initialDataLabels

		self.TDataBufferArr = self.target.initialData

		#first choose a suitable value for sigma
		self.kliep = Kliep(Properties.kliepParEta, Properties.kliepParLambda, Properties.kliepParB, Properties.kliepParThreshold, Properties.kliepDefSigma)
		#self.kliep = Kliep(Properties.kliepParEta, Properties.kliepParLambda, Properties.kliepParB, Properties.MAXVAR*Properties.kliepParThreshold, Properties.kliepDefSigma)

		if self.useKliepCVSigma==1:
			self.kliep.kliepDefSigma = self.kliep.chooseSigma(self.SDataBufferArr, self.TDataBufferArr)

		#calculate alpha values
		Properties.logger.info('Estimating initial DRM')
		gmOld.alphah, kernelMatSrcData, kernelMatTrgData, gmOld.refPoints = self.kliep.KLIEP(self.SDataBufferArr, self.TDataBufferArr)
		#initialize the updated gaussian model
		gmUpdated.setAlpha(gmOld.alphah)
		gmUpdated.setRefPoints(gmOld.refPoints)
		#now resize the windows appropriately
		self.SDataBufferArr = self.SDataBufferArr[:, -Properties.MAX_WINDOW_SIZE:]
		self.SDataLabels = self.SDataLabels[-Properties.MAX_WINDOW_SIZE:]

		self.TDataBufferArr = self.TDataBufferArr[:, -Properties.MAX_WINDOW_SIZE:]

		kernelMatSrcData = kernelMatSrcData[-Properties.MAX_WINDOW_SIZE:,:]
		kernelMatTrgData = kernelMatTrgData[-Properties.MAX_WINDOW_SIZE:,:]

		Properties.logger.info('Initializing Ensemble with the first model')
		#target model
		#first calculate weight for source instances
		weightSrcData = self.kliep.calcInstanceWeights(kernelMatSrcData, gmUpdated.alphah)
		#since weightSrcData is a column matrix, convert it to a list before sending to generating new model
		SDataBufferArrTransposed = self.SDataBufferArr.T

		self.regModel.trainUsingWeights(self, SDataBufferArrTransposed.tolist(), self.SDataLabels, weightSrcData[0].tolist())

		Properties.logger.info(self.regModel.model.getModelSummary())

		sDataIndex = 0
		tDataIndex = 0
		totError = 0.0

		#enoughInstToUpdate is used to see if there are enough instances in the windows to
		#estimate the weights

		Properties.logger.info('Starting MultiStream Classification with FUSION')
		while self.target.data.shape[1] > tDataIndex:
			"""
			if source stream is not empty, do proper sampling. Otherwise, just take
			the new instance from the target isntance.
			"""
			if self.source.data.shape[1] > sDataIndex:
				fromSource = random.uniform(0,1)<probFromSource
			else:
				print("\nsource stream sampling not possible")
				fromSource = False

			if fromSource:
				print('.', end="")
				#remove the first instance, and add the new instance in the buffers
				newSrcDataArr = self.source.data[:, sDataIndex][np.newaxis].T
				self.SDataBufferArr = self.SDataBufferArr[:, 1:]
				self.SDataLabels = self.SDataLabels[1:]
				kernelMatSrcData = kernelMatSrcData[1:, :]
				#add new instance to the buffers
				self.SDataBufferArr = np.append(self.SDataBufferArr, newSrcDataArr, axis=1)
				self.SDataLabels.append(self.source.dataLabels[sDataIndex])

				#update kernelMatSrcData
				dist_tmp = np.power(np.tile(newSrcDataArr, (1, gmUpdated.refPoints.shape[1])) - gmUpdated.refPoints, 2)
				dist_2 = np.sum(dist_tmp, axis=0, dtype='float64')
				kernelSDataNewFromRefs = np.exp(-dist_2 / (2 * math.pow(self.kliep.kliepDefSigma, 2)), dtype='float64')
				kernelMatSrcData = np.append(kernelMatSrcData, kernelSDataNewFromRefs[np.newaxis], axis=0)

				#print("Satisfying the constrains.")
				gmUpdated.alphah, kernelMatSrcData = self.kliep.satConstraints(self.SDataBufferArr, self.TDataBufferArr, gmUpdated.refPoints, gmUpdated.alphah, kernelMatSrcData)
				sDataIndex += 1
			else:
				# Target Stream
				print('#', end="") # '#' indicates new point from target
				newTargetDataArr = self.target.data[:, tDataIndex][np.newaxis].T
				# get Target Accuracy on the new instance
				resTarget = self.regModel.test(np.reshape(newTargetDataArr, (1,-1)))
				totError = abs(resTarget-self.target.dataLabels[tDataIndex])
				avgError = float(totError)/(tDataIndex+1)
				if (tDataIndex%100)==0:
					Properties.logger.info('\nTotal test instance: '+ str(tDataIndex+1) + ', Total Error: ' + str(totError) + ', Average Error: ' + str(avgError))
				fAcc.write(str(avgError)+ "\n")

				#update alpha, and satisfy constraints
				gmUpdated.alphah, kernelMatSrcData = self.kliep.updateAlpha(self.SDataBufferArr, self.TDataBufferArr, newTargetDataArr, gmUpdated.refPoints, gmUpdated.alphah, kernelMatSrcData)

				#remove the first instance from buffers
				self.TDataBufferArr = self.TDataBufferArr[:, 1:]
				#update ref points
				gmUpdated.refPoints = gmUpdated.refPoints[:, 1:]
				# update kernelMatSrcData, as ref points has been updated
				kernelMatSrcData = kernelMatSrcData[:, 1:]
				# update kernelMatTrgData, as ref points has been updated
				kernelMatTrgData = kernelMatTrgData[1:, 1:]

				#update ref points
				gmUpdated.refPoints = np.append(gmUpdated.refPoints, newTargetDataArr, axis=1)

				#add to kernelMatSrcData for the last ref point
				dist_tmp = np.power(
					np.tile(newTargetDataArr,
							(1, self.SDataBufferArr.shape[1])) - self.SDataBufferArr, 2)
				dist_2 = np.sum(dist_tmp, axis=0, dtype='float64')
				kernel_dist_2 = np.exp(-dist_2 / (2 * math.pow(self.kliep.kliepDefSigma, 2)), dtype='float64')
				kernelMatSrcData = np.append(kernelMatSrcData, kernel_dist_2[np.newaxis].T, axis=1)
				#now update kernelMatTrgData, as ref points has been updated
				#first add distance from the new ref points to all the target points
				dist_tmp = np.power(
					np.tile(newTargetDataArr,
							(1, self.TDataBufferArr.shape[1])) - self.TDataBufferArr, 2)
				dist_2 = np.sum(dist_tmp, axis=0, dtype='float64')
				kernel_dist_2 = np.exp(-dist_2 / (2 * math.pow(self.kliep.kliepDefSigma, 2)), dtype='float64')
				kernelMatTrgData = np.append(kernelMatTrgData, kernel_dist_2[np.newaxis].T, axis=1)

				#now add distances for the newly added instance to all the ref points
				#add the new instance to the buffers
				self.TDataBufferArr = np.append(self.TDataBufferArr, newTargetDataArr, axis=1)

				dist_tmp = np.power(np.tile(newTargetDataArr, (1, gmUpdated.refPoints.shape[1])) - gmUpdated.refPoints, 2)
				dist_2 = np.sum(dist_tmp, axis=0, dtype='float64')
				kernelTDataNewFromRefs = np.exp(-dist_2 / (2 * math.pow(self.kliep.kliepDefSigma, 2)), dtype='float64')
				kernelMatTrgData = np.append(kernelMatTrgData, kernelTDataNewFromRefs[np.newaxis], axis=0)

				tDataIndex += 1

			#print "sDataIndex: ", str(sDataIndex), ", tDataIndex: ", str(tDataIndex)
			enoughInstToUpdate = self.SDataBufferArr.shape[1]>=Properties.kliepParB and self.TDataBufferArr.shape[1]>=Properties.kliepParB
			if enoughInstToUpdate:
				#print("Enough points in source and target sliding windows. Attempting to detect any change of distribution.")
				changeDetected, changeScore, kernelMatTrgData = self.kliep.changeDetection(self.TDataBufferArr, gmOld.refPoints, gmOld.alphah, gmUpdated.refPoints, gmUpdated.alphah, kernelMatTrgData)
				#print("Change Score: ", changeScore)

			#instances from more than one class are needed for svm training
			if len(set(self.SDataLabels))>1 and (changeDetected or (self.enableForceUpdate and (tDataIndex + sDataIndex - idxLastUpdate)>self.forceUpdatePeriod)): #or (tDataIndex>0 and (targetConfSum/tDataIndex)<0.1):
				fConf.write(str(7777777.0) + "\n")
				Properties.logger.info(
					'\n-------------------------- Change of Distribution ------------------------------------')
				Properties.logger.info('Change of distribution found')
				Properties.logger.info(
					'sDataIndex=' + str(sDataIndex) + '\ttDataIndex=' + str(tDataIndex))
				Properties.logger.info('Change Detection Score: ' + str(changeScore) + ', Threshold: ' + str(self.kliep.kliepParThreshold))

				#Build a new model
				#First calculate the weights for each source instances
				gmOld.alphah, kernelMatSrcData, kernelMatTrgData, gmOld.refPoints = self.kliep.KLIEP(self.SDataBufferArr,
																								   self.TDataBufferArr)
				#update the updated gaussian model as well
				gmUpdated.setAlpha(gmOld.alphah)
				gmUpdated.setRefPoints(gmOld.refPoints)

				weightSrcData = self.kliep.calcInstanceWeights(kernelMatSrcData, gmUpdated.alphah)
				#Build a new model
				Properties.logger.info('Training a model due to change detection')
				SDataBufferArrTransposed = self.SDataBufferArr.T

				self.regModel.trainUsingWeights(self, SDataBufferArrTransposed.tolist(), self.SDataLabels,
												weightSrcData[0].tolist())

				Properties.logger.info(self.regModel.model.getModelSummary())

				#update the idx
				idxLastUpdate = tDataIndex + sDataIndex
				changeDetected = False
		#save the timestamp
		fConf.close()
		fAcc.close()
		globalEndTime = time.time()
		Properties.logger.info(
			'\nGlobal Start Time: ' + datetime.datetime.fromtimestamp(globalEndTime).strftime('%Y-%m-%d %H:%M:%S'))
		Properties.logger.info('Total Time Spent: ' + str(globalEndTime-globalStartTime) + ' seconds')
		Properties.logger.info('Done !!')