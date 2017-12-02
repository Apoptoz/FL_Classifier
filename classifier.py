import data
from random import randint
import numpy as np
from sklearn.metrics import accuracy_score

from functools import reduce

#### GA VARIABLES ####
mutProb = .15
tournamentSize = 5
elitism = True
######################

WEIGHT_ERROR = 0.2

alpha_cut = 0

nbRules = 10



truthCache = {}
inferenceCache = {}


#Class who can compute Triangle functions
#We might need to edit it to enter a mid point
class Triangle:
    def __init__(self,min,max):
        self.min = min
        self.max = max
        self.alpha_cut = alpha_cut
        
    def valAt(self,x): 
        alpha = 2/(self.max-self.min)
        f = lambda x: alpha*(x-self.min)
        g = lambda x: alpha*(-x+self.max)
        mid = (self.min+self.max)/2
        if(x<self.min or x>self.max):
            return 0.0
        elif(x<mid):
            return f(x)
        else:
            return g(x)
    def at(self,x):
        val = self.valAt(x)
        return val if val >= alpha_cut else 0


#For the moment all parameters share the same membership functions
    
#### MEMBERSHIP FUNCTIONS ####
# But x will only be in [0;1]
smallTriangle = Triangle(-0.5,0.5)
medTriangle = Triangle(0,1)
largeTriangle = Triangle(0.5,1.5)
##############################



    
class Indiv:
    def __init__(self):
        self.rules = []
        for i in range(nbRules):
            self.rules.append(generateRule())
    def __str__(self):
        s = ""
        for i in range(nbRules):
            s +="Rule "+str(i)+": "+ self.rules[i]+"\n"
        return s

#Generate random rules
def generateRule():
    randBits = []
    randRule = randint(0,pow(2,12)-1)
    rule = "{0:b}".format(randRule)

    randClass = randint(0,2)
    
    for i in range(12-len(rule)):
        randBits.append(0)
    for i in range(len(rule)):
        randBits.append(int(rule[i]))
    return randBits
    '''
    if (randClass == 0):
        randBits += [0,0,1]
    elif(randClass == 1):
        randBits += [0,1,0]
    else:
        randBits += [1,0,0]
    '''



def getCompetitionStrength(rule):
    competitionStrength = [0,0,0]
    
    for classNumber in range(3):
        classArray = data.getClassArray(classNumber)
        competitionStrength[classNumber] = sum( [ getMuA(rule,row) for _,row in classArray.iterrows()] )

    return competitionStrength
    


#This function return the class number and the truth degree of that class based on the training function.
def getConf(rule):
    #Transform [0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1] in '001001010001'
    hashedRule = "".join(str(i) for i in rule)
    if hashedRule in truthCache:
        return truthCache[hashedRule]
    else:
        #Go through the training data to get competitionStrength
        competitionStrength = getCompetitionStrength(rule)
        #Divide by the sum to get truth degree (between 0 and 1)
        strSum = sum(competitionStrength)
        if strSum != 0:
            truthDegree = [i/strSum for i in competitionStrength]
            #Get the class with the best value, and its value
            maxIndex,maxValue = max(enumerate(truthDegree),key=lambda x:x[1])
        else:
            maxIndex,maxValue = (-1,0) #No classes are recognized     
        
        truthCache[hashedRule] = (maxIndex,maxValue)
        return (maxIndex,maxValue)

#Get the best class and its confidence degree for each rule
def getConfVect(rules):
    return [getConf(rule) for rule in rules]



def getTruth(rule):
    ruleString = str(rule)
    if ruleString in truthCache:
        return truthCache[ruleString] 

    competitionStrength = getCompetitionStrength(rule)
    sumComp = sum(competitionStrength)
    if sumComp == 0: #This rule correspond to no class
        truthCache[ruleString] = [-1,0]
        return [-1,0]
    else:
        index, value = max(enumerate([competitionStrength[0]/sumComp,competitionStrength[1]/sumComp,competitionStrength[2]/sumComp]), key = lambda e: e[1])
        truthCache[ruleString] = [index, value]
        return [index, value]

def toCacheString(rule,data_row):
    strRule = "".join(str(i) for i in rule)
    strRow = ""
    for x in range(len(data_row)):
        strRow += "%0.3f" %data_row[x]
    return strRule+strRow


def getMuA(rule,data_row):
    cacheString = toCacheString(rule,data_row)
    if cacheString in inferenceCache:
        return inferenceCache[cacheString]
    else:
        maxArray = []
        ruleCounter = 0
        for x in range(0, len(data_row)):
            datum = data_row[x]
            if rule[ruleCounter:ruleCounter+3] != [0,0,0]:
                small = 0
                medium = 0
                large = 0
                if rule[ruleCounter] == 1:
                    small = smallTriangle.at(datum)
                if rule[ruleCounter+1] == 1:
                    medium = medTriangle.at(datum)
                if rule[ruleCounter+2] == 1:
                    large = largeTriangle.at(datum)
                maxArray.append(max(small,medium,large))
            ruleCounter += 3
        if maxArray == []:
            muA = 0 #I'm not sure about that
        else:
            muA = min(maxArray)
        inferenceCache[cacheString] = muA
        return muA

def getMuAVect(rules,data_row):
    return [getMuA(rule,data_row) for rule in rules]


def getPredictedConfVect(confVect,muAVect):
    predictedConfVect = [0,0,0]
    cnt = [1,1,1]


    for i in range(len(confVect)):
        ruleClass,ruleConf = confVect[i]
        if ruleClass != -1:
            predictedConfVect[ruleClass] += muAVect[i]*ruleConf
            cnt[ruleClass] += 1

    averagedPredictedConfVect = [predictedConfVect[i]/cnt[i] for i in range(3)]
    return averagedPredictedConfVect

def getPredictedClass(rules,data_row):
    predictedConfVect = getPredictedConfVect(getConfVect(rules),getMuAVect(rules,data_row))
    predictedClass,predictedConf = max(enumerate(predictedConfVect),key=lambda x:x[1])
    return predictedClass,predictedConf


def getPredictedClasses(indiv,data):
    predictedClassArray = []
    for _,data_row in data.iterrows():
        predictedClass,predictedConf = getPredictedClass(indiv.rules,data_row)
        predictedClassArray.append(predictedClass)
    return predictedClassArray


def getAccuracy(indiv):
    predictedClassArray = getPredictedClasses(indiv,data.X_test)
    score = accuracy_score(data.y_test,predictedClassArray)
    return score


    
#Compute µ_a given a u_i and a rule
def getMuAPast(muArray,rule):

    #Multiply the muArray with the rule to "eliminate" the unused fuzzy sets

    ####
    #That was really a bad idea: because you then take the min and max of values of 0 that should not be used.
    ####
    '''
    
    muValues = [muArray[i]*rule[i] if rule[i]==1 else -1 for i in range(12)]
    print(muValues)
    #Then, make an array for all the max membership values of each parameters
    #And take the min of that array
    '''
    maxArray = []
    
    for i in [0,3,6,9]:
        if rule[i:i+3] != [0,0,0]: #Don't care about that parameter
            maxArray.append(max([ muArray[j]*rule[j] for j in range(i,i+3)]))
    
    #muA = min( [ max(muValues[i:i+3]) for i in [0,3,6,9] ] )
    if maxArray == []: #Incorrect rule?
        return -1
    muA = min(maxArray)
    return muA
               

#We might prefer to preprocess muArray for all elements
#So we don't compute it each times

#Return the class predicted by the rule
def getClassFromRule(rule):
    #Compute how well does that class for this rule
    competitionStrength = [0,0,0]
    for classNumber in range(3):
        classArray = data.getClassArray(classNumber)
        competitionStrength[classNumber] += np.dot(classArray,rule)
    maxIndex = getMaxIndex(competitionStrength)
    print("The class of this rule is class " + str(maxIndex) + \
          " with a score of: " + str(competitionStrength[maxIndex]))
    return maxIndex,competitionStrength

# or just max(enumerate(a),key=lambda x: x[1])[0]
def getMaxIndex(l):
    if len(l)==1:
        return 0
    else:
        max = l[0]
        index = 0
        for i in range(1,len(l)):
            if l[i] > max:
                max = l[i]
                index = i
        return index



###IMPLEMENTATION###
#It would be better to preprocess all the muArrays from the training data,
#And to label them with their class
#So we can use panda to take each and not recalculate them
#Return an array of all values returned by membership functions
def getMuArray(row):
    muArray = []
    x1 = row['SepalLength']
    x2 = row['SepalWidth']
    x3 = row['PetalLength']
    x4 = row['PetalWidth']

    muArray += [smallTriangle.at(x1),medTriangle.at(x1),largeTriangle.at(x1)]
    muArray += [smallTriangle.at(x2),medTriangle.at(x2),largeTriangle.at(x2)]
    muArray += [smallTriangle.at(x3),medTriangle.at(x3),largeTriangle.at(x4)]
    muArray += [smallTriangle.at(x4),medTriangle.at(x4),largeTriangle.at(x4)]
    return muArray



def simple_infer(rules):
    
    inferences = []
    for index,datum in data.X_train.iterrows():
        classes = [[],[],[]]
        inf = [0,0,0]
        
        for rule in rules:

            #Cache

            if cache not in inferenceCache:
                inferenceCache[cache] = getMuA(getMuArray(datum),rule)
            inferred = inferenceCache[cache]
            #print(inferred)

            inferred = getMuA(getMuArray(datum),rule)
            
            confidence = getTruth(rule)
            classes[confidence[0]].append(inferred)
        for i in range(len(classes)):
            if classes[i] != []:
                inf[i] += sum(classes[i])/len(classes[i])
        inferences.append(max(enumerate(inf), key=lambda x:x[1])[0])
    return inferences

# Infers classes for all data using the provided rules
# Returns list of lists of the three strengths of memberships to classes 1, 2 and 3
# Where class1 is at index 0, class2 at index 1, and class3 at index 2 of each list

def infer(rules, forAccuracy = False):
    class1 = []
    class2 = []
    class3 = []

    classes = [class1, class2, class3]
    inferences = []

    for index,datum in data.X_test.iterrows():
        for rule in rules:
            ruleString = str(rule)
            if ruleString not in inferenceCache:
                inferenceCache[ruleString] = getMuA(getMuArray(datum),rule)
                #inferenceCache[ruleString] = applyRule(rule, datum)
            inferred = inferenceCache[ruleString]
            confidence = getTruth(rule)
            #nferred = confidences[i]*getMuA(getMuArray(datum),rule)
            classes[confidence[0]].append(inferred*confidence[1])
        
        if class1:
            class1_inferred = reduce(lambda x, y: x + y, classes[0]) / len(classes[0])
        else:
            class1_inferred = 0
        if class2:
            class2_inferred = reduce(lambda x, y: x + y, classes[1]) / len(classes[1])
        else:
            class2_inferred = 0
        if class3:
            class3_inferred = reduce(lambda x, y: x + y, classes[2]) / len(classes[2])
        else:
            class3_inferred = 0

        l = [class1_inferred, class2_inferred, class3_inferred]

        if forAccuracy:
            inferences.append(l)
        else:
            inferences.append(l[data.y_train.iloc[index]]) 
        class1 = []
        class2 = []
        class3 = []

    # following code gives the value and index (class) for each inference

    #for inference in inferences:
    #    index, value = max(enumerate(inference), key = lambda e: e[1])

    return inferences


def getAccuracyPast(inferences):
    total = len(inferences)
    correct = 0
    weightedCorrect = 0.0
    for i in range(len(inferences)):
        index, value = max(enumerate(inferences[i]), key = lambda e: e[1])
        if index == data.y_train.iloc[i]:
            correct += 1
            weightedCorrect += value
    accuracy = float(correct) / float(total)
    weightedAccuracy = weightedCorrect / float(total)
    print("Accuracy: ", accuracy)
    print("Weighted Accuracy: ", weightedAccuracy)


def computeFitness(inferences):
    #print("ComputeFit")
    fitDatum = []
    for confidence in inferences:
        fitDatum.append(confidence)
    return sum(fitDatum)


def applyRule(rule, data):
    ruleCounter = 0
    calcedRuleArray = []
    for x in range(0, len(data)):
        datum = data[x]
        small = 0
        medium = 0
        large = 0
        

        if rule[ruleCounter] == 1:
            small = smallTriangle.at(datum)
        if rule[ruleCounter+1] == 1:
            medium = medTriangle.at(datum)
        if rule[ruleCounter+2] == 1:
           large = largeTriangle.at(datum)
        calcedRuleArray.append(max(small, medium, large))
        ruleCounter += 3
    calcedRule = min(calcedRuleArray[0], calcedRuleArray[1], calcedRuleArray[2])
    finalInference = calcedRule
    return finalInference


# We wanted to use 3confidences per rule to compute data, then have some kind of average for each class. But that seems to take too much time.
# So we will try to only take the class corresponding to the max confidence, and compute the accuracy.
# We kind of lose the fuzziness


# For each rule compute the truth value, take the max class.
# Infer with the data, get a new confidence for that class.
# Average for each class, add the confidence of the correct answer to fitness
# YAY


# change getTruth to only return the max of the three along with the class index
# change infer apporpriately.
# compare to true values (y_test), make array of the calculated confidences for all those
# sum them to get Fitness value
