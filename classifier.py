import data
from random import randint
import numpy as np

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



def getMuArrays(classNumber=-1):
    muArrayList = []
    
    if classNumber == -1:
        array = data.X_train
    else:
        array = data.getClassArray(classNumber)
    for _,row in array.iterrows():
        muArrayList.append(getMuArray(row))
    return muArrayList

def getCompetitionStrength(rule):
    competitionStrength = [0,0,0]

    for classNumber in range(3):
        muArrays = getMuArrays(classNumber)

        competitionStrength[classNumber] = sum( [ getMuA(muArrays[i],rule) for i in range(len(muArrays)) ] )

    return competitionStrength



def getTruth(rule):
    ruleString = str(rule)
    if ruleString in truthCache:
        return truthCache[ruleString] 

    competitionStrength = getCompetitionStrength(rule)
    sumComp = sum(competitionStrength)
    if sumComp == 0:
        truthCache[ruleString] = [0,0]
        return [0,0]
    else:
        index, value = max(enumerate([competitionStrength[0]/sumComp,competitionStrength[1]/sumComp,competitionStrength[2]/sumComp]), key = lambda e: e[1])
        truthCache[ruleString] = [index, value]
        return [index, value]

#Compute µ_a given a u_i and a rule
def getMuA(muArray,rule):

    #Multiply the muArray with the rule to "eliminate" the unused fuzzy sets
    muValues = [muArray[i]*rule[i] for i in range(12)]
    #Then, make an array for all the max membership values of each parameters
    #And take the min of that array
    muA = min( [ max(muValues[i:i+3]) for i in [0,3,6,9] ] )
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

# Infers classes for all data using the provided rules
# Returns list of lists of the three strengths of memberships to classes 1, 2 and 3
# Where class1 is at index 0, class2 at index 1, and class3 at index 2 of each list

def infer(rules):
    class1 = []
    class2 = []
    class3 = []

    classes = [class1, class2, class3]
    inferences = []

    for index,datum in data.X_test.iterrows():
        for rule in rules:
            ruleString = str(rule)
            if ruleString not in inferenceCache:
                inferenceCache[ruleString] = applyRule(rule, datum)
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
        inferences.append(l[data.y_train.iloc[index]]) 
        class1 = []
        class2 = []
        class3 = []

    # following code gives the value and index (class) for each inference

    #for inference in inferences:
    #    index, value = max(enumerate(inference), key = lambda e: e[1])

    return inferences


def computeFitness(inferences):
    print("ComputeFit")
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
