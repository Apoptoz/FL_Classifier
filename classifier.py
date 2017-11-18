import data
from random import randint
import numpy as np

from functools import reduce

#### GA VARIABLES ####
mutProb = .15
tournamentSize = 5
elitism = True
######################


alpha_cut = 0

nbRules = 10




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

        competitionStrength[classNumber] = sum( [ np.dot(muArrays[i],rule) for i in range(len(muArrays)) ] )

    return competitionStrength



def getTruth(rule):
    competitionStrength = getCompetitionStrength(rule)
    sumComp = sum(competitionStrength)
    return [competitionStrength[0]/sumComp,competitionStrength[1]/sumComp,competitionStrength[2]/sumComp]

               
               
               

#We might prefer to preprocess muArray for all elements
#So we don't compute it each times

#Return the class predicted by the rule
def getClassFromRule(rule):
    #Compute how well does that class for this rule
    competitionStrength = [0,0,0]
    for classNumber in range(3):
        classArray = data.getClassArray(classNumber)
        competitionStrength[classNumber] += np.dot(classArray,rule)
        #Idée : créer array de 12
        # µs(x1),µm(x1)...,µl(xm)
        # Produit scalaire avec rule
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

def infer(rules, data):
    class1 = []
    class2 = []
    class3 = []

    classes = [class1, class2, class3]
    inferences = []

    for datum in data:
        for rule in rules:
            confidences = getTruth(rule)
            for i in range(0, len(confidences)):
                inferred = applyRule(rule, confidences[i], datum)
                classes[i].append(inferred)
        class1_inferred = reduce(lambda x, y: x + y, classes[0]) / len(classes[0])
        class2_inferred = reduce(lambda x, y: x + y, classes[1]) / len(classes[1])
        class3_inferred = reduce(lambda x, y: x + y, classes[2]) / len(classes[2])
        inferences.append([class1_inferred, class2_inferred, class3_inferred]) 
        class1 = []
        class2 = []
        class3 = []

    # following code gives the value and index (class) for each inference

    #for inference in inferences:
    #    index, value = max(enumerate(inference), key = lambda e: e[1])

    return inferences


def applyRule(rule, confidence, data):
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
    finalInference = calcedRule * confidence
    return finalInference
