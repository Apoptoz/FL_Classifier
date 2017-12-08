
import classifier 
from random import randint,random

#### GA VARIABLES ####
mutProb = .05
tournamentSize = 5
elitism = True
popSize = 20
GENERATION_NUMBER = 20
######################



class Indiv:
    def __init__(self,init=True):
        self.rules = []
        if init:
            for i in range(classifier.nbRules):
                self.rules.append(classifier.generateRule())
			
    def __str__(self):
        s = ""
        for i in range(classifier.nbRules):
            s +="Rule "+str(i)+": "+ str(self.rules[i])+"\t"
            s += str(classifier.getConf(self.rules[i])) + "\n"
        return s
		
    def getFitness(self):
        acc = classifier.getAccuracy(self)
        goodRulesNb,badRulesNb = classifier.checkRules(self)
        complexity = classifier.calcComplexity(self)

        w1 = 0.6
        w2 = 0.4

        score = w1*(1.0-acc) + w2*(float(complexity) / float(len(self.rules)*4))
        # this maximizes the accuracy and minimizes "complexity"
        score = -1 * score

        
        return score

        #1 - nbofaccuratelyclassified / number of cases find
        
        #inferences = classifier.infer(self.rules)
        #inferences = classifier.simple_infer(self.rules)
        #return classifier.computeFitness(inferences)
   
class Population:


    """ x = Indiv()
    x.getfit()"""
    
    def __init__(self,init,size=popSize):
      if init:
         self.listpop = [Indiv() for _ in range(size)]
      else:
         self.listpop = []         

    def getFittest(self):   
      nb_max = self.listpop[0].getFitness()
      index = 0
      
      for i in range(1,len(self.listpop)):
          nextFitness = self.listpop[i].getFitness()
          if nextFitness > nb_max:
            nb_max = nextFitness         
            index = i      
      return self.listpop[index]
      
      
def tournament(pop):

   tourList = Population(False,tournamentSize)
   
   
   for j in range(tournamentSize):
   
      
      indexT = randint(0,popSize-1)
      
      pop.listpop[indexT]
      tourList.listpop.append(pop.listpop[indexT])
      
   return tourList.getFittest()
   
    
   

def crossOver(Indiv1,Indiv2):
   
    newIndiv = Indiv(False)
   
    for i in range(classifier.nbRules):
   
        rule1 = Indiv1.rules[i]
      
        rule2 = Indiv2.rules[i]
            
        newIndiv.rules.append(crossoverRules(rule1,rule2))

    return newIndiv
      
def crossoverRules(rule1,rule2):
   
   newRule = [] 
      
   for i in range(len(rule1)):
      prob = random()
      if prob < 0.5: 
         newRule.append(rule1[i])
      
      else:
         newRule.append(rule2[i])
         
   return newRule      
      


def mutation(indiv):


   for i in range(classifier.nbRules):
   
      for j in range(classifier.nbRules):
         
         prob = random()
         
         if   prob < mutProb:
             indiv.rules[i][j] = 1 - indiv.rules[i][j]
            
if __name__ == '__main__':
             
    pop = Population(True)


    for i in range(GENERATION_NUMBER):

        newpop = Population(False)
        print("Generation number : "+str(i))
   
        for j in range(popSize):
       
            parent1 = tournament(pop)
            parent2 = tournament(pop)
      
            child = crossOver(parent1,parent2)
            newpop.listpop.append(child)
      
        for j in range(popSize):
            mutation(newpop.listpop[j])
       
        pop = newpop
        
        thisFittest = pop.getFittest()
        print("Fittest accuracy : " + str(classifier.getAccuracy(thisFittest)))
        print(thisFittest)
