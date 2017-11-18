
from random import random
from string import ascii_letters

choice = lambda x: x[int(random() * len(x))]

EXPECTED_STR = "test tested testo testy."  """ the script will used this table of caracters for a new generation"""


""" percentages"""

CHANCE_TO_MUTATE = 0.1 """ percentage of the mutation 10% """

GRADED_RETAIN_PERCENT = 0.2  """ percentage of high grad 20% """
CHANCE_RETAIN_NONGRATED = 0.05  """ percentage of low grad 5% """

""" constants"""
POPULATION_COUNT = 100
GENERATION_COUNT_MAX = 100000



def rand_population():

"""
return a list of POP_SIZE individuals , randomly genetayed

"""

population = []

for i in xrange():

    






