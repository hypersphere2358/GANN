import numpy as np
import random
import math
import pygame
from pygame.locals import *
from operator import itemgetter

"""
This version of neural network has fixed topology.
It makes changes only on weights of the network by genetic algorithm.
"""

# INFO_FONT = pygame.font.Font('freesansbold.ttf', 9)

# set up the colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
BORN = (255, 0, 0)

class NeuralNetwork:
    def __init__(self, input_n, hidden_layer_n, output_n):


        self.input_n = input_n
        self.hidden_layer_n = hidden_layer_n
        self.output_n = output_n

        # calculate genome length
        self.genome_length = (input_n * input_n) * hidden_layer_n + (input_n * output_n)

        # make genome. each gene value means weight of the topology
        # self.target_genome = [2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.0, -1.0, 2.0, -1.0]
        # self.genome = self.target_genome
        self.genome = []
        for i in range(self.genome_length):
            self.genome.append(random.gauss(0, 1.0))

        self.genome_to_table()
        self.fitness = 0.0

    def print_genome(self):
        print(self.genome)

    def target_fitness(self, genome):
        if len(genome) != len(self.target_genome):
            print("Error: genome sizes are different - can't calculate fitness")
            exit(1)

        return -sum(abs(c - t) for c, t in zip(genome, self.target_genome))

    def genome_to_table(self):
        self.path_table = []
        for i in range(self.hidden_layer_n + 1):
            table1 = []
            if i < self.hidden_layer_n:
                for j in range(self.input_n):
                    table1.append(self.genome[i * (self.input_n * self.input_n) + (j * self.input_n):i * (self.input_n * self.input_n) + ((j+1) * self.input_n)])
            else:
                for j in range(self.input_n):
                    table1.append(self.genome[i * (self.input_n * self.input_n) + (j * self.output_n):i * (self.input_n * self.input_n) + ((j + 1) * self.output_n)])
            self.path_table.append(table1)

        # print(self.path_table)

    def sigmoid_discrete(self, x, cut):
        if x >= cut:
            return 1.0
        else:
            return 0.0

    def sigmoid_continuous(self, x):
        # return (2.0 / (1 + math.exp(-4.0 * x))) - 1.0
        # print(x)
        if x <= -170.0:
            x = -170.0
        if x >= 170.0:
            x = 170.0
        return (1.0 / (1 + math.exp(-4.0 * x)))

    def processing(self, input):
        if self.input_n != len(input):
            print("Error! - wrong input size")
            exit(1)

        mid_input = input

        # print("input: %s" % str(input))
        final_input = []
        for i in range(len(self.path_table)):
            mid_output = []

            s = [0.0] * len(self.path_table[i][0])
            for j in range(len(self.path_table[i])):

                for k in range(len(self.path_table[i][j])):
                    s[k] += self.path_table[i][j][k] * mid_input[j]

            for j in s:
                t = j / len(mid_input)
                mid_output.append(self.sigmoid_continuous(t))

            mid_input = mid_output


        output = mid_output

        # customizing output
        for i, v in enumerate(output):
            if v > 0.5:
                output[i] = 1.0
            else:
                output[i] = 0.0

        # print("output: %s" % str(output))
        # print("")
        return output

class GeneticAlgorithm:
    def __init__(self, popoulation_n):
        # important parameters
        self.mutation_power = 1.0
        self.mutation_prob = 0.3

        self.population_n = popoulation_n
        self.generation = 1
        self.selction_ratio = 0.4

        self.population = []
        for i in range(popoulation_n):
            self.population.append(NeuralNetwork(4,3,3))

        self.selection_info = []
        self.crossover_info = []
        self.mutation_info = []

    def fitness_averaging(self, N):
        for i, v in enumerate(self.population):
            self.population[i].fitness *= (1 / N)

    def get_population_info_surf_rect(self, info_font, prev_TF):
        yy = 0
        if prev_TF == False:
            yy = 400

        info_surf = []
        info_rect = []

        info_surf.append(info_font.render('Generation: %5d' % (self.generation), True, GREEN))
        info_rect.append(info_surf[-1].get_rect())
        info_rect[-1].topleft = (10, 10 + yy)

        for i, v in enumerate(self.population):
            if i < 20:  # 상위 20개만 출력
                printing = [int(x / 0.1) for x in v.genome]
                printing = [x / 10 for x in printing]
                info_surf.append(info_font.render('#%2d: Fit:%3.2f, Genome: %s' % (i+1, v.fitness, str(printing)), True, WHITE))
                info_rect.append(info_surf[-1].get_rect())
                info_rect[-1].topleft = (10, 20 + i*10 + yy)

        return info_surf, info_rect

    def get_evolution_info_surf_rect(self, info_font):
        info_surf = []
        info_rect = []

        info_surf.append(info_font.render('[EVOLUTION INFORMATION]', True, RED))
        info_rect.append(info_surf[-1].get_rect())
        info_rect[-1].topleft = (10, 10 + 230)

        temp = [x[0] for x in self.selection_info]
        info_surf.append(info_font.render('Selction: Top %s is selected' %  (temp), True, WHITE))
        info_rect.append(info_surf[-1].get_rect())
        info_rect[-1].topleft = (10, 10 + 260)

        temp = [x[0] for x in self.crossover_info]
        info_surf.append(info_font.render('Crossover: crossover sites are %s' % (temp), True, WHITE))
        info_rect.append(info_surf[-1].get_rect())
        info_rect[-1].topleft = (10, 10 + 280)

        temp = [x[1] for x in self.mutation_info]
        info_surf.append(info_font.render('Mutation Individual: %s' % (temp), True, WHITE))
        info_rect.append(info_surf[-1].get_rect())
        info_rect[-1].topleft = (10, 10 + 300)

        temp = [x[0] for x in self.mutation_info]
        info_surf.append(info_font.render('Number of Mutation: %s' % (temp), True, WHITE))
        info_rect.append(info_surf[-1].get_rect())
        info_rect[-1].topleft = (10, 10 + 320)

        # temp = [x[3] for x in self.mutation_info]
        # info_surf.append(info_font.render('Mutation Places: %s' % (temp), True, WHITE))
        # info_rect.append(info_surf[-1].get_rect())
        # info_rect[-1].topleft = (10, 10 + 340)

        return info_surf, info_rect

    def print_info(self):
        for i, v in enumerate(self.population):
            print("Generation: %d" % self.generation)
            print("Individual #%d, Survival Time: %.2f" % (i, v.fitness))

    def get_population(self):
        return self.population

    def print_population(self):
        print(self.population)

    def new_generation(self):
        self.selection_info = []
        self.crossover_info = []
        self.mutation_info = []
        # sort population by fitness to create new generation and evolve
        self.fitness_sort()

        result = []
        selection = []
        crossover = []
        mutation = []


        n = int(self.population_n * self.selction_ratio)
        t = []
        for i in range(0, self.population_n - n*2):
            t.append(self.population[n+i].genome)
        for i in range(0, self.population_n - n * 2):
            self.population[n*2 + i].genome = t[i]

        for i in range(0, n):
            self.population[n+i].genome = self.population[i].genome

        selection += self.selection(self.population)
        crossover += self.crossover(self.population[n:n*2])
        mutation += self.mutation(self.population[n*2:])

        # result += self.crossover()
        # rest_n = len(self.population) - len(result)
        # result += self.mutation(self.population)[:rest_n]

        result = selection + crossover + mutation

        genomes = []
        for v in result:
            genomes.append(v.genome)

        del self.population
        del result

        self.population = []
        for i in range(self.population_n):
            self.population.append(NeuralNetwork(4, 3, 3))
            self.population[i].genome = genomes[i]
            self.population[i].genome_to_table()

        self.generation += 1

    def fitness_sort(self):
        fitness_values = {}
        for i, v in enumerate(self.population):
            fitness_values[i] = v.fitness

        order = sorted(fitness_values.items(), key=itemgetter(1), reverse=True)

        sorted_population = []
        for v in order:
            sorted_population.append(self.population[v[0]])

        self.population = sorted_population


    def selection(self, population):
        """
        this method forms mating pool from top p% of the population
        :param population: this has to be sorted by its fitness in advance
        :return:
        """
        n = int(self.population_n * self.selction_ratio)
        # selection information: [# of NN, fitness]
        for i in range(n):
            self.selection_info.append([i+1, population[i].fitness])
        return population[:n]



    def crossover(self, organisms):
        mating_pool = []
        mating_pool += organisms
        n = len(mating_pool)

        for i in range(int(n / 2)):
            parent1 = []
            parent2 = []
            parent1 += mating_pool[i*2].genome
            parent2 += mating_pool[i*2 + 1].genome
            if len(parent1) != len(parent2):
                print("Error: parents have different length: can't make offsprings")
            crossover_pos = random.randrange(0, len(parent1))
            temp = parent1[crossover_pos:]
            parent1[crossover_pos:] = parent2[crossover_pos:]
            parent2[crossover_pos:] = temp

            mating_pool[i*2].genome = parent1
            mating_pool[i*2 + 1].genome = parent2

            # crossover information: [crossover position, fitness]
            self.crossover_info.append([crossover_pos, mating_pool[i * 2].fitness])
            self.crossover_info.append([crossover_pos, mating_pool[i * 2 + 1].fitness])

        if n % 2 != 0:
            self.crossover_info.append([-1, mating_pool[n-1].fitness])

        return mating_pool

    def mutation(self, organisms):
        """
        Do mutation process on each genome on organisms
        :param organisms:
        :return:
        """
        n = int(self.population_n * self.selction_ratio)

        for i in range(len(organisms)):
            # mutation information: [# of mutation, # of NN, fitness, ,pos]
            self.mutation_info.append([0, i + 1, organisms[i].fitness, []])


            for j in range(len(organisms[i].genome)):
                r = random.random()
                if r <= self.mutation_prob:
                    self.mutation_info[-1][0] += 1
                    m = random.gauss(0.0, self.mutation_power)

                    self.mutation_info[-1][3].append(j)
                    organisms[i].genome[j] += m

        return organisms


if __name__ == "__main__":
    # t = NeuralNetwork(4,2,2)
    # t.genome_to_table()
    # print(t.processing([1.0, 1.0, 1.0, 1.0]))

    t = []
    for i in range(10):
        t.append(NeuralNetwork(1,1,2))

    for v in t:
        v.print_genome()

    ga = GeneticAlgorithm(t)
    ga.print_population()
    ga.new_generation()
    ga.print_population()
    ga.new_generation()
    ga.print_population()
