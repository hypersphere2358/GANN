import pygame
import sys
import random
import numpy
import math
from neat_fixed import *

from pygame.locals import *

WORLD_X = 1200
WORLD_Y = 700

pygame.init()
DISPLAYSURF = pygame.display.set_mode((WORLD_X, WORLD_Y))
pygame.display.set_caption(' # 9 ')

FPS = 100
fpsClock = pygame.time.Clock()


# set up the colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
BORN = (255, 0, 0)

# set font
BASICFONT = pygame.font.Font('freesansbold.ttf', 11)
INFO_FONT = pygame.font.Font('freesansbold.ttf', 9)


class Creature:



    drawing_sensor_TF = True
    drawing_output_light_TF = True

    def __init__(self):
        self.pos_x = random.randrange(0, WORLD_X)
        self.pos_y = random.randrange(0, WORLD_Y)
        self.health_color = (255, 0, 0)
        self.size = 15
        self.survival_time = 0.0
        self.initial_life = 255.0
        self.health = self.initial_life
        self.health_scale = 5
        self.health_status = self.get_health_status()
        self.speed = 3.0
        self.moving_step = 2.5
        self.direction = random.random() * 360.0   # direction at which creature is heading (0 ~ 360)
        self.sensor_degree = 60.0
        self.sensor_length = 100.0
        self.sensor_size = 75.0
        self.sensor_radius = 20.0
        self.decay_rate = 0.001
        self.turn_degree = 50.0
        self.calculate_sensor_pos()
        self.alive = 1.0

        self.output_light = [0.0, 0.0, 0.0]

    def get_health_status(self):
        div = self.initial_life * 2 / self.health_scale
        status = int(self.health / div)
        if status >= (self.health_scale - 1):
            status = self.health_scale - 1
        return status

    def calculate_sensor_pos(self):
        # compute two end points for each sensor (1, 2, 3)
        # total 12 values, but 6 points are the same
        x = int(self.pos_x)
        y = int(self.pos_y)

        # self.sensor1_p1_x = self.sensor2_p1_x = self.sensor3_p1_x = x
        # self.sensor1_p1_y = self.sensor2_p1_y = self.sensor3_p1_y = y

        self.sensor1_p1_x = x + (self.sensor_length - self.sensor_size) * math.cos((self.direction + self.sensor_degree) * math.pi / 180.0)
        self.sensor1_p1_y = y - (self.sensor_length - self.sensor_size) * math.sin((self.direction + self.sensor_degree) * math.pi / 180.0)

        self.sensor2_p1_x = x + (self.sensor_length - self.sensor_size) * math.cos(self.direction * math.pi / 180.0)
        self.sensor2_p1_y = y - (self.sensor_length - self.sensor_size) * math.sin(self.direction * math.pi / 180.0)

        self.sensor3_p1_x = x + (self.sensor_length - self.sensor_size) * math.cos((self.direction - self.sensor_degree) * math.pi / 180.0)
        self.sensor3_p1_y = y - (self.sensor_length - self.sensor_size) * math.sin((self.direction - self.sensor_degree) * math.pi / 180.0)


        self.sensor1_p2_x = x + self.sensor_length * math.cos((self.direction + self.sensor_degree) * math.pi / 180.0)
        self.sensor1_p2_y = y - self.sensor_length * math.sin((self.direction + self.sensor_degree) * math.pi / 180.0)

        self.sensor2_p2_x = x + self.sensor_length * math.cos(self.direction * math.pi / 180.0)
        self.sensor2_p2_y = y - self.sensor_length * math.sin(self.direction * math.pi / 180.0)

        self.sensor3_p2_x = x + self.sensor_length * math.cos((self.direction - self.sensor_degree) * math.pi / 180.0)
        self.sensor3_p2_y = y - self.sensor_length * math.sin((self.direction - self.sensor_degree) * math.pi / 180.0)

    def move(self):
        # move first
        self.pos_x += self.moving_step * math.cos(self.direction * math.pi / 180.0)
        self.pos_y -= self.moving_step * math.sin(self.direction * math.pi / 180.0)

        # check if the creature is outside the area
        # creature doesn't get damage by reason of being outside the area.
        if int(self.pos_x) >= WORLD_X:
            self.pos_x = 0.0
        elif int(self.pos_x) < 0:
                self.pos_x = float(WORLD_X - 1)
        elif int(self.pos_y) >= WORLD_Y:
            self.pos_y = 0.0
        elif int(self.pos_y) < 0:
            self.pos_y = float(WORLD_Y - 1)

    def draw(self, sensor_finding_TF, drawing_sensor_TF, drawing_output_light_TF, drawing_health_TF):
        x = int(self.pos_x)
        y = int(self.pos_y)

        # creature's body
        self.health_color = (255, min(255, int(abs(self.initial_life - self.health))), min(255, int(abs(self.initial_life - self.health))))
        pygame.draw.circle(DISPLAYSURF, self.health_color, (x, y), self.size, 0)    # 몸체
        # 눈
        pygame.draw.circle(DISPLAYSURF, YELLOW, (
            int(self.pos_x + (self.size - 5) * math.cos((self.direction + 17.0) * math.pi / 180.0)),
            int(self.pos_y - (self.size - 5) * math.sin((self.direction + 17.0) * math.pi / 180.0))), 3, 0)
        pygame.draw.circle(DISPLAYSURF, YELLOW, (
            int(self.pos_x + (self.size - 5) * math.cos((self.direction - 17.0) * math.pi / 180.0)),
            int(self.pos_y - (self.size - 5) * math.sin((self.direction - 17.0) * math.pi / 180.0))), 3, 0)

        # default sensor color
        self.sensor1_line_color = WHITE
        self.sensor2_line_color = WHITE
        self.sensor3_line_color = WHITE

        if drawing_sensor_TF == True:
            # in case that sensor detected food
            if sensor_finding_TF[0] == True:
                self.sensor1_line_color = BLUE
            if sensor_finding_TF[1] == True:
                self.sensor2_line_color = BLUE
            if sensor_finding_TF[2] == True:
                self.sensor3_line_color = BLUE

            # 세 개의 센서
            pygame.draw.line(DISPLAYSURF, self.sensor1_line_color, (self.sensor1_p1_x, self.sensor1_p1_y), (self.sensor1_p2_x, self.sensor1_p2_y))
            pygame.draw.line(DISPLAYSURF, self.sensor2_line_color, (self.sensor2_p1_x, self.sensor2_p1_y), (self.sensor2_p2_x, self.sensor2_p2_y))
            pygame.draw.line(DISPLAYSURF, self.sensor3_line_color, (self.sensor3_p1_x, self.sensor3_p1_y), (self.sensor3_p2_x, self.sensor3_p2_y))

        # 다음 행동을 나타내는 지시등
        if drawing_output_light_TF == True:
            if self.output_light[0] == 1.0:
                pygame.draw.circle(DISPLAYSURF, BLUE, (
                int(self.pos_x + 25 * math.cos((self.direction + 90.0)* math.pi / 180.0)),
                int(self.pos_y - 25 * math.sin((self.direction + 90.0)* math.pi / 180.0))), 3, 0)
            else:
                pygame.draw.circle(DISPLAYSURF, WHITE, (
                int(self.pos_x + 25 * math.cos((self.direction + 90.0) * math.pi / 180.0)),
                int(self.pos_y - 25 * math.sin((self.direction + 90.0) * math.pi / 180.0))), 3, 0)

            if self.output_light[1] == 1.0:
                pygame.draw.circle(DISPLAYSURF, BLUE, (
                int(self.pos_x + 25 * math.cos((self.direction - 90.0) * math.pi / 180.0)),
                int(self.pos_y - 25 * math.sin((self.direction - 90.0) * math.pi / 180.0))), 3, 0)
            else:
                pygame.draw.circle(DISPLAYSURF, WHITE, (
                int(self.pos_x + 25 * math.cos((self.direction - 90.0) * math.pi / 180.0)),
                int(self.pos_y - 25 * math.sin((self.direction - 90.0) * math.pi / 180.0))), 3, 0)

            if self.output_light[2] == 1.0:
                pygame.draw.circle(DISPLAYSURF, BLUE, (
                int(self.pos_x + 25 * math.cos((self.direction + 180.0) * math.pi / 180.0)),
                int(self.pos_y - 25 * math.sin((self.direction + 180.0) * math.pi / 180.0))), 3, 0)
            else:
                pygame.draw.circle(DISPLAYSURF, WHITE, (
                int(self.pos_x + 25 * math.cos((self.direction + 180.0) * math.pi / 180.0)),
                int(self.pos_y - 25 * math.sin((self.direction + 180.0) * math.pi / 180.0))), 3, 0)

        if drawing_health_TF == True:
            self.draw_health()


    def starving(self, multiple):
        if self.health > 0:
            self.health -= self.decay_rate*300*multiple

    def feeding(self):
        if self.health < self.initial_life * 2:
            self.health += 100

    def death_check(self):
        if self.health <= 0.0 or self.health >= self.initial_life * 2:
            self.alive = 0.0
            return True

    def draw_health(self):
        self.scoreSurf = BASICFONT.render('%3.2f' % (self.health), True, WHITE)
        self.scoreRect = self.scoreSurf.get_rect()
        self.scoreRect.topleft = (self.pos_x - 13, self.pos_y + 14)
        DISPLAYSURF.blit(self.scoreSurf, self.scoreRect)

    def detecting(self, all_food):
        finding = [0.0, 0.0, 0.0]
        for f_id, f in enumerate(all_food):
            food_p = [f.pos_x, f.pos_y]

            # eat food
            dist = dist_two_points(food_p, [self.pos_x, self.pos_y])
            if dist <= self.size + f.size:
                del all_food[f_id]
                self.feeding()
                continue

            f.draw()

            # calculate distances between food and each sensor
            dist1 = dist_point_linesegment(food_p, [self.sensor1_p1_x, self.sensor1_p1_y], [self.sensor1_p2_x, self.sensor1_p2_y])
            dist2 = dist_point_linesegment(food_p, [self.sensor2_p1_x, self.sensor2_p1_y], [self.sensor2_p2_x, self.sensor2_p2_y])
            dist3 = dist_point_linesegment(food_p, [self.sensor3_p1_x, self.sensor3_p1_y], [self.sensor3_p2_x, self.sensor3_p2_y])

            # set value if sensor detects food
            if dist1 <= self.sensor_radius:
                finding[0] = 1.0
            if dist2 <= self.sensor_radius:
                finding[1] = 1.0
            if dist3 <= self.sensor_radius:
                finding[2] = 1.0

        self.finding = finding


    def update(self, left, move_forward, right):
        self.output_light[0] = left
        self.output_light[1] = right
        self.output_light[2] = move_forward

        if left == 1.0:
            self.direction = (self.direction + self.turn_degree) % 360.0
            self.starving(0.5)
        if right == 1.0:
            self.direction = (self.direction - self.turn_degree) % 360.0
            self.starving(0.5)

        if move_forward == 1.0:
            self.move()
            self.starving(0.5)

        self.calculate_sensor_pos()

        self.starving(1.0)
        self.health_status = self.get_health_status()
        self.survival_time += 0.1





class Food:
    def __init__(self):
        self.margin = 50.0
        self.pos_x = random.randrange(0 + self.margin, WORLD_X - self.margin)
        self.pos_y = random.randrange(0 + self.margin, WORLD_Y - self.margin)
        self.size = 5

    def draw(self):
        x = int(self.pos_x)
        y = int(self.pos_y)
        pygame.draw.circle(DISPLAYSURF, GREEN, (x, y), self.size, 0)

def dist_point_linesegment(p, v, w):
    return math.sqrt(squared_dist_point_linesegment(p, v, w))

def dist_two_points(v, w):
    return math.sqrt(squared_dist_two_points(v, w))

def squared_dist_point_linesegment(p, v, w):
    l = squared_dist_two_points(v, w)
    if l == 0.0:
        return squared_dist_two_points(p, v)

    t = ((p[0] - v[0]) * (w[0] - v[0]) + (p[1] - v[1]) * (w[1] - v[1])) / l
    t = max(0, min(1, t))

    proj = [0.0, 0.0]
    proj[0] = v[0] + t * (w[0] - v[0])
    proj[1] = v[1] + t * (w[1] - v[1])

    return squared_dist_two_points(p, proj)

def squared_dist_two_points(v, w):
    return (v[0] - w[0]) * (v[0] - w[0]) + (v[1] - w[1]) * (v[1] - w[1])

def fitness_graph(data):
    topleft = [400,250]
    width = 1200
    height = 150
    step = 1

    # keep max survival time
    max_survival = max(data[0])

    # axis
    pygame.draw.line(DISPLAYSURF, WHITE, (topleft[0], topleft[1]), (topleft[0], topleft[1] + height))
    pygame.draw.line(DISPLAYSURF, WHITE, (topleft[0], topleft[1] + height), (topleft[0] + width, topleft[1] + height))

    for i in range(len(data[1])):
        if i == 0:
            pygame.draw.line(DISPLAYSURF, BLUE, (topleft[0] + i * step, topleft[1] + height), (topleft[0] + (i + 1) * step, topleft[1] + height - int(height * data[1][i] / max_survival)))  # min data
            pygame.draw.line(DISPLAYSURF, RED,  (topleft[0] + i * step, topleft[1] + height), (topleft[0] + (i + 1) * step, topleft[1] + height - int(height * data[0][i] / max_survival)))  # max data
            pygame.draw.line(DISPLAYSURF, GREEN, (topleft[0] + i * step, topleft[1] + height), (topleft[0] + (i + 1) * step, topleft[1] + height - int(height * data[2][i] / max_survival)))  # avg data

        else:
            pygame.draw.line(DISPLAYSURF, BLUE, (topleft[0] + i * step, topleft[1] + height - int(height * data[1][i-1] / max_survival)), (topleft[0] + (i + 1) * step, topleft[1] + height - int(height * data[1][i] / max_survival)))  # min data
            pygame.draw.line(DISPLAYSURF, RED,  (topleft[0] + i * step, topleft[1] + height - int(height * data[0][i-1] / max_survival)), (topleft[0] + (i + 1) * step, topleft[1] + height - int(height * data[0][i] / max_survival)))  # max data
            pygame.draw.line(DISPLAYSURF, GREEN,  (topleft[0] + i * step, topleft[1] + height - int(height * data[2][i - 1] / max_survival)), (topleft[0] + (i + 1) * step, topleft[1] + height - int(height * data[2][i] / max_survival)))  # avg data


    info_surf = []
    info_rect = []



def main():
    all_creatures = []
    all_food = []
    tick_time = 0
    N = 40
    alive_N = N
    food_N = 33
    generation_fitness = [0.0] * N

    info_display_switch = True
    info_display_switch2 = False
    sensor_display_switch = True
    action_display_switch = True
    health_display_switch = True
    averaging_number = 1
    averaging_cnt = 0

    my_GA = GeneticAlgorithm(N)

    max_data = []
    min_data = []
    avg_data = []

    # initial creature, food setting
    for i in range(N):
        all_creatures.append(Creature())

    for i in range(0,food_N):
        all_food.append(Food())

    while True: # main game loop
        DISPLAYSURF.fill(BLACK)
        if info_display_switch2 == True:
            for i, v in enumerate(prev_info_surf):
                DISPLAYSURF.blit(prev_info_surf[i],prev_info_rect[i])
            for i, v in enumerate(curr_info_surf):
                DISPLAYSURF.blit(curr_info_surf[i], curr_info_rect[i])
            for i, v in enumerate(evol_info_surf):
                DISPLAYSURF.blit(evol_info_surf[i], evol_info_rect[i])
            fitness_graph([max_data, min_data, avg_data])


        else:
            # make new food consistently
            if len(all_food) < food_N:
                all_food.append(Food())

            if alive_N == 0:
                # 평균 안하는 방법
                # if info_display_switch == True:
                #     info_display_time = 1.0
                # alive_N = N
                # for i, v in enumerate(my_GA.population):
                #     generation_fitness[i] = v.fitness
                #
                # max_data.append(max(generation_fitness))
                # min_data.append(min(generation_fitness))
                #
                # my_GA.fitness_sort()
                # prev_info_surf, prev_info_rect = my_GA.get_population_info_surf_rect(INFO_FONT, True)
                # my_GA.new_generation()
                # curr_info_surf, curr_info_rect = my_GA.get_population_info_surf_rect(INFO_FONT, False)
                # evol_info_surf, evol_info_rect = my_GA.get_evolution_info_surf_rect(INFO_FONT)

                averaging_cnt += 1
                if averaging_cnt == averaging_number:
                    averaging_cnt = 0
                    my_GA.fitness_averaging(averaging_number)
                    for i, v in enumerate(my_GA.population):
                        generation_fitness[i] = v.fitness

                    # temp는 평균 안낼때만 임시로 사용. 평균낼때는 의미없는 값
                    # temp = []
                    # for i, v in enumerate(all_creatures):
                    #     temp.append(v.survival_time)
                    max_data.append(max(generation_fitness))
                    min_data.append(min(generation_fitness))
                    avg_data.append(sum(generation_fitness) / len(generation_fitness))

                    my_GA.fitness_sort()
                    prev_info_surf, prev_info_rect = my_GA.get_population_info_surf_rect(INFO_FONT, True)
                    my_GA.new_generation()
                    curr_info_surf, curr_info_rect = my_GA.get_population_info_surf_rect(INFO_FONT, False)
                    evol_info_surf, evol_info_rect = my_GA.get_evolution_info_surf_rect(INFO_FONT)

                    if info_display_switch == True:
                        info_display_switch2 = True

                alive_N = N

                del all_creatures
                all_creatures = []
                for i in range(N):
                    all_creatures.append(Creature())

                continue

            # move and draw all creatures who is alive
            for c_id, c in enumerate(all_creatures):
                if c.alive == 0.0:
                    continue

                if c.death_check() == True:
                    alive_N -= 1

                    my_GA.population[c_id].fitness += c.survival_time
                    continue

                # detect first
                c.detecting(all_food)

                # draw creature
                c.draw(c.finding, sensor_display_switch, action_display_switch, health_display_switch)

                # update creature by Neural Network
                re = my_GA.population[c_id].processing(c.finding + [(0.0 if c.health >= c.initial_life else 1.0)])
                c.update(re[0], re[1], re[2])



            info_surf = []
            info_rect = []

            #======================================================================================================================================================
            # evolution status information display
            # ======================================================================================================================================================

            info_surf.append(BASICFONT.render('Generation: %5d   Survival Time of Prev (max: %.2f min: %.2f)' % (my_GA.generation, max(generation_fitness), min(generation_fitness)), True, WHITE))
            info_rect.append(info_surf[-1].get_rect())
            info_rect[-1].topleft = (5, 5)

            info_surf.append(BASICFONT.render('Individuals: %3d / %3d' % (alive_N, N), True, WHITE))
            info_rect.append(info_surf[-1].get_rect())
            info_rect[-1].topleft = (5, 17)

            info_surf.append(BASICFONT.render('Information(Z): ', True, WHITE))
            info_rect.append(info_surf[-1].get_rect())
            info_rect[-1].topleft = (5, 29)

            info_surf.append(BASICFONT.render('Sensor(X): ', True, WHITE))
            info_rect.append(info_surf[-1].get_rect())
            info_rect[-1].topleft = (5, 41)

            info_surf.append(BASICFONT.render('Action(C): ', True, WHITE))
            info_rect.append(info_surf[-1].get_rect())
            info_rect[-1].topleft = (5, 53)

            info_surf.append(BASICFONT.render('Health(V): ', True, WHITE))
            info_rect.append(info_surf[-1].get_rect())
            info_rect[-1].topleft = (5, 65)


            info_surf.append(BASICFONT.render('ON', True, RED))
            if info_display_switch == False:
                info_surf[-1] = BASICFONT.render('OFF', True, BLUE)
            info_rect.append(info_surf[-1].get_rect())
            info_rect[-1].topleft = (85, 29)

            info_surf.append(BASICFONT.render('ON', True, RED))
            if sensor_display_switch == False:
                info_surf[-1] = BASICFONT.render('OFF', True, BLUE)
            info_rect.append(info_surf[-1].get_rect())
            info_rect[-1].topleft = (85, 41)

            info_surf.append(BASICFONT.render('ON', True, RED))
            if action_display_switch == False:
                info_surf[-1] = BASICFONT.render('OFF', True, BLUE)
            info_rect.append(info_surf[-1].get_rect())
            info_rect[-1].topleft = (85, 53)

            info_surf.append(BASICFONT.render('ON', True, RED))
            if health_display_switch == False:
                info_surf[-1] = BASICFONT.render('OFF', True, BLUE)
            info_rect.append(info_surf[-1].get_rect())
            info_rect[-1].topleft = (85, 65)

            for i, v in enumerate(info_surf):
                DISPLAYSURF.blit(info_surf[i], info_rect[i])


        for event in pygame.event.get():
            if event.type == QUIT:

                # delete all objects
                del all_creatures
                del all_food

                pygame.quit()
                sys.exit()
            elif event.type == MOUSEBUTTONDOWN:
                info_display_switch2 = False
                # all_creatures.append(Creature())
                pass

            elif event.type == KEYUP:
                print(event.key)
                if event.key == K_z:
                    if info_display_switch == True:
                        info_display_switch = False
                    else:
                        info_display_switch = True
                elif event.key == K_x:
                    if sensor_display_switch == True:
                        sensor_display_switch = False
                    else:
                        sensor_display_switch = True
                elif event.key == K_c:
                    if action_display_switch == True:
                        action_display_switch = False
                    else:
                        action_display_switch = True
                elif event.key == K_v:
                    if health_display_switch == True:
                        health_display_switch = False
                    else:
                        health_display_switch = True


        pygame.display.update()
        fpsClock.tick(FPS)



if __name__ == '__main__':
    main()
