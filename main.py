import enum
import math
import random
import uuid
from enum import Enum

import mesa
import numpy as np
from collections import defaultdict

import mesa.space
from mesa import Agent, Model
from mesa.datacollection import DataCollector
from mesa.time import RandomActivation
from mesa.visualization.ModularVisualization import VisualizationElement, ModularServer
from mesa.visualization.modules import ChartModule

MAX_ITERATION = 100
PROBA_CHGT_ANGLE = 0.01


def move(x, y, speed, angle):
    return x + speed * math.cos(angle), y + speed * math.sin(angle)


def go_to(x, y, speed, dest_x, dest_y):
    if np.linalg.norm((x - dest_x, y - dest_y)) < speed:
        return (dest_x, dest_y), 2 * math.pi * random.random()
    else:
        angle = math.acos((dest_x - x)/np.linalg.norm((x - dest_x, y - dest_y)))
        if dest_y < y:
            angle = - angle
        return move(x, y, speed, angle), angle


class MarkerPurpose(Enum):
    DANGER = enum.auto(),
    INDICATION = enum.auto()


class ContinuousCanvas(VisualizationElement):
    local_includes = [
        "./js/simple_continuous_canvas.js",
    ]

    def __init__(self, canvas_height=500,
                 canvas_width=500, instantiate=True):
        VisualizationElement.__init__(self)
        self.canvas_height = canvas_height
        self.canvas_width = canvas_width
        self.identifier = "space-canvas"
        if (instantiate):
            new_element = ("new Simple_Continuous_Module({}, {},'{}')".
                           format(self.canvas_width, self.canvas_height, self.identifier))
            self.js_code = "elements.push(" + new_element + ");"

    def portrayal_method(self, obj):
        return obj.portrayal_method()

    def render(self, model):
        representation = defaultdict(list)
        for obj in model.schedule.agents:
            portrayal = self.portrayal_method(obj)
            if portrayal:
                portrayal["x"] = ((obj.x - model.space.x_min) /
                                  (model.space.x_max - model.space.x_min))
                portrayal["y"] = ((obj.y - model.space.y_min) /
                                  (model.space.y_max - model.space.y_min))
            representation[portrayal["Layer"]].append(portrayal)
        for obj in model.mines:
            portrayal = self.portrayal_method(obj)
            if portrayal:
                portrayal["x"] = ((obj.x - model.space.x_min) /
                                  (model.space.x_max - model.space.x_min))
                portrayal["y"] = ((obj.y - model.space.y_min) /
                                  (model.space.y_max - model.space.y_min))
            representation[portrayal["Layer"]].append(portrayal)
        for obj in model.markers:
            portrayal = self.portrayal_method(obj)
            if portrayal:
                portrayal["x"] = ((obj.x - model.space.x_min) /
                                  (model.space.x_max - model.space.x_min))
                portrayal["y"] = ((obj.y - model.space.y_min) /
                                  (model.space.y_max - model.space.y_min))
            representation[portrayal["Layer"]].append(portrayal)
        for obj in model.obstacles:
            portrayal = self.portrayal_method(obj)
            if portrayal:
                portrayal["x"] = ((obj.x - model.space.x_min) /
                                  (model.space.x_max - model.space.x_min))
                portrayal["y"] = ((obj.y - model.space.y_min) /
                                  (model.space.y_max - model.space.y_min))
            representation[portrayal["Layer"]].append(portrayal)
        for obj in model.quicksands:
            portrayal = self.portrayal_method(obj)
            if portrayal:
                portrayal["x"] = ((obj.x - model.space.x_min) /
                                  (model.space.x_max - model.space.x_min))
                portrayal["y"] = ((obj.y - model.space.y_min) /
                                  (model.space.y_max - model.space.y_min))
            representation[portrayal["Layer"]].append(portrayal)
        return representation


class Obstacle:  # Environnement: obstacle infranchissable
    def __init__(self, x, y, r):
        self.x = x
        self.y = y
        self.r = r

    def portrayal_method(self):
        portrayal = {"Shape": "circle",
                     "Filled": "true",
                     "Layer": 1,
                     "Color": "black",
                     "r": self.r}
        return portrayal


class Quicksand:  # Environnement: ralentissement
    def __init__(self, x, y, r):
        self.x = x
        self.y = y
        self.r = r

    def portrayal_method(self):
        portrayal = {"Shape": "circle",
                     "Filled": "true",
                     "Layer": 1,
                     "Color": "olive",
                     "r": self.r}
        return portrayal


class Mine:  # Environnement: élément à ramasser
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def portrayal_method(self):
        portrayal = {"Shape": "circle",
                     "Filled": "true",
                     "Layer": 2,
                     "Color": "black",
                     "r": 2}
        return portrayal


class Marker:  # La classe pour les balises
    def __init__(self, x, y, purpose, direction=None):
        self.x = x
        self.y = y
        self.purpose = purpose
        if purpose == MarkerPurpose.INDICATION:
            if direction is not None:
                self.direction = direction
            else:
                raise ValueError("Direction should not be none for indication marker")

    def portrayal_method(self):
        portrayal = {"Shape": "circle",
                     "Filled": "true",
                     "Layer": 2,
                     "Color": "red" if self.purpose == MarkerPurpose.DANGER else "green",
                     "r": 2}
        return portrayal


class Robot(Agent):  # La classe des agents
    def __init__(self, unique_id: int, model: Model, x, y, speed, sight_distance, angle=0.0):
        super().__init__(unique_id, model)
        self.x = x
        self.y = y
        self.speed = speed
        self.default_speed = speed
        self.sight_distance = sight_distance
        self.angle = angle
        self.step_to_ignore = 0
        self.counter_sand = 0


    def step(self):
        # pass  # TODO L'intégralité du code du TP peut être ajoutée ici.

        # ignore les marqueurs
        if self.step_to_ignore != 0:
            self.step_to_ignore -= 1

        not_ok = True # ne peut pas se déplacer
        move_again = True


        # vérifie s'il y a du sable mouvant :
        quicksands = [sable for sable in self.model.quicksands if
                        np.sqrt((sable.x-self.x)**2 + (sable.y-self.y)**2) <= sable.r]
        if quicksands and self.speed == self.default_speed :
                self.speed /= 2
                self.counter_sand +=1

        # quand il sort de cette zone
        elif not quicksands and self.speed != self.default_speed : 
                self.counter_sand +=1
                # on met le marqueur DANGER
                self.model.markers.append(Marker(self.x, self.y, MarkerPurpose.DANGER))
                # remet la vitesse
                self.speed = self.default_speed
                # on initialise le compteur pour ignorer les derniers marqueurs DANGER
                self.step_to_ignore = 3

        while not_ok:
            #deplacement
            next_x, next_y = move(self.x, self.y, self.speed, self.angle)

            close_robots = [robot for robot in self.model.schedule.agents if 
                                np.sqrt((robot.x-next_x)**2 + (robot.y-next_y)**2) < robot.speed]

            close_obstacles = [obstacle for obstacle in self.model.obstacles 
                        if np.sqrt((obstacle.x-next_x)**2 + (obstacle.y-next_y)**2) <= obstacle.r ]

            # s'il n'y a pas d'obstacles et d'autres robots, et si y_min<= y <= y_max et x_min <= x <=x_max, le robot se déplace
            if not(close_robots or close_obstacles) and (next_x <= self.model.space.x_min and self.model.space.x_min <= next_x and self.model.space.y_min <= next_y and next_y <= self.model.space.y_max):
                not_ok = False
            else :
                self.angle = 2*math.pi * random.random()

        #peut se déplacer

        # chercher les mines dans son champ de vision
        close_mines = [mine for mine in self.model.mines if
                       (np.sqrt((mine.x-self.x)**2 + (mine.y-self.y)**2) < self.sight_distance)]
        if close_mines:
            #s'il est sur une mine
            found = [mine for mine in close_mines if (self.x, self.y) == (mine.x, mine.y)]
            if found :
                self.model.mines.remove(found[0])
                self.counter += 1
                # ajouter un marqueur INDICATION
                self.model.markers.append(Marker(self.x, self.y, MarkerPurpose.INDICATION, self.angle))
                self.step_to_ignore = 5

            #s'il détecte des mines ( autre que la mine sur laquelle il se situe)
            else:
                index = np.argmin([np.sqrt((mine.x-self.x)**2 + (mine.y-self.y)**2) for mine in close_mines])
                goal_mine = close_mines[index]
                (self.x, self.y), self.angle = go_to( self.x, self.y, self.speed, goal_mine.x, goal_mine.y)
                move_again = False

        else : 
            # pas de mine, deplacement classique
            # vérifie les marqueurs pour adapter sa trajectoire
            danger_markers = [marker for marker in self.model.markers if marker.purpose == MarkerPurpose.DANGER and
                                (np.sqrt((marker.x-self.x)**2 + (marker.y-self.y)**2) < self.sight_distance)]
            indic_markers = [marker for marker in self.model.markers if marker.purpose == MarkerPurpose.INDICATION and
                                (np.sqrt((marker.x-self.x)**2 + (marker.y-self.y)**2) < self.sight_distance)]

            # demi-tour
            self.angle = (self.angle + math.pi*0.9) % (2*math.pi)
            self.x, self.y = move(self.x, self.y, self.speed, self.angle)
            move_again = False

            if indic_markers and self.step_to_ignore <= 0 :
                    marker = indic_markers[0] 
                    # va vers ce marqueur
                    (self.x, self.y), self.angle = go_to( self.x, self.y, self.speed, marker.x, marker.y)
                    # enleve ce marqueur
                    self.model.markers.remove(marker)
                    # tourne
                    self.angle = (marker.direction + math.pi/2) % (2*math.pi)
                    move_again = False 

        if random.random() <= PROBA_CHGT_ANGLE:
            self.angle = 2 * math.pi * random.random()
  

        if move_again :
            next_x, next_y = move(self.x, self.y, self.speed, self.angle) 
            self.x, self.y = next_x, next_y                  

    def portrayal_method(self):
        portrayal = {"Shape": "arrowHead", "s": 1, "Filled": "true", "Color": "Red", "Layer": 3, 'x': self.x,
                     'y': self.y, "angle": self.angle}
        return portrayal


class MinedZone(Model):
    collector = DataCollector(
        model_reporters={"Mines": lambda model: len(model.mines),
                         "Danger markers": lambda model: len([m for m in model.markers if
                                                              m.purpose == MarkerPurpose.DANGER]),
                         "Indication markers": lambda model: len([m for m in model.markers if
                                                                  m.purpose == MarkerPurpose.INDICATION]),
                         "Destructed mines": lambda model: sum([robot.counter for robot in model.schedule.agents]),
                         "Steps in quicksand": lambda model: sum([robot.counter_sand for robot in model.schedule.agents])
                         },
        agent_reporters={})

    def __init__(self, n_robots, n_obstacles, n_quicksand, n_mines, speed):
        Model.__init__(self)
        self.space = mesa.space.ContinuousSpace(600, 600, False)
        self.schedule = RandomActivation(self)
        self.mines = []
        self.markers = []
        self.obstacles = []
        self.quicksands = []
        for _ in range(n_obstacles):
            self.obstacles.append(Obstacle(
                random.random() * 500, random.random() * 500, 10 + 20 * random.random()))
        for _ in range(n_quicksand):
            self.quicksands.append(Quicksand(
                random.random() * 500, random.random() * 500, 10 + 20 * random.random()))
        for _ in range(n_robots):
            x, y = random.random() * 500, random.random() * 500
            while [o for o in self.obstacles if np.linalg.norm((o.x - x, o.y - y)) < o.r] or \
                    [o for o in self.quicksands if np.linalg.norm((o.x - x, o.y - y)) < o.r]:
                x, y = random.random() * 500, random.random() * 500
            self.schedule.add(
                Robot(int(uuid.uuid1()), self, x, y, speed,
                      2 * speed, random.random() * 2 * math.pi))
        for _ in range(n_mines):
            x, y = random.random() * 500, random.random() * 500
            while [o for o in self.obstacles if np.linalg.norm((o.x - x, o.y - y)) < o.r] or \
                    [o for o in self.quicksands if np.linalg.norm((o.x - x, o.y - y)) < o.r]:
                x, y = random.random() * 500, random.random() * 500
            self.mines.append(Mine(x, y))
        self.datacollector = self.collector

    def step(self):
        self.datacollector.collect(self)
        self.schedule.step()
        if not self.mines:
            self.running = False


def run_single_server():
    chart = ChartModule([{"Label": "Mines",
                          "Color": "Orange"},
                         {"Label": "Danger markers",
                          "Color": "Red"},
                         {"Label": "Indication markers",
                          "Color": "Green"},
                         {"Label": "Destructed mines",
                          "Color": "Blue"},
                         { "Label": "Steps in quicksand",
                          "Color": "Magenta"}
                         ],
                        data_collector_name='datacollector')
    server = ModularServer(MinedZone,
                           [ContinuousCanvas(),
                            chart],
                           "Deminer robots",
                           {"n_robots": mesa.visualization.
                            ModularVisualization.UserSettableParameter('slider', "Number of robots", 7, 3,
                                                                       15, 1),
                            "n_obstacles": mesa.visualization.
                            ModularVisualization.UserSettableParameter(
                                'slider', "Number of obstacles", 5, 2, 10, 1),
                            "n_quicksand": mesa.visualization.
                            ModularVisualization.UserSettableParameter(
                                'slider', "Number of quicksand", 5, 2, 10, 1),
                            "speed": mesa.visualization.
                            ModularVisualization.UserSettableParameter(
                                'slider', "Robot speed", 15, 5, 40, 5),
                            "n_mines": mesa.visualization.
                            ModularVisualization.UserSettableParameter('slider', "Number of mines", 15, 5, 30, 1)})
    server.port = 8521
    server.launch()


if __name__ == "__main__":
    run_single_server()
