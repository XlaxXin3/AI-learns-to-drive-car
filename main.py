import pygame
import sys
import neat
from car import Car

screen_width, screen_height = 1500, 800
generation = 0


def run_car(genomes, config):
    nets = []
    cars = []

    for id, g in genomes:
        net = neat.nn.FeedForwardNetwork.create(g, config)
        nets.append(net)
        g.fitness = 0

        cars.append(Car())

    pygame.init()
    screen = pygame.display.set_mode((screen_width, screen_height))
    clock = pygame.time.Clock()
    map = pygame.image.load('Assets\\map.png')

    global generation
    generation += 1
    while 1:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit(0)

        for index, car in enumerate(cars):
            output = nets[index].activate(car.get_data())
            i = output.index(max(output))

            if i == 0:
                car.angle += 10
            else:
                car.angle -= 10

        remain_cars = 0

        for i, car in enumerate(cars):
            if car.get_alive():
                remain_cars += 1
                car.update(map)
                genomes[i][1].fitness += car.get_reward()

        if remain_cars == 0:
            break

        screen.blit(map, (0, 0))
        draw_cars(cars, screen)

        blit_text(screen, generation, remain_cars)
        pygame.display.flip()
        clock.tick(60)


def draw_cars(cars, screen):
    for car in cars:
        if car.get_alive():
            car.draw(screen)


def blit_text(screen, generation, remain_cars):
    custom_font = "Assets\\PixeloidSans.ttf"
    generation_font = pygame.font.Font(custom_font, 48)
    font = pygame.font.Font(custom_font, 18)
    text = generation_font.render("Generation: " + str(generation), True, (0, 0, 0))
    screen.blit(text, (30,45))

    text = font.render("Remaining cars: " + str(remain_cars), True, (0, 0, 0))
    screen.blit(text, (40,105))


if __name__ == '__main__':
    config_path = 'config-feedforward.txt'
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet,
                                neat.DefaultStagnation, config_path)

    p = neat.Population(config)

    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    p.run(run_car, 1000)
