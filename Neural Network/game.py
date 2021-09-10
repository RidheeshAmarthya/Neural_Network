import pygame
import sys

pygame.init()
screen = pygame.display.set_mode((576, 1024))
clock = pygame.time.Clock()

bg_main = pygame.image.load('Game_Assets/bg.png').convert() #convert makes it smoother for pygame
bg_main = pygame.transform.scale2x(bg_main)
bg_x = 0
bg_y = 0

floor = pygame.image.load('Game_Assets/base.png').convert()
floor = pygame.transform.scale2x(floor)
floor_x = 0
floor_y = 0

sun = pygame.image.load('Game_Assets/sun.png')
sun = pygame.transform.scale(sun, (90,90))

bird0 = pygame.image.load('Game_Assets/bird1.png')
bird0 = pygame.transform.scale2x(bird0)
bird1 = pygame.image.load('Game_Assets/bird2.png')
bird1 = pygame.transform.scale2x(bird1)
bird2 = pygame.image.load('Game_Assets/bird3.png')
bird2 = pygame.transform.scale2x(bird2)

bird = [bird0, bird1, bird2]
bird_frame = 0

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    screen.blit(bg_main,(bg_x, 0))
    screen.blit(bg_main, (bg_x + 574, 0))

    if bg_x == -574:
        bg_x = 0
    bg_x -= 0.5

    screen.blit(floor, (floor_x, 900))
    screen.blit(floor, (floor_x + 574, 900))

    if floor_x == -574:
        floor_x = 0
    floor_x -= 2

    screen.blit(sun, (400, 40))

    screen.blit(bird[bird_frame], (250, 500)) #not working??
    if bird_frame == 2:
        bird_frame = 0

    pygame.display.update()
    clock.tick(60)

print("You should not be seeing this...")

