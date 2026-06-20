import taichi as ti
import pygame
import numpy as np
import math
import random

ti.init(arch=ti.cpu)
pygame.init()
WIDTH, HEIGHT = 700, 700
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()
pygame.display.set_caption("Bouncing Ball Simulation")

# variables
center = (WIDTH / 2.0, HEIGHT / 2.0)
boundary_radius = WIDTH / 2.0


@ti.data_oriented
class Simulation:
    def __init__(self):
        self.max_bodies = 50000
        self.positions = ti.Vector.field(2, dtype=ti.f32, shape=self.max_bodies)
        self.velocities = ti.Vector.field(2, dtype=ti.f32, shape=self.max_bodies)
        self.radii = ti.field(dtype=ti.f32, shape=self.max_bodies)
        self.masses = ti.field(dtype=ti.f32, shape=self.max_bodies)
        self.colors = ti.Vector.field(3, dtype=ti.f32, shape=self.max_bodies)
        self.active_bodies = ti.field(dtype=ti.i32, shape=())

    @ti.kernel
    def boundary_collision(self):
        for i in range(self.active_bodies[None]):
            pos = self.positions[i]
            vel = self.velocities[i]
            radius = self.radii[i]

            ti_center = ti.Vector([WIDTH / 2.0, HEIGHT / 2.0])
            to_center = pos - ti_center
            dist = to_center.norm()
            if dist + radius > boundary_radius:
                normal = to_center / dist
                vel -= 2 * vel.dot(normal) * normal
                self.velocities[i] = vel
                self.positions[i] = ti_center + normal * (boundary_radius - radius)

    @ti.kernel
    def particle_collision(self):
        for i in range(self.active_bodies[None]):
            for j in range(i + 1, self.active_bodies[None]):
                pos_i = self.positions[i]
                pos_j = self.positions[j]
                radius_i = self.radii[i]
                radius_j = self.radii[j]
                mass_i = self.masses[i]
                mass_j = self.masses[j]

                distance = pos_i - pos_j
                dist = distance.norm()
                if 0 < dist < radius_i + radius_j:
                    normal = distance / dist
                    vel_i = self.velocities[i]
                    vel_j = self.velocities[j]

                    relative_vel = vel_i - vel_j
                    vel_along_normal = relative_vel.dot(normal)

                    if vel_along_normal <= 0:
                        restitution = 0.4
                        impulse_scalar = -(1 + restitution) * vel_along_normal
                        impulse_scalar /= (1 / self.masses[i]) + (1 / self.masses[j])

                        impulse = impulse_scalar * normal
                        self.velocities[i] += impulse / mass_i
                        self.velocities[j] -= impulse / mass_j

                        penetration = radius_i + radius_j - dist
                        self.positions[i] += normal * (
                            penetration * (mass_j / (mass_i + mass_j))
                        )
                        self.positions[j] -= normal * (
                            penetration * (mass_i / (mass_i + mass_j))
                        )

    def add_ball(self, pos, vel, mass, color):
        idx = self.active_bodies[None]
        if idx < self.max_bodies:
            self.positions[idx] = pos
            self.velocities[idx] = vel
            self.radii[idx] = math.sqrt(mass / math.pi)
            self.masses[idx] = mass
            self.colors[idx] = color
            self.active_bodies[None] += 1

    def clear(self):
        self.active_bodies[None] = 0
        self.positions.fill(0)
        self.velocities.fill(0)
        self.radii.fill(0)
        self.masses.fill(0)
        self.colors.fill(0)

    @ti.kernel
    def update_positions(self, dt: ti.f32):
        for i in range(self.active_bodies[None]):
            acceleration = ti.Vector([0.0, 200])
            self.velocities[i] += acceleration * dt
            self.positions[i] += self.velocities[i] * dt

    def physics_step(self, dt):
        self.update_positions(dt)
        for _ in range(1):  # Multiple iterations for better collision resolution
            self.boundary_collision()
            self.particle_collision()


sim = Simulation()


class Input:
    def __init__(self):
        self.mouse_down = False
        self.spawning_mass = 10

    def handle_events(self):
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP] and self.spawning_mass < 10000:
            self.spawning_mass += 20
        if keys[pygame.K_DOWN] and self.spawning_mass > 10:
            self.spawning_mass -= 20

        for event in pygame.event.get():
            if event.type == pygame.QUIT or (
                event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
            ):
                pygame.quit()
                exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                self.mouse_down = True
            elif event.type == pygame.MOUSEBUTTONUP:
                self.mouse_down = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_0:
                sim.clear()
                self.spawning_mass = 10

        if self.mouse_down:
            current_mouse_pos = np.array(pygame.mouse.get_pos(), dtype=np.float32)
            sim.add_ball(
                current_mouse_pos,
                (0, 0),
                self.spawning_mass,
                (
                    random.randint(10, 255),
                    random.randint(10, 255),
                    random.randint(10, 255),
                ),
            )


input_handler = Input()


class Renderer:
    def __init__(self):
        pass

    def render_boundary(self):
        pygame.draw.circle(
            screen,
            (255, 255, 255),
            (int(center[0]), int(center[1])),
            int(boundary_radius),
            0,
        )

    def render_balls(self, sim):
        for i in range(sim.active_bodies[None]):
            pos = sim.positions[i]
            radius = sim.radii[i]
            color = sim.colors[i]
            pygame.draw.circle(
                screen,
                (int(color[0]), int(color[1]), int(color[2])),
                (int(pos[0]), int(pos[1])),
                max(1, int(radius)),
            )

    def render(self):
        screen.fill((0, 0, 0))
        self.render_boundary()
        self.render_balls(sim)


renderer = Renderer()

if __name__ == "__main__":
    while True:
        dt = 1/120
        clock.tick(120)
        input_handler.handle_events()
        sim.physics_step(dt)
        renderer.render()
        pygame.display.flip()
        fps = clock.get_fps()
        pygame.display.set_caption(
            f"Bouncing Ball Simulation - Mass: {input_handler.spawning_mass} - Balls: {sim.active_bodies[None]} - FPS: {fps:.2f}"
        )
