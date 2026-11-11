# Hello! Welcome to one of my University projects: Simulating a potential exoplanet in the HD 98800 quadruple star system.

## Introduction

This project aimed to simulate a Jupiter-mass exoplanet to investigate whether a stable orbit is feasible around HD 98800 B and whether the planet could be responsible for the formation of the surrounding debris disk.

The system was simulated using the 3rd-order Runge-Kutta numerical integration method, with varying orbital parameters to investigate the conditions under which a stable orbital configuration could exist.

One of the main challenges of this project is the binary nature of HD 98800 B. Since the system consists of two stars and a planet, its orbital dynamics form a three-body problem. Unlike simpler two-body systems, the general three-body problem has no known analytical solution, meaning that the system's behaviour must instead be investigated using numerical methods.

A further challenge arises from the sensitivity of numerical simulations to their initial conditions. Runge-Kutta methods require predefined initial conditions, such as the positions and velocities of the bodies, from which their subsequent motion is calculated. Even small variations in these initial conditions can lead to significantly different orbital outcomes over long timescales, potentially resulting in substantial differences after several hundred years.


To model such a system I made the following approximations:

To set the initial conditions of HD 98800 the following approximations are made to simplify calculations:
1. All bodies in the system have zero angular momentum
2. All bodies have equal density over their volume and can be assumed as point masses.
3. A and B, Aa and Ab, Ba and Bb all start at apastron, and the planet starts vertically above the centre of mass in B. By making this approximation, it becomes trivial to compute the initial conditions of all systems.
4. There is only one exoplanet in HD 98800 B, and this planet has no moons and has the same mass and radius as Jupiter. The planet must have enough mass to clear debris in its orbital path to create gaps in the ring approximately **3** AU wide, but must remain small enough to be undetectable.
5. The inclination of the ring is exactly **90°** with respect to the plane of orbit in B, and the planes of orbit of A and B are parallel.

Results can be found in the document in the repo (not uploaded yet).

## Code

The code runs with Python versions 3.13-3.15 (others have not been tested) and works with the most up to date version of its libraries as of 11/11/26

AI was not used at all in the making of this program.

The code was written 2~ years ago (and my skills have vastly improved since then, so please brace yourself!) 
