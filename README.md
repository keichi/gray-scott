# gray-scott

This is a 3D 7-point stencil code to simulate the following Gray-Scott
reaction diffusion model:

```
u_t = Du * u_xx - u * v^2 + F * (1 - u)
v_t = Dv * v_xx + v * v^2 - (F + k) * v
```

## How to build

Make sure MPI and ADIOS2 are installed.

```
$ cmake .
$ make
```

## How to change the parameters

Edit settings.json to change the parameters for the simulation.

| Key           | Description                           |
| ------------- | ------------------------------------- |
| L             | Size of global array (L x L x L cube) |
| Du            | Diffusion coefficient of U            |
| Dv            | Diffusion coefficient of V            |
| F             | Feed rate of U                        |
| k             | Kill rate of V                        |
| dt            | Timestep                              |
| steps         | Total number of steps to simulate     |
| plotgap       | Number of steps between output        |
| noise         | Amount of noise to inject             |
| output        | Output file/stream name               |
| adios_config  | ADIOS2 XML file name                  |

Decomposition is automatically determined by MPI_Dims_create.
