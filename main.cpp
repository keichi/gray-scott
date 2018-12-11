#include <cstdio>
#include <fstream>
#include <iostream>
#include <mpi.h>
#include <vector>

#include <adios2.h>

#include "gray_scott.h"

void print_settings(const Settings &s)
{
    std::cout << "====================" << std::endl;
    std::cout << "L:            " << s.L << std::endl;
    std::cout << "steps:        " << s.steps << std::endl;
    std::cout << "iterations:   " << s.iterations << std::endl;
    std::cout << "F:            " << s.F << std::endl;
    std::cout << "k:            " << s.k << std::endl;
    std::cout << "dt:           " << s.dt << std::endl;
    std::cout << "Du:           " << s.Du << std::endl;
    std::cout << "Dv:           " << s.Dv << std::endl;
    std::cout << "noise:        " << s.noise << std::endl;
    std::cout << "output:       " << s.output << std::endl;
    std::cout << "adios_config: " << s.adios_config << std::endl;
    std::cout << "====================" << std::endl;
}

int main(int argc, char **argv)
{
    int rank;
    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    Settings settings = Settings::from_json("settings.json");

    if (rank == 0) {
        print_settings(settings);
    }

    GrayScott sim(settings, MPI_COMM_WORLD);

    sim.init();

    adios2::ADIOS adios(settings.adios_config, MPI_COMM_WORLD, adios2::DebugON);

    adios2::IO io = adios.DeclareIO("SimulationOutput");

    adios2::Variable<double> varU = io.DefineVariable<double>(
        "U",
        {sim.GZ * sim.local_size_z, sim.GY * sim.local_size_y,
         sim.GX * sim.local_size_x},
        {sim.local_grid_z * sim.local_size_z,
         sim.local_grid_y * sim.local_size_y,
         sim.local_grid_x * sim.local_size_x},
        {sim.local_size_z, sim.local_size_y, sim.local_size_x});

    adios2::Variable<double> varV = io.DefineVariable<double>(
        "V",
        {sim.GZ * sim.local_size_z, sim.GY * sim.local_size_y,
         sim.GX * sim.local_size_x},
        {sim.local_grid_z * sim.local_size_z,
         sim.local_grid_y * sim.local_size_y,
         sim.local_grid_x * sim.local_size_x},
        {sim.local_size_z, sim.local_size_y, sim.local_size_x});


    adios2::Engine writer = io.Open(settings.output, adios2::Mode::Write);

    for (int i = 0; i < settings.steps; i++) {
        sim.iterate();

        if (i % settings.iterations == 0) {
            if (rank == 0) {
                std::cout << "Writing step: " << i << std::endl;
            }
            std::vector<double> u = sim.u_noghost();
            std::vector<double> v = sim.v_noghost();

            writer.BeginStep();
            writer.Put<double>(varU, u.data());
            writer.Put<double>(varV, v.data());
            writer.EndStep();
        }
    }

    writer.Close();

    MPI_Finalize();
}
