#include <cstdio>
#include <fstream>
#include <iostream>
#include <mpi.h>
#include <vector>

#include <adios2.h>

#include "gray_scott.h"

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    Settings settings;
    GrayScott sim(settings);

    sim.init();

    adios2::ADIOS adios("adios2_config.xml", MPI_COMM_WORLD, adios2::DebugON);

    adios2::IO io = adios.DeclareIO("SimulationOutput");

    adios2::Variable<double> varT = io.DefineVariable<double>(
        "T", {sim.GY * sim.local_size_y, sim.GX * sim.local_size_x},
        {sim.local_grid_y * sim.local_size_y,
         sim.local_grid_x * sim.local_size_x},
        {sim.local_size_y, sim.local_size_x});

    adios2::Engine writer = io.Open("foo.bp", adios2::Mode::Write);

    for (int i = 0; i < settings.TOTAL_STEP; i++) {
        sim.iterate();

        if (i % settings.INTERVAL == 0) {
            std::vector<double> u = sim.data_noghost();

            writer.BeginStep();
            writer.Put<double>(varT, u.data());
            writer.EndStep();
        }
    }
    MPI_Finalize();
}
