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

    Settings settings = Settings::from_json("settings.json");
    GrayScott sim(settings);

    sim.init();

    adios2::ADIOS adios("adios2_config.xml", MPI_COMM_WORLD, adios2::DebugON);

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


    adios2::Engine writer = io.Open("foo.bp", adios2::Mode::Write);

    for (int i = 0; i < settings.steps; i++) {
        sim.iterate();

        if (i % settings.iterations == 0) {
            std::cout << "Writing step: " << i << std::endl;
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
