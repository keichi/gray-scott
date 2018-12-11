#ifndef __GRAY_SCOTT_H__
#define __GRAY_SCOTT_H__

#include <random>
#include <vector>

#include <mpi.h>

#include "settings.h"

class GrayScott
{
public:
    unsigned long GX, GY, GZ;
    unsigned long local_grid_x, local_grid_y, local_grid_z;
    unsigned long local_size_x, local_size_y, local_size_z;

    GrayScott(const Settings &settings, MPI_Comm comm);
    ~GrayScott();

    void init();
    void iterate();
    std::vector<double> u_noghost() const;
    std::vector<double> v_noghost() const;
    void dump() const;

protected:
    Settings settings;

    std::vector<double> u, v, u2, v2;

    int rank;
    int procs;
    int west, east, up, down, north, south;
    MPI_Comm comm;
    MPI_Comm cart_comm;

    MPI_Datatype xy_face_type;
    MPI_Datatype xz_face_type;
    MPI_Datatype yz_face_type;

    std::random_device rand_dev;
    std::mt19937 mt_gen;
    std::uniform_real_distribution<double> uniform_dist;

    void init_field();
    void init_mpi();

    void calc(const std::vector<double> &u, const std::vector<double> &v,
              std::vector<double> &u2, std::vector<double> &v2);
    double calcU(double tu, double tv) const;
    double calcV(double tu, double tv) const;
    double laplacian(int ix, int iy, int iz,
                     const std::vector<double> &s) const;

    void sendrecv(std::vector<double> &u, std::vector<double> &v);
    void sendrecv_xy(std::vector<double> &local_data);
    void sendrecv_xz(std::vector<double> &local_data);
    void sendrecv_yz(std::vector<double> &local_data);

    std::vector<double> data_noghost(const std::vector<double> &data) const;

    // Check if point is included in my subdomain
    bool is_inside(int x, int y, int z) const;
    // Convert global coordinate to local index
    int g2i(int gx, int gy, int gz) const;
    // Convert local coordinate to local index
    inline int l2i(int x, int y, int z) const
    {
        return z + y * (local_size_z + 2) +
               x * (local_size_y + 2) * (local_size_z + 2);
    }
};

#endif
