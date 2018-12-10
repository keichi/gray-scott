#ifndef __GRAY_SCOTT_H__
#define __GRAY_SCOTT_H__

#include <random>
#include <vector>

#include <mpi.h>

#include "settings.h"

class GrayScott
{
public:
    unsigned long GX, GY;
    unsigned long local_grid_x, local_grid_y;
    unsigned long local_size_x, local_size_y;

    GrayScott(const Settings &settings);
    ~GrayScott();

    void init();
    void iterate();
    std::vector<double> data_noghost() const;

protected:
    Settings settings;

    std::vector<double> u, v, u2, v2;

    int rank;
    int procs;
    int left, right, up, down;
    MPI_Comm comm;

    std::random_device rand_dev;
    std::mt19937 mt_gen;
    std::uniform_real_distribution<double> uniform_dist;

    void init_field();
    void init_mpi();

    void calc(const std::vector<double> &u, const std::vector<double> &v,
              std::vector<double> &u2, std::vector<double> &v2);
    double calcU(double tu, double tv) const;
    double calcV(double tu, double tv) const;
    double laplacian(int ix, int iy, const std::vector<double> &s) const;

    void sendrecv(std::vector<double> &u, std::vector<double> &v);
    void sendrecv_x(std::vector<double> &local_data);
    void sendrecv_y(std::vector<double> &local_data);

    // 自分の領域に含まれるか
    bool is_inside(int x, int y) const;
    // グローバル座標をローカルインデックスに
    int g2i(int gx, int gy) const;
};

#endif
