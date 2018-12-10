#include <mpi.h>
#include <random>
#include <vector>

#include "gray_scott.h"

GrayScott::GrayScott(const Settings &settings)
    : settings(settings), rand_dev(), mt_gen(rand_dev()),
      uniform_dist(-1.0, 1.0)
{
}

GrayScott::~GrayScott() {}

void GrayScott::init()
{
    init_mpi();
    init_field();
}

void GrayScott::iterate()
{
    sendrecv(u, v);
    calc(u, v, u2, v2);

    u.swap(u2);
    v.swap(v2);
}

std::vector<double> GrayScott::data_noghost() const
{
    std::vector<double> buf(local_size_x * local_size_y);

    const int lx = local_size_x + 2;
    const int ly = local_size_y + 2;
    for (int iy = 1; iy < ly - 1; iy++) {
        for (int ix = 1; ix < lx - 1; ix++) {
            buf[(ix - 1) + (iy - 1) * local_size_x] =
                u[ix + iy * (local_size_x + 2)];
        }
    }

    return buf;
}

bool GrayScott::is_inside(int x, int y) const
{
    int sx = local_size_x * local_grid_x;
    int sy = local_size_y * local_grid_y;
    int ex = sx + local_size_x;
    int ey = sy + local_size_y;
    if (x < sx) return false;
    if (x >= ex) return false;
    if (y < sy) return false;
    if (y >= ey) return false;
    return true;
}

int GrayScott::g2i(int gx, int gy) const
{
    int sx = local_size_x * local_grid_x;
    int sy = local_size_y * local_grid_y;
    int x = gx - sx;
    int y = gy - sy;
    return (x + 1) + (y + 1) * (local_size_x + 2);
}

void GrayScott::init_field()
{
    const int V = (local_size_x + 2) * (local_size_y + 2);
    u.resize(V, 1.0);
    v.resize(V, 0.0);
    u2.resize(V, 0.0);
    v2.resize(V, 0.0);

    int d = 6;
    for (int i = settings.L / 2 - d; i < settings.L / 2 + d; i++) {
        for (int j = settings.L / 2 - d; j < settings.L / 2 + d; j++) {
            if (!is_inside(i, j)) continue;
            int k = g2i(i, j);
            u[k] = 0.5;
            v[k] = 0.25;
        }
    }
}

double GrayScott::calcU(double tu, double tv) const
{
    return -tu * tv * tv + settings.F * (1.0 - tu);
}

double GrayScott::calcV(double tu, double tv) const
{
    return tu * tv * tv - (settings.F + settings.k) * tv;
}

double GrayScott::laplacian(int ix, int iy, const std::vector<double> &s) const
{
    double ts = 0.0;
    const int l = local_size_x + 2;
    ts += s[ix - 1 + iy * l];
    ts += s[ix + 1 + iy * l];
    ts += s[ix + (iy - 1) * l];
    ts += s[ix + (iy + 1) * l];
    ts -= 4.0 * s[ix + iy * l];
    return ts / 4.0;
}

void GrayScott::calc(const std::vector<double> &u, const std::vector<double> &v,
                     std::vector<double> &u2, std::vector<double> &v2)
{
    const int lx = local_size_x + 2;
    const int ly = local_size_y + 2;
    for (int iy = 1; iy < ly - 1; iy++) {
        for (int ix = 1; ix < lx - 1; ix++) {
            double du = 0;
            double dv = 0;
            const int i = ix + iy * lx;
            du = settings.Du * laplacian(ix, iy, u);
            dv = settings.Dv * laplacian(ix, iy, v);
            du += calcU(u[i], v[i]);
            dv += calcV(u[i], v[i]);
            du += settings.noise * uniform_dist(mt_gen);
            u2[i] = u[i] + du * settings.dt;
            v2[i] = v[i] + dv * settings.dt;
        }
    }
}

void GrayScott::init_mpi()
{
    int dims[2] = {};
    int periods[2] = {1, 1};
    int coords[2] = {};

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &procs);

    MPI_Dims_create(procs, 2, dims);
    GX = dims[0];
    GY = dims[1];
    local_size_x = settings.L / GX;
    local_size_y = settings.L / GY;

    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &comm);
    MPI_Cart_coords(comm, rank, 2, coords);
    local_grid_x = coords[0];
    local_grid_y = coords[1];

    MPI_Cart_shift(comm, 0, 1, &left, &right);
    MPI_Cart_shift(comm, 1, 1, &down, &up);

    MPI_Type_contiguous(local_size_x + 2, MPI_DOUBLE, &row_type);
    MPI_Type_commit(&row_type);

    MPI_Type_vector(local_size_y, 1, local_size_x + 2, MPI_DOUBLE, &col_type);
    MPI_Type_commit(&col_type);
}

void GrayScott::sendrecv_x(std::vector<double> &local_data)
{
    const int lx = local_size_x;
    MPI_Status st;
    MPI_Sendrecv(&local_data[lx + 1 * (lx + 2)], 1, col_type, right, 0,
                 &local_data[0 + 1 * (lx + 2)], 1, col_type, left, 0, comm,
                 &st);
    MPI_Sendrecv(&local_data[1 + 1 * (lx + 2)], 1, col_type, left, 0,
                 &local_data[(lx + 1) + 1 * (lx + 2)], 1, col_type, right, 0,
                 comm, &st);
}

void GrayScott::sendrecv_y(std::vector<double> &local_data)
{
    const int lx = local_size_x;
    const int ly = local_size_y;
    MPI_Status st;
    MPI_Sendrecv(&local_data[lx + 2], 1, row_type, up, 0,
                 &local_data[(ly + 1) * (lx + 2)], 1, row_type, down, 0, comm,
                 &st);
    MPI_Sendrecv(&local_data[ly * (lx + 2)], 1, row_type, down, 0,
                 &local_data[0 * (lx + 2)], 1, row_type, up, 0, comm, &st);
}

void GrayScott::sendrecv(std::vector<double> &u, std::vector<double> &v)
{
    sendrecv_x(u);
    sendrecv_y(u);
    sendrecv_x(v);
    sendrecv_y(v);
}
