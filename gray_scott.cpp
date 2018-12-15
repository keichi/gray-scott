#include <iomanip>
#include <mpi.h>
#include <random>
#include <vector>

#include "gray_scott.h"

GrayScott::GrayScott(const Settings &settings, MPI_Comm comm)
    : settings(settings), comm(comm), rand_dev(), mt_gen(rand_dev()),
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

std::vector<double> GrayScott::u_noghost() const { return data_noghost(u); }

std::vector<double> GrayScott::v_noghost() const { return data_noghost(v); }

std::vector<double>
GrayScott::data_noghost(const std::vector<double> &data) const
{
    std::vector<double> buf(size_x * size_y * size_z);

    for (int x = 1; x < size_x + 1; x++) {
        for (int y = 1; y < size_y + 1; y++) {
            for (int z = 1; z < size_z + 1; z++) {
                buf[(x - 1) + (y - 1) * size_x + (z - 1) * size_x * size_y] =
                    data[l2i(x, y, z)];
            }
        }
    }

    return buf;
}

void GrayScott::init_field()
{
    const int V = (size_x + 2) * (size_y + 2) * (size_z + 2);
    u.resize(V, 1.0);
    v.resize(V, 0.0);
    u2.resize(V, 0.0);
    v2.resize(V, 0.0);

    const int d = 6;
    for (int x = settings.L / 2 - d; x < settings.L / 2 + d; x++) {
        for (int y = settings.L / 2 - d; y < settings.L / 2 + d; y++) {
            for (int z = settings.L / 2 - d; z < settings.L / 2 + d; z++) {
                if (!is_inside(x, y, z)) continue;
                int i = g2i(x, y, z);
                u[i] = 0.25;
                v[i] = 0.33;
            }
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

double GrayScott::laplacian(int x, int y, int z,
                            const std::vector<double> &s) const
{
    double ts = 0.0;
    ts += s[l2i(x - 1, y, z)];
    ts += s[l2i(x + 1, y, z)];
    ts += s[l2i(x, y - 1, z)];
    ts += s[l2i(x, y + 1, z)];
    ts += s[l2i(x, y, z - 1)];
    ts += s[l2i(x, y, z + 1)];
    ts += -6.0 * s[l2i(x, y, z)];

    return ts / 6.0;
}

void GrayScott::calc(const std::vector<double> &u, const std::vector<double> &v,
                     std::vector<double> &u2, std::vector<double> &v2)
{
    for (int x = 1; x < size_x + 1; x++) {
        for (int y = 1; y < size_y + 1; y++) {
            for (int z = 1; z < size_z + 1; z++) {
                const int i = l2i(x, y, z);
                double du = 0.0;
                double dv = 0.0;
                du = settings.Du * laplacian(x, y, z, u);
                dv = settings.Dv * laplacian(x, y, z, v);
                du += calcU(u[i], v[i]);
                dv += calcV(u[i], v[i]);
                du += settings.noise * uniform_dist(mt_gen);
                u2[i] = u[i] + du * settings.dt;
                v2[i] = v[i] + dv * settings.dt;
            }
        }
    }
}

void GrayScott::init_mpi()
{
    int dims[3] = {};
    const int periods[3] = {1, 1, 1};
    int coords[3] = {};

    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &procs);

    MPI_Dims_create(procs, 3, dims);
    npx = dims[0];
    npy = dims[1];
    npz = dims[2];
    size_x = settings.L / npx;
    size_y = settings.L / npy;
    size_z = settings.L / npz;

    MPI_Cart_create(comm, 3, dims, periods, 0, &cart_comm);
    MPI_Cart_coords(cart_comm, rank, 3, coords);
    px = coords[0];
    py = coords[1];
    pz = coords[2];

    MPI_Cart_shift(cart_comm, 0, 1, &west, &east);
    MPI_Cart_shift(cart_comm, 1, 1, &down, &up);
    MPI_Cart_shift(cart_comm, 2, 1, &south, &north);

    // XY faces: (size_x + 2) * (size_y + 2)
    MPI_Type_vector((size_x + 2) * (size_y + 2), 1, size_z + 2, MPI_DOUBLE,
                    &xy_face_type);
    MPI_Type_commit(&xy_face_type);

    // XZ faces: loca_size_x * size_z
    MPI_Type_vector(size_x, size_z, (size_y + 2) * (size_z + 2), MPI_DOUBLE,
                    &xz_face_type);
    MPI_Type_commit(&xz_face_type);

    // YZ faces: (loca_size_y + 2) * size_z
    MPI_Type_vector(size_y + 2, size_z, size_z + 2, MPI_DOUBLE, &yz_face_type);
    MPI_Type_commit(&yz_face_type);
}

void GrayScott::sendrecv_xy(std::vector<double> &local_data) const
{
    const int lz = size_z;
    MPI_Status st;

    // Send XY face z=lz to north and receive z=0 from south
    MPI_Sendrecv(&local_data[l2i(0, 0, lz)], 1, xy_face_type, north, 1,
                 &local_data[l2i(0, 0, 0)], 1, xy_face_type, south, 1,
                 cart_comm, &st);
    // Send XY face z=1 to south and receive z=lz+1 from north
    MPI_Sendrecv(&local_data[l2i(0, 0, 1)], 1, xy_face_type, south, 1,
                 &local_data[l2i(0, 0, lz + 1)], 1, xy_face_type, north, 1,
                 cart_comm, &st);
}

void GrayScott::sendrecv_xz(std::vector<double> &local_data) const
{
    const int ly = size_y;
    MPI_Status st;

    // Send XZ face y=ly to up and receive y=0 from down
    MPI_Sendrecv(&local_data[l2i(1, ly, 1)], 1, xz_face_type, up, 2,
                 &local_data[l2i(1, 0, 1)], 1, xz_face_type, down, 2, cart_comm,
                 &st);
    // Send XZ face y=1 to down and receive y=ly+1 from up
    MPI_Sendrecv(&local_data[l2i(1, 1, 1)], 1, xz_face_type, down, 2,
                 &local_data[l2i(1, ly + 1, 1)], 1, xz_face_type, up, 2,
                 cart_comm, &st);
}

void GrayScott::sendrecv_yz(std::vector<double> &local_data) const
{
    const int lx = size_x;
    MPI_Status st;

    // Send YZ face x=lx to east and receive x=0 from west
    MPI_Sendrecv(&local_data[l2i(lx, 0, 1)], 1, yz_face_type, east, 3,
                 &local_data[l2i(0, 0, 1)], 1, yz_face_type, west, 3, cart_comm,
                 &st);
    // Send YZ face x=1 to west and receive x=lx+1 from east
    MPI_Sendrecv(&local_data[l2i(1, 0, 1)], 1, yz_face_type, west, 3,
                 &local_data[l2i(lx + 1, 0, 1)], 1, yz_face_type, east, 3,
                 cart_comm, &st);
}

void GrayScott::sendrecv(std::vector<double> &u, std::vector<double> &v) const
{
    sendrecv_xy(u);
    sendrecv_xz(u);
    sendrecv_yz(u);

    sendrecv_xy(v);
    sendrecv_xz(v);
    sendrecv_yz(v);
}
