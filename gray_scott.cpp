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

void GrayScott::dump() const
{
    const int lx = local_size_x + 2;
    const int ly = local_size_y + 2;
    const int lz = local_size_z + 2;

    for (int iz = 0; iz < lz; iz++) {
        std::cout << "z=" << iz << std::endl;
        for (int iy = ly - 1; iy >= 0; iy--) {
            for (int ix = 0; ix < lx; ix++) {
                std::cout << std::fixed << std::setprecision(2)
                          << u[l2i(ix, iy, iz)] << " ";
            }
            std::cout << std::endl;
        }

        std::cout << std::endl;
    }
}

std::vector<double> GrayScott::u_noghost() const { return data_noghost(u); }

std::vector<double> GrayScott::v_noghost() const { return data_noghost(v); }

std::vector<double>
GrayScott::data_noghost(const std::vector<double> &data) const
{
    std::vector<double> buf(local_size_x * local_size_y * local_size_z);

    const int lx = local_size_x + 2;
    const int ly = local_size_y + 2;
    const int lz = local_size_z + 2;

    for (int iz = 1; iz < lz - 1; iz++) {
        for (int iy = 1; iy < ly - 1; iy++) {
            for (int ix = 1; ix < lx - 1; ix++) {
                buf[(ix - 1) + (iy - 1) * local_size_x +
                    (iz - 1) * local_size_x * local_size_y] =
                    data[l2i(ix, iy, iz)];
            }
        }
    }

    return buf;
}

bool GrayScott::is_inside(int x, int y, int z) const
{
    int sx = local_size_x * local_grid_x;
    int sy = local_size_y * local_grid_y;
    int sz = local_size_z * local_grid_z;

    int ex = sx + local_size_x;
    int ey = sy + local_size_y;
    int ez = sz + local_size_z;

    if (x < sx) return false;
    if (x >= ex) return false;
    if (y < sy) return false;
    if (y >= ey) return false;
    if (z < sz) return false;
    if (z >= ez) return false;

    return true;
}

int GrayScott::g2i(int gx, int gy, int gz) const
{
    int sx = local_size_x * local_grid_x;
    int sy = local_size_y * local_grid_y;
    int sz = local_size_z * local_grid_z;

    int x = gx - sx;
    int y = gy - sy;
    int z = gz - sz;

    return l2i(x + 1, y + 1, z + 1);
}

void GrayScott::init_field()
{
    const int V = (local_size_x + 2) * (local_size_y + 2) * (local_size_z + 2);
    u.resize(V, 1.0);
    v.resize(V, 0.0);
    u2.resize(V, 0.0);
    v2.resize(V, 0.0);

    int d = 6;
    for (int i = settings.L / 2 - d; i < settings.L / 2 + d; i++) {
        for (int j = settings.L / 2 - d; j < settings.L / 2 + d; j++) {
            for (int k = settings.L / 2 - d; k < settings.L / 2 + d; k++) {
                if (!is_inside(i, j, k)) continue;
                int ix = g2i(i, j, k);
                u[ix] = 0.25;
                v[ix] = 0.33;
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

double GrayScott::laplacian(int ix, int iy, int iz,
                            const std::vector<double> &s) const
{
    double ts = 0.0;
    ts += s[l2i(ix - 1, iy, iz)];
    ts += s[l2i(ix + 1, iy, iz)];
    ts += s[l2i(ix, iy - 1, iz)];
    ts += s[l2i(ix, iy + 1, iz)];
    ts += s[l2i(ix, iy, iz - 1)];
    ts += s[l2i(ix, iy, iz + 1)];
    ts += -6.0 * s[l2i(ix, iy, iz)];

    return ts / 6.0;
}

void GrayScott::calc(const std::vector<double> &u, const std::vector<double> &v,
                     std::vector<double> &u2, std::vector<double> &v2)
{
    const int lx = local_size_x + 2;
    const int ly = local_size_y + 2;
    const int lz = local_size_z + 2;

    for (int iz = 1; iz < lz - 1; iz++) {
        for (int iy = 1; iy < ly - 1; iy++) {
            for (int ix = 1; ix < lx - 1; ix++) {
                const int i = l2i(ix, iy, iz);
                double du = 0.0;
                double dv = 0.0;
                du = settings.Du * laplacian(ix, iy, iz, u);
                dv = settings.Dv * laplacian(ix, iy, iz, v);
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
    int periods[3] = {1, 1, 1};
    int coords[3] = {};

    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &procs);

    MPI_Dims_create(procs, 3, dims);
    GX = dims[0];
    GY = dims[1];
    GZ = dims[2];
    local_size_x = settings.L / GX;
    local_size_y = settings.L / GY;
    local_size_z = settings.L / GZ;

    MPI_Cart_create(comm, 3, dims, periods, 0, &cart_comm);
    MPI_Cart_coords(cart_comm, rank, 3, coords);
    local_grid_x = coords[0];
    local_grid_y = coords[1];
    local_grid_z = coords[2];

    MPI_Cart_shift(cart_comm, 0, 1, &west, &east);
    MPI_Cart_shift(cart_comm, 1, 1, &down, &up);
    MPI_Cart_shift(cart_comm, 2, 1, &south, &north);

    // XY face: (local_size_x + 2) * (local_Size_y + 2)
    MPI_Type_vector((local_size_x + 2) * (local_size_y + 2), 1,
                    local_size_z + 2, MPI_DOUBLE, &xy_face_type);
    MPI_Type_commit(&xy_face_type);

    // XZ face: loca_size_x * local_size_z
    MPI_Type_vector(local_size_x, local_size_z,
                    (local_size_y + 2) * (local_size_z + 2), MPI_DOUBLE,
                    &xz_face_type);
    MPI_Type_commit(&xz_face_type);

    // YZ face: (loca_size_y + 2) * local_size_z
    MPI_Type_vector(local_size_y + 2, local_size_z, local_size_z + 2,
                    MPI_DOUBLE, &yz_face_type);
    MPI_Type_commit(&yz_face_type);
}

// Exchange XY face with north/south
void GrayScott::sendrecv_xy(std::vector<double> &local_data)
{
    const int lz = local_size_z;
    MPI_Status st;

    // Send XY surface z=lz to north and receive surface z=0 from south
    MPI_Sendrecv(&local_data[l2i(0, 0, lz)], 1, xy_face_type, north, 1,
                 &local_data[l2i(0, 0, 0)], 1, xy_face_type, south, 1,
                 cart_comm, &st);
    // Send XY surface z=1 to south and receive surface z=lz+1 from north
    MPI_Sendrecv(&local_data[l2i(0, 0, 1)], 1, xy_face_type, south, 1,
                 &local_data[l2i(0, 0, lz + 1)], 1, xy_face_type, north, 1,
                 cart_comm, &st);
}

// Exchange XZ face with up/down
void GrayScott::sendrecv_xz(std::vector<double> &local_data)
{
    const int ly = local_size_y;
    MPI_Status st;

    // Send XZ surface y=ly to up and receive surface y=0 from down
    MPI_Sendrecv(&local_data[l2i(1, ly, 1)], 1, xz_face_type, up, 2,
                 &local_data[l2i(1, 0, 1)], 1, xz_face_type, down, 2, cart_comm,
                 &st);
    // Send XZ surface y=1 to down and receive surface y=ly+1 from up
    MPI_Sendrecv(&local_data[l2i(1, 1, 1)], 1, xz_face_type, down, 2,
                 &local_data[l2i(1, ly + 1, 1)], 1, xz_face_type, up, 2,
                 cart_comm, &st);
}

// Exchange YZ face with west/east
void GrayScott::sendrecv_yz(std::vector<double> &local_data)
{
    const int lx = local_size_x;
    MPI_Status st;

    // Send YZ surface x=lx to east and receive surface x=0 from west
    MPI_Sendrecv(&local_data[l2i(lx, 0, 1)], 1, yz_face_type, east, 3,
                 &local_data[l2i(0, 0, 1)], 1, yz_face_type, west, 3, cart_comm,
                 &st);
    // Send YZ surface x=1 to west and receive surface x=lx+1 from east
    MPI_Sendrecv(&local_data[l2i(0, 0, 1)], 1, yz_face_type, west, 3,
                 &local_data[l2i(lx + 1, 0, 1)], 1, yz_face_type, east, 3,
                 cart_comm, &st);
}

void GrayScott::sendrecv(std::vector<double> &u, std::vector<double> &v)
{
    sendrecv_xy(u);
    sendrecv_xz(u);
    sendrecv_yz(u);

    sendrecv_xy(v);
    sendrecv_xz(v);
    sendrecv_yz(v);
}
