#include <cstdio>
#include <fstream>
#include <iostream>
#include <mpi.h>
#include <vector>

#include <adios2.h>

const int L = 128;
const int TOTAL_STEP = 20000;
const int INTERVAL = 200;
const double F = 0.04;
const double k = 0.06075;
const double dt = 0.2;
const double Du = 0.05;
const double Dv = 0.1;

struct MPIinfo {
    int rank;
    int procs;
    int GX, GY;
    int local_grid_x, local_grid_y;
    int local_size_x, local_size_y;

    // 自分から見て +dx, +dyだけずれたプロセスのランクを返す
    int get_rank(int dx, int dy)
    {
        int rx = (local_grid_x + dx + GX) % GX;
        int ry = (local_grid_y + dy + GY) % GY;
        return rx + ry * GX;
    }

    // 自分の領域に含まれるか
    bool is_inside(int x, int y)
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
    // グローバル座標をローカルインデックスに
    int g2i(int gx, int gy)
    {
        int sx = local_size_x * local_grid_x;
        int sy = local_size_y * local_grid_y;
        int x = gx - sx;
        int y = gy - sy;
        return (x + 1) + (y + 1) * (local_size_x + 2);
    }
};

void init(std::vector<double> &u, std::vector<double> &v, MPIinfo &mi)
{
    int d = 3;
    for (int i = L / 2 - d; i < L / 2 + d; i++) {
        for (int j = L / 2 - d; j < L / 2 + d; j++) {
            if (!mi.is_inside(i, j)) continue;
            int k = mi.g2i(i, j);
            u[k] = 0.7;
        }
    }
    d = 6;
    for (int i = L / 2 - d; i < L / 2 + d; i++) {
        for (int j = L / 2 - d; j < L / 2 + d; j++) {
            if (!mi.is_inside(i, j)) continue;
            int k = mi.g2i(i, j);
            v[k] = 0.9;
        }
    }
}

double calcU(double tu, double tv) { return tu * tu * tv - (F + k) * tu; }

double calcV(double tu, double tv) { return -tu * tu * tv + F * (1.0 - tv); }

double laplacian(int ix, int iy, std::vector<double> &s, MPIinfo &mi)
{
    double ts = 0.0;
    const int l = mi.local_size_x + 2;
    ts += s[ix - 1 + iy * l];
    ts += s[ix + 1 + iy * l];
    ts += s[ix + (iy - 1) * l];
    ts += s[ix + (iy + 1) * l];
    ts -= 4.0 * s[ix + iy * l];
    return ts;
}

void calc(std::vector<double> &u, std::vector<double> &v,
          std::vector<double> &u2, std::vector<double> &v2, MPIinfo &mi)
{
    const int lx = mi.local_size_x + 2;
    const int ly = mi.local_size_y + 2;
    for (int iy = 1; iy < ly - 1; iy++) {
        for (int ix = 1; ix < lx - 1; ix++) {
            double du = 0;
            double dv = 0;
            const int i = ix + iy * lx;
            du = Du * laplacian(ix, iy, u, mi);
            dv = Dv * laplacian(ix, iy, v, mi);
            du += calcU(u[i], v[i]);
            dv += calcV(u[i], v[i]);
            u2[i] = u[i] + du * dt;
            v2[i] = v[i] + dv * dt;
        }
    }
}

void setup_info(MPIinfo &mi)
{
    int rank = 0;
    int procs = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &procs);
    int d2[2] = {};
    MPI_Dims_create(procs, 2, d2);
    mi.rank = rank;
    mi.procs = procs;
    mi.GX = d2[0];
    mi.GY = d2[1];
    mi.local_grid_x = rank % mi.GX;
    mi.local_grid_y = rank / mi.GX;
    mi.local_size_x = L / mi.GX;
    mi.local_size_y = L / mi.GY;
}

void sendrecv_x(std::vector<double> &local_data, MPIinfo &mi)
{
    const int lx = mi.local_size_x;
    const int ly = mi.local_size_y;
    std::vector<double> sendbuf(ly);
    std::vector<double> recvbuf(ly);
    int left = mi.get_rank(-1, 0);
    int right = mi.get_rank(1, 0);
    for (int i = 0; i < ly; i++) {
        int index = lx + (i + 1) * (lx + 2);
        sendbuf[i] = local_data[index];
    }
    MPI_Status st;
    MPI_Sendrecv(sendbuf.data(), ly, MPI_DOUBLE, right, 0, recvbuf.data(), ly,
                 MPI_DOUBLE, left, 0, MPI_COMM_WORLD, &st);
    for (int i = 0; i < ly; i++) {
        int index = (i + 1) * (lx + 2);
        local_data[index] = recvbuf[i];
    }

    for (int i = 0; i < ly; i++) {
        int index = 1 + (i + 1) * (lx + 2);
        sendbuf[i] = local_data[index];
    }
    MPI_Sendrecv(sendbuf.data(), ly, MPI_DOUBLE, left, 0, recvbuf.data(), ly,
                 MPI_DOUBLE, right, 0, MPI_COMM_WORLD, &st);
    for (int i = 0; i < ly; i++) {
        int index = lx + 1 + (i + 1) * (lx + 2);
        local_data[index] = recvbuf[i];
    }
}

void sendrecv_y(std::vector<double> &local_data, MPIinfo &mi)
{
    const int lx = mi.local_size_x;
    const int ly = mi.local_size_y;
    std::vector<double> sendbuf(lx + 2);
    std::vector<double> recvbuf(lx + 2);
    int up = mi.get_rank(0, -1);
    int down = mi.get_rank(0, 1);
    MPI_Status st;
    // 上に投げて下から受け取る
    for (int i = 0; i < lx + 2; i++) {
        int index = i + 1 * (lx + 2);
        sendbuf[i] = local_data[index];
    }
    MPI_Sendrecv(sendbuf.data(), lx + 2, MPI_DOUBLE, up, 0, recvbuf.data(),
                 lx + 2, MPI_DOUBLE, down, 0, MPI_COMM_WORLD, &st);
    for (int i = 0; i < lx + 2; i++) {
        int index = i + (ly + 1) * (lx + 2);
        local_data[index] = recvbuf[i];
    }
    // 下に投げて上から受け取る
    for (int i = 0; i < lx + 2; i++) {
        int index = i + (ly) * (lx + 2);
        sendbuf[i] = local_data[index];
    }
    MPI_Sendrecv(sendbuf.data(), lx + 2, MPI_DOUBLE, down, 0, recvbuf.data(),
                 lx + 2, MPI_DOUBLE, up, 0, MPI_COMM_WORLD, &st);
    for (int i = 0; i < lx + 2; i++) {
        int index = i + 0 * (lx + 2);
        local_data[index] = recvbuf[i];
    }
}

void sendrecv(std::vector<double> &u, std::vector<double> &v, MPIinfo &mi)
{
    sendrecv_x(u, mi);
    sendrecv_y(u, mi);
    sendrecv_x(v, mi);
    sendrecv_y(v, mi);
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    MPIinfo mi;
    setup_info(mi);
    const int V = (mi.local_size_x + 2) * (mi.local_size_y + 2);
    std::vector<double> u(V, 0.0), v(V, 0.0);
    std::vector<double> u2(V, 0.0), v2(V, 0.0);

    adios2::ADIOS adios(MPI_COMM_WORLD);
    adios2::IO io = adios.DeclareIO("Output");
    adios2::Variable<double> varT = io.DefineVariable<double>(
        "T",
        {static_cast<unsigned long>(mi.GX * mi.local_size_x),
         static_cast<unsigned long>(mi.GY * mi.local_size_y)},
        {static_cast<unsigned long>(mi.local_grid_x * mi.local_size_x),
         static_cast<unsigned long>(mi.local_grid_y * mi.local_size_y)},
        {static_cast<unsigned long>(mi.local_size_x),
         static_cast<unsigned long>(mi.local_size_y)});
    varT.SetMemorySelection(
        {{1, 1},
         {static_cast<unsigned long>(mi.local_size_x + 2),
          static_cast<unsigned long>(mi.local_size_y + 2)}});

    adios2::Engine writer = io.Open("foo.bp", adios2::Mode::Write);

    init(u, v, mi);
    for (int i = 0; i < TOTAL_STEP; i++) {
        if (i & 1) {
            sendrecv(u2, v2, mi);
            calc(u2, v2, u, v, mi);
        } else {
            sendrecv(u, v, mi);
            calc(u, v, u2, v2, mi);
        }
        if (i % INTERVAL == 0) {
            writer.BeginStep();
            writer.Put<double>(varT, u.data());
            writer.EndStep();
        }
    }
    MPI_Finalize();
}
