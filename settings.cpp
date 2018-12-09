#include <fstream>

#include "json.hpp"
#include "settings.h"

void to_json(nlohmann::json &j, const Settings &s)
{
    j = nlohmann::json{
        {"L", s.L},   {"steps", s.steps}, {"iterations", s.iterations},
        {"F", s.F},   {"k", s.k},         {"dt", s.dt},
        {"Du", s.Du}, {"Dv", s.Dv},
    };
}

void from_json(const nlohmann::json &j, Settings &s)
{
    j.at("L").get_to(s.L);
    j.at("steps").get_to(s.steps);
    j.at("iterations").get_to(s.iterations);
    j.at("F").get_to(s.F);
    j.at("k").get_to(s.k);
    j.at("dt").get_to(s.dt);
    j.at("Du").get_to(s.Du);
    j.at("Dv").get_to(s.Dv);
}

Settings::Settings()
{
    L = 128;
    steps = 20000;
    iterations = 200;
    F = 0.04;
    k = 0.06075;
    dt = 0.2;
    Du = 0.05;
    Dv = 0.1;
}

Settings Settings::from_json(const std::string &fname)
{
    std::ifstream ifs(fname);
    nlohmann::json j;

    ifs >> j;

    return j.get<Settings>();
}
