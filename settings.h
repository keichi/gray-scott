#ifndef __SETTINGS_H__
#define __SETTINGS_H__

#include <string>

class Settings
{
public:
    int L;
    int steps;
    int iterations;
    double F;
    double k;
    double dt;
    double Du;
    double Dv;
    double noise;

    Settings();
    static Settings from_json(const std::string &fname);
};

#endif
