// Custom stepper pulse time generation
//
// Copyright (C) 2024  Erdene Luvsandorj <erdene.lu.n@gmail.com>
//
// This file may be distributed under the terms of the GNU GPLv3 license.

#include <math.h>   // 
#include <stdlib.h> // malloc
#include <string.h> // memset
#include "compiler.h" // __visible
#include "itersolve.h" // struct stepper_kinematics
#include "pyhelper.h" // errorf
#include "trapq.h" // move_get_coord
#include <stddef.h>

struct custom_stepper {
    struct stepper_kinematics sk;
    double l0;      //  l0=distance between shoulder and z
    double l1, l2;  //  l1=shoulder, l2=arm
};

/** 
    custom kinematics
    x**2+y**2=(l0+l1*cos(a)+l2*cos(b))**2
    l1*sin(a)-l2*sin(b)=z
**/

static inline double
get_r(double l0, double x, double y)
{
    return sqrt(x*x+y*y)-l0;
}

static inline double
get_p(double l0, double l1, double l2, double x, double y, double z)
{
    double r = get_r(l0, x, y);
    return (r*r+z*z-l1*l1-l2*l2)/(2*l1*l2);
}

static inline double
get_shoulder_angle(double l0, double l1, double l2, double x, double y, double z)
{
    double cos_p = get_p(l0, l1, l2, x, y, z);
    double a = l1+l2*cos_p;
    double b = l2*sqrt(1-cos_p*cos_p);
    double d = sqrt(a*a+b*b);
    double angle = asin(z/d)+asin(b/d);
    return angle;
}

static inline double
get_arm_angle(double l0, double l1, double l2, double x, double y, double z)
{
    double cos_p = get_p(l0, l1, l2, x, y, z);
    double shoulder_angle = get_shoulder_angle(l0, l1, l2, x, y, z);
    return acos(cos_p) - shoulder_angle;
}

static inline double
get_bed_angle(double x, double y)
{
    return asin(x/sqrt(x*x+y*y));
}

static double
custom_stepper_bed_angle_calc(struct stepper_kinematics *sk, struct move *m, double move_time)
{
    struct coord c = move_get_coord(m, move_time);
    return get_bed_angle(c.x, c.y);
}

static double
custom_stepper_shoulder_angle_calc(struct stepper_kinematics *sk, struct move *m, double move_time)
{
    struct custom_stepper *fs = container_of(
                sk, struct custom_stepper, sk);
    struct coord c = move_get_coord(m, move_time);
    return get_shoulder_angle(fs->l0, fs->l1, fs->l2, c.x, c.y, c.z);
}

static double
custom_stepper_arm_angle_calc(struct stepper_kinematics *sk, struct move *m, double move_time)
{
    struct custom_stepper *fs = container_of(
                sk, struct custom_stepper, sk);
    struct coord c = move_get_coord(m, move_time);
    return get_arm_angle(fs->l0, fs->l1, fs->l2, c.x, c.y, c.z);
}

struct stepper_kinematics * __visible
custom_stepper_alloc(char type
    , double l0, double l1, double l2)
{
    struct custom_stepper *fs = malloc(sizeof(*fs));
    memset(fs, 0, sizeof(*fs));
    fs->l0 = l0;
    fs->l1 = l1;
    fs->l2 = l2;
    if (type == 'b') {
        fs->sk.calc_position_cb = custom_stepper_bed_angle_calc;
        fs->sk.active_flags = AF_X | AF_Y;
    } else if (type == 's') {
        fs->sk.calc_position_cb = custom_stepper_shoulder_angle_calc;
        fs->sk.active_flags = AF_X | AF_Y | AF_Z;
    } else if (type == 'a') {
        fs->sk.calc_position_cb = custom_stepper_arm_angle_calc;
        fs->sk.active_flags = AF_X | AF_Y | AF_Z;
    }
    return &fs->sk;
}
