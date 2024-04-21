// Inverse kinematics stepper pulse time generation
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

struct inverse_stepper {
    struct stepper_kinematics sk;
    double l0;      //  l0=distance between shoulder and z
    double l1, l2;  //  l1=shoulder, l2=arm
    double angle1, angle2;   //  angle1=between shoulder and xy, angle2= between shoulder and arm 
};

static inline double
degree_to_redians(double degree)
{
    return degree / 180.0 * M_PI;
}

/** 
    inverse kinematics
    l0 + l1 * cos(a) + l2 * cos(a+b) = sqrt(x*x + y*y)
    l1 * sin(a) + l2 * sin(a+b) = z
**/

static inline double
get_radius(double l0, double x, double y)
{
    return sqrt(x*x+y*y)-l0;
}

static inline double
calc_arm_angle_cos(double l0, double l1, double l2, double x, double y, double z)
{
    if (l1 == 0.0)
        return 0.0;
    if (l2 == 0.0)
        return 0.0;
    double r = get_radius(l0, x, y);
    return (r*r+z*z-l1*l1-l2*l2)/(2*l1*l2);
}

static double
inverse_stepper_bed_angle_calc(struct stepper_kinematics *sk, struct move *m, double move_time)
{
    struct inverse_stepper *fs = container_of(
                sk, struct inverse_stepper, sk);
    struct coord c = move_get_coord(m, move_time);
    if (c.y == 0)
        return M_PI/2;
    return atan2(c.x, c.y);
}

static double
inverse_stepper_shoulder_angle_calc(struct stepper_kinematics *sk, struct move *m, double move_time)
{
    struct inverse_stepper *fs = container_of(
                sk, struct inverse_stepper, sk);
    struct coord c = move_get_coord(m, move_time);
    double cos_b = calc_arm_angle_cos(fs->l0, fs->l1, fs->l2, c.x, c.y, c.z);
    double sin_b = sqrt(1 - cos_b*cos_b);
    double r = get_radius(fs->l0, c.x, c.y);
    double d = fs->l1+fs->l2*cos_b;
    double angle = atan2(c.z, r) - (d==0 ? M_PI/2 : atan2(fs->l2*sin_b, d));
    angle -= degree_to_redians(fs->angle1);
    return angle;
}

static double
inverse_stepper_arm_angle_calc(struct stepper_kinematics *sk, struct move *m, double move_time)
{
    struct inverse_stepper *fs = container_of(
                sk, struct inverse_stepper, sk);
    struct coord c = move_get_coord(m, move_time);
    double angle = acos(calc_arm_angle_cos(fs->l0, fs->l1, fs->l2, c.x, c.y, c.z));
    angle -= degree_to_redians(fs->angle2);
    return angle;
}

struct stepper_kinematics * __visible
inverse_stepper_alloc(char type
    , double l0, double l1, double l2
    , double angle1, double angle2)
{
    struct inverse_stepper *fs = malloc(sizeof(*fs));
    memset(fs, 0, sizeof(*fs));
    fs->l0 = l0;
    fs->l1 = l1;
    fs->l2 = l2;
    fs->angle1 = angle1;
    fs->angle2 = angle2;
    if (type == 'b') {
        fs->sk.calc_position_cb = inverse_stepper_bed_angle_calc;
        fs->sk.active_flags = AF_X | AF_Y;
    } else if (type == 's') {
        fs->sk.calc_position_cb = inverse_stepper_shoulder_angle_calc;
        fs->sk.active_flags = AF_X | AF_Y | AF_Z;
    } else if (type == 'a') {
        fs->sk.calc_position_cb = inverse_stepper_arm_angle_calc;
        fs->sk.active_flags = AF_X | AF_Y | AF_Z;
    }
    return &fs->sk;
}