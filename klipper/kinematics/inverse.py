# Code for handling the inverse kinematics
#
# Copyright (C) 2024  Erdene Luvsandorj <erdene.lu.n@gmail.com>
#
# This file may be distributed under the terms of the GNU GPLv3 license.

import math
import stepper
import logging

class InverseKinematics:
    def __init__(self, toolhead, config):
        
        # printer config
        self.l0 = config.getfloat('l0', above=0.)
        self.l1 = config.getfloat('l1', above=0.)
        self.l2 = config.getfloat('l2', above=0.)
        self.angle1 = config.getfloat('angle1')
        self.angle2 = config.getfloat('angle2')
        
        # limit
        # l1 bend limit
        self.angle1_min = math.degrees(10)
        # distance limit
        self.r = self.l0+self.l1*math.cos(self.angle1_min)+self.l2 

        # Setup steppers
        self.steppers = []
        for type in 'bas':
            s = stepper.PrinterStepper(config.getsection('stepper_'+type), True)
            s.setup_itersolve('inverse_stepper_alloc', type.encode()
                , self.l0, self.l1, self.l2
                , self.angle1, self.angle2)
            s.set_trapq(toolhead.get_trapq())
            toolhead.register_step_generator(s.generate_steps)
            self.steppers.append(s)

    def get_steppers(self):
        return self.steppers
    def calc_position(self, stepper_positions):

        # inversion of kin_inverse.c
        
        # bed rotation angle
        bed_angle = stepper_positions['b']
        
        # shoulder rotation angle relative to x,y plane
        l1_angle = stepper_positions['s']
        
        # arm rotation angle relative to shoulder
        l2_angle = stepper_positions['a']
        
        # convert to cartesian x, y, z
        l1_angle = math.radians(self.angle1)+l1_angle
        l2_angle = math.radians(self.angle2)+l2_angle
        r = self.l0+self.l1*math.cos(l1_angle)+self.l2*math.cos(l1_angle+l2_angle)
        x = r*math.sin(bed_angle)
        y = r*math.cos(bed_angle)
        z = self.l1*math.sin(l1_angle)+self.l2*math.sin(l1_angle+l2_angle)

        return [x, y, z]
    
    def set_position(self, newpos, homing_axes):
        for s in self.steppers:
            s.set_position(newpos)
    def home(self, homing_state):
        pass
    def check_move(self, move):
        # check move
        end_pos = move.end_pos
        if self.r <= math.sqrt(end_pos[0]**2+end_pos[1]**2+end_pos[2]**2):
            raise move.move_error("out of bound")
    def get_status(self, eventtime):
        pass

def load_kinematics(toolhead, config):
    return InverseKinematics(toolhead, config)
