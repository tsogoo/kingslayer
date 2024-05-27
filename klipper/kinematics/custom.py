# Code for handling the custom kinematics
#
# Copyright (C) 2024  Erdene Luvsandorj <erdene.lu.n@gmail.com>
#
# This file may be distributed under the terms of the GNU GPLv3 license.

import math
import stepper
import logging

class CustomKinematics:
    def __init__(self, toolhead, config):
        
        # printer config
        self.l0 = config.getfloat('l0', above=0.)
        self.l1 = config.getfloat('l1', above=0.)
        self.l2 = config.getfloat('l2', above=0.)

        # Setup steppers
        self.steppers = []
        for type in 'sab':
            s = stepper.PrinterStepper(config.getsection('stepper_'+type), True)
            s.setup_itersolve('custom_stepper_alloc', type.encode()
                , self.l0, self.l1, self.l2)
            s.set_trapq(toolhead.get_trapq())
            toolhead.register_step_generator(s.generate_steps)
            self.steppers.append(s)

    def get_steppers(self):
        return self.steppers
    def calc_position(self, stepper_positions):
        pass
        # # bed rotation angle
        # bed_angle = stepper_positions['b']
        
        # # shoulder rotation angle relative to x,y plane
        # l1_angle = stepper_positions['s']
        
        # # arm rotation angle relative to shoulder
        # l2_angle = stepper_positions['a']
        
        # # l1_angle = self.angle1+l1_angle
        # # l2_angle = self.angle2+l2_angle

        # return self.get_pos(bed_angle, l1_angle, l2_angle)
    
    def set_position(self, newpos, homing_axes):
        pass
        # for s in self.steppers:
        #     s.set_position(newpos)
    def home(self, homing_state):
        pass
    def check_move(self, move):
        pass
        # # check move
        # end_pos = move.end_pos
        # x = end_pos[0]
        # y = end_pos[1]
        # z = end_pos[2]
        # r = math.sqrt(x*x+y*y)-self.l0
        
        # # arm
        # cos_b = (r*r+z*z-self.l1*self.l1-self.l2*self.l2)/(2*self.l1*self.l2)
        # if cos_b > 1 or cos_b < -1:
        #     raise move.move_error("arm: coord limit")
        # b = math.degrees(math.acos(cos_b))        
        # if b > 150:
        #     raise move.move_error("arm: bend max limit /down/")
        # if b < 0:
        #     raise move.move_error("arm: bend min limit /up/")
        
        # # shoulder
        # sin_b = math.sqrt(1 - cos_b*cos_b)
        # d = math.sqrt(r*r+z*z)
        # if z > d:
        #     raise move.move_error("shoulder: z max limit")
        # a = math.degrees(math.asin(self.l2*sin_b/d)+math.asin(z/d))
        # if a > 90:
        #     raise move.move_error("shoulder: bend max limit /up/")
        # if a < 10:
        #     raise move.move_error("shoulder: bend min limit /down/")
    def get_status(self, eventtime):
        pass

def load_kinematics(toolhead, config):
    return CustomKinematics(toolhead, config)