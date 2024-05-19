# Code for handling the custom kinematics
#
# Copyright (C) 2024  Erdene Luvsandorj <erdene.lu.n@gmail.com>
#
# This file may be distributed under the terms of the GNU GPLv3 license.

import math
import stepper
import logging

class Custom2Kinematics:
    def __init__(self, toolhead, config):
        
        # printer config
        self.l0 = config.getfloat('l0', above=0.)
        self.l1 = config.getfloat('l1', above=0.)
        self.l2 = config.getfloat('l2', above=0.)
        self.a1 = config.getfloat('a1', above=0.)
        self.a2 = config.getfloat('a2', above=0.)
        
        # future use
        self.printer = config.get_printer()
        
        # Setup steppers
        self.rails = []
        for type in 'bsa':
            r = stepper.LookupMultiRail(config.getsection('stepper_'+type), units_in_radians=True)
            self.rails.append(r)
        self.steppers = [s for rail in self.rails for s in rail.get_steppers()]
        for s in self.get_steppers():
            s.setup_itersolve('custom_stepper_alloc', type.encode()
                , self.l0, self.l1, self.l2)
            s.set_trapq(toolhead.get_trapq())
            toolhead.register_step_generator(s.generate_steps)

    def get_steppers(self):
        return self.steppers
    def calc_position(self, stepper_positions):
        pass
    def set_position(self, newpos, homing_axes):
        pass
    def home(self, homing_state):
        
        # force move stepper without kinematic
        force_move = self.printer.lookup_object('force_move')
        toolhead = self.printer.lookup_object('toolhead')
        dir = 1
        for rail in self.rails:
            # ignore bed homing
            if rail.get_name(True) == 'b':
                continue
            if rail.get_name(True) == 'a':
                dir = -1
            else:
                dir = 1
            for stepper in rail.get_steppers():
                move_end_print_time = toolhead.get_last_move_time()
                triggered = 0
                for mcu_endstop, n in rail.get_endstops():
                    while 1:
                        triggered = mcu_endstop.query_endstop(move_end_print_time)
                        if triggered:
                            break
                        force_move.manual_move(stepper, .1*dir, 200, 200)
                        move_end_print_time = toolhead.get_last_move_time()
        # set kinematic position
        curpos = toolhead.get_position()
        curpos[0] = 0
        curpos[1] = (
            self.l0
            +self.l1*math.cos(math.radians(self.a1))
            +self.l2*math.cos(math.radians(self.a2))
        )
        curpos[2] = (
            self.l1*math.sin(math.radians(self.a1))
            -self.l2*math.sin(math.radians(self.a2))
        )
        toolhead.set_position(curpos, homing_axes=(0, 1, 2))

    def check_move(self, move):
        pass
    def get_status(self, eventtime):
        pass

def load_kinematics(toolhead, config):
    return Custom2Kinematics(toolhead, config)