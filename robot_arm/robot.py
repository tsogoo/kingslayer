from enum import Enum
import os

class RobotTask(Enum):
    Take = "Take"   # take figure from board
    Place = "Place" # place figure on board
    Out = "Out"     # same as place but place outside of board
    Move = "Move"   # default
    In = "In"       # same as take but take from outside of board

class RobotMove:
    
    def __init__(self, task=RobotTask.Move, x=0, y=0, z=40, speed=1000, up_down_speed=1000):
        self.task = task    # move, eat
        self.x = x
        self.y = y
        self.z = z
        self.speed = speed
        self.up_down_speed = up_down_speed

class Robot:

    def __init__(self):
        # connection
        self.echos(self.startup_code())

    def move(self, moves):
        for move in moves:
            gcodes = self.move_code(move)
            self.echos(gcodes)
        self.echos(self.after_move_code())

    def move_code(self, move):
        # Queen position
        if move.task == RobotTask.In:
            move.x = 380
            move.y = 200
        if move.task == RobotTask.Out:
            move.x = 380
            move.y = 0
        
        gcode = []
        gcode.append("G1 Z"+str(move.z)+" F"+str(move.up_down_speed))
        gcode.append("G1 X"+str(move.x)+" Y"+str(move.y)+" F"+str(move.speed))
        
        gcode.append("G1 Z-20 F"+str(move.up_down_speed))   # take or place height
        if move.task in (RobotTask.Place, RobotTask.Out):
            gcode.append("SET_SERVO servo=servo_arm angle=0")
        else:
            gcode.append("SET_SERVO servo=servo_arm angle=120")
        gcode.append("G1 Z"+str(move.z)+" F"+str(move.up_down_speed))

        print(gcode)
        return gcode

    def after_move_code(self):
        gcode = []
        gcode.append("G1 X0 Y210 Z40.19")
        gcode.append("G28")
        return gcode
    
    def startup_code(self):
        return ["G28"]
    
    def echo(self, gcode):
        # TODO echo
        # cmd = 'echo {}'.format(gcode)
        # os.system(cmd)
        pass
    
    def echos(self, gcodes):
        for gcode in gcodes:
            self.echo(gcode)


class RobotHandler:

    def __init__(self):
        self.robot = Robot()
    
    def move(self, moves):
        self.robot.move(moves)