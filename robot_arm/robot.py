from enum import Enum
import requests
import math

class RobotTask(Enum):
    Take = "Take"   # take figure from board
    Place = "Place" # place figure on board
    Out = "Out"     # same as place but place outside of board
    Move = "Move"   # default
    In = "In"       # same as take but take from outside of board

class RobotMove:
    
    def __init__(self, task=RobotTask.Move, x=0, y=0, z=40, speed=10000, up_down_speed=5000):
        self.task = task    # move, eat
        self.x = x
        self.y = y
        self.z = z
        self.speed = speed
        self.up_down_speed = up_down_speed

class Robot:

    def __init__(self):
        self.apiHandler = RobotApiHandler()
        self.l0 = 60
        self.l1 = 300
        self.l2 = 300
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
        gcode.append("G1 Y350 Z{} F{}".format(move.z, move.up_down_speed))
        gcode.append("G1 X{} Y{} F{}".format(move.x, move.y, move.speed))
        
        gcode.append("G1 Z-20 F{}".format(move.up_down_speed))   # take or place height
        if move.task in (RobotTask.Place, RobotTask.Out):
            gcode.append("SET_SERVO servo=servo_arm angle=0")
        else:
            gcode.append("SET_SERVO servo=servo_arm angle=120")
        gcode.append("G1 Z{} F{}".format(move.z, move.up_down_speed))

        print(gcode)
        return gcode

    def after_move_code(self):
        gcode = []
        gcode.append("G1 X0 Y250 Z40")
        return gcode
    
    def startup_code(self):
        return ["G28"]
    
    def echo(self, gcode):
        self.apiHandler.command(RobotApiCommand.Command, gcode)
    
    def echos(self, gcodes):
        for gcode in gcodes:
            self.echo(gcode)


class RobotHandler:

    def __init__(self):
        self.robot = Robot()
    
    def move(self, moves):
        self.robot.move(moves)


class RobotApiCommand(Enum):
    Connect = "Connect"
    Status = "Status"
    Command = "Command"


class RobotApiHandler:

    def __init__(self):
        pass
        # self.command(RobotApiCommand.Connect)

    # send commands to octoprint
    def command(self, command, data=None):
        
        baseurl = 'http://192.168.1.2:5000/api'
        headers = {
            'Content-Type': 'application/json',
            'X-Api-Key': 'F51BE844EAE241D886CDF3A9224EB179'
        }

        # check octoprint status
        if command == RobotApiCommand.Status:
            response = requests.get(baseurl+'/connection', headers=headers)
            if response.status_code == 200:
                if response.json()['current']['state'] == "Operational":
                    return
            raise Exception("PrinterNotReady")

        if command == RobotApiCommand.Connect:
            # connect
            data = {
                "command": "connect",
                "port": "/tmp/printer"
            }
            response = requests.post(baseurl+'/connection', json=data, headers=headers)
            if response.status_code == 204:
                return
            raise Exception("PrinterConnectionError")
            
        if command == RobotApiCommand.Command:
            # command
            data = {
                "command": data
            }
            response = requests.post(baseurl+'/printer/command', json=data, headers=headers)
            if response.status_code == 204:
                return
            raise Exception("PrinterCommandError")