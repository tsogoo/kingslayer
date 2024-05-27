from enum import Enum
import requests
import socket
import os

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
        # self.send_all(self.startup_code())

    def move(self, moves):
        for move in moves:
            gcodes = self.move_code(move)
            self.send_all(gcodes)
        self.send_all(self.after_move_code())

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

        # print(gcode)
        return gcode

    def after_move_code(self):
        gcode = []
        gcode.append("G1 X0 Y250 Z40")
        return gcode
    
    def startup_code(self):
        return ["G28"]
    
    def send(self, gcode):
        self.apiHandler.command(RobotApiCommand.Command, gcode)
    
    def send_all(self, gcodes):
        for gcode in gcodes:
            self.send(gcode)


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
        self.send_to_octoprint = True
        if self.send_to_octoprint:
            self.octoprint_url = ""
        # self.command(RobotApiCommand.Connect)

    # send commands to octoprint
    def command(self, command, data=None):
        
        if not self.send_to_octoprint:
            # send command directly to klipper
            if command == RobotApiCommand.Command:
                socket_path = '/tmp/klippy_uds'
                client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                client.connect(socket_path)
                message = '{"id": 1, "method": "gcode/script", "params": {"script": "%s"}}' % data
                client.sendall(("%s\x03" % (message)).encode())
                # response = client.recv(1024)
                # print(f'Received response: {response.decode()}')
                client.close()
        else:
            # send command to octoprint
            if command == RobotApiCommand.Command:
                # check printer is online
                # send command
                # subscribe command response
                
                baseurl = self.octoprint_url
                headers = {
                    'Content-Type': 'application/json',
                    'X-Api-Key': 'F51BE844EAE241D886CDF3A9224EB179'
                }

                # check printer status
                attempt = 5
                response = requests.get(baseurl+'/connection', headers=headers)
                if response.status_code == 200:
                    if response.json()['current']['state'] == "Operational":
                        attempt = 0
                
                # connect to printer
                while attempt > 0:
                    data = {
                        "command": "connect",
                        "port": "/tmp/printer"
                    }
                    response = requests.post(baseurl+'/connection', json=data, headers=headers)
                    if response.status_code == 204:
                        break
                    attempt = attempt-1
                    if attempt == 0:
                        raise Exception("PrinterConnectionError")

                # send command
                data = {
                    "command": data
                }
                response = requests.post(baseurl+'/printer/command', json=data, headers=headers)
                if response.status_code == 204:
                    return
                raise Exception("PrinterCommandError")

    # e = chess_engine_helper, m = main/kingslayer/
    def move(self, e, m, best_move):

        moves = []
        
        _from, _to = e.get_position(best_move)
        xs, ys = _from[0], _from[1]
        xd, yd = _to[0], _to[1]
        
        # source
        x_s, y_s = m.get_figure_actual_position(xs, ys, True)
        
        # destination
        is_occupied = e.is_occupied(best_move)
        x_d, y_d = m.get_figure_actual_position(xd, yd, is_occupied)

        # king castle
        castling, king_castling = e.is_castling(best_move)
        
        # king castling move
        if castling:
            # king
            moves.append(RobotMove(RobotTask.Take, x_s, y_s))
            moves.append(RobotMove(RobotTask.Place, x_d, y_d))
            
            # rook
            x_s, y_s = m.get_figure_actual_position(xs+3 if king_castling else xs-4, ys, True)
            x_d, y_d = m.get_figure_actual_position(xs+1 if king_castling else xs-1, ys, True)
            moves.append(RobotMove(RobotTask.Take, x_s, y_s))
            moves.append(RobotMove(RobotTask.Place, x_d, y_d))
        else:
            if is_occupied:
                moves.append(RobotMove(RobotTask.Take, x_d, y_d))
                moves.append(RobotMove(RobotTask.Out))
            
            is_promotion = e.is_promotion(best_move)
            # promotion move, pawn become queen
            if is_promotion:
                # pawn
                moves.append(RobotMove(RobotTask.Take, x_s, y_s))
                moves.append(RobotMove(RobotTask.Out))
                
                # queen
                moves.append(RobotMove(RobotTask.In))
                moves.append(RobotMove(RobotTask.Place, x_d, y_d))
            else:
                # normal move
                moves.append(RobotMove(RobotTask.Take, x_s, y_s))
                moves.append(RobotMove(RobotTask.Place, x_d, y_d))

        self.robot_handler.move(moves)
    

# r = RobotApiHandler()
# r.command(command=RobotApiCommand.Command, data="G28")
# r.command(command=RobotApiCommand.Command, data="G1 X20")