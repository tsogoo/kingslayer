from enum import Enum
import requests
import socket
import os
import logging
import time
import paho.mqtt.client as mqtt
import yaml
import threading

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
        self.robotApiHandler = RobotApiHandler()

    def move_handle(self, moves):
        for move in moves:
            gcodes = self.move_code(move)
            self.commands_handle(gcodes)
        self.commands_handle(self.after_move_code())

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

        return gcode

    def after_move_code(self):
        gcode = []
        gcode.append("G1 X0 Y250 Z40")
        return gcode
    
    def startup_code(self):
        return ["G28"]
    
    def command_handle(self, gcode):
        self.robotApiHandler.command(RobotApiCommand.Command, gcode)
    
    def commands_handle(self, gcodes):
        for gcode in gcodes:
            self.command_handle(gcode)

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

        self.move_handle(moves)


class RobotHandler:

    def __init__(self):
        self.robot = Robot()
    
    def move(self, moves):
        self.robot.move(moves)


class RobotApiCommand(Enum):
    Command = "Command"


class RobotApiHandler:

    def __init__(self, config=None):
        
        if not config:
            with open('app.yaml', 'r') as file:
                config = yaml.safe_load(file)

        self.klipperApiHandler = KlipperApiHandler(config)

        if config['broker'] and config['broker']['enabled']:
            self.conf = config['broker']
            client = mqtt.Client()
            client.on_connect = self.on_connect
            client.on_message = self.on_message
            client.connect(self.conf['url'], self.conf['port'], 60)
            client.loop_start()

            self.client = client


    def command(self, command, data=None):
        if command == RobotApiCommand.Command:
            self.klipperApiHandler.request(data)
            
    def on_connect(self, client, userdata, flags, rc):
        client.subscribe(self.conf['topic']['request'])

    def on_message(self, client, userdata, msg):
        print(f"Topic: {msg.topic}\nMessage: {msg.payload.decode()}")


class KlipperApiHandler:
    
    def __init__(self, config: dict):
        
        self.requests = []
        self.key = 1
        self.lock = False
        self.r = ''

        # Create a Unix domain socket
        client_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.client_socket = client_socket

        # Connect to the server
        client_socket.connect(config['klipper']['socket'])

        # request, response thread
        queue_thread = threading.Thread(target=self.request_handle)
        queue_thread.start()
        response_thread = threading.Thread(target=self.response_handle)
        response_thread.start()

    def request(self, command: str):
        self.requests.append({
            "k": self.key,
            "c": command
        })
        self.key = self.key + 1
        if self.key == 10000000:
            self.key = 1

    def request_handle(self):
        client_socket = self.client_socket
        
        self.subscribe_output()

        while True:
            time.sleep(1)
            if self.lock or len(self.requests) == 0:
                continue
            try:
                self.r = self.requests[0]
                self.run_code()
                self.requests.pop(0)
                self.lock = True
            except Exception as e:
                logging.getLogger(__name__).exception("request_error: %s", e)


    def response_handle(self):
        client_socket = self.client_socket
        
        while True:
            time.sleep(1)
            data = client_socket.recv(1024)
            if not data:
                logging.getLogger(__name__).error("socket_closed")
            else:
                msg = data.decode()
                print(f"Response: {msg}")
                print(f"Request: {self.r}")
                if 'G1' in self.r or 'SERVO' in self.r:
                    if 'toolhead:move' in msg:
                        self.lock = False
                else:
                    if 'result' in msg:
                        self.lock = False
                
        
    def run_code(self):
        r = self.r
        message = '{"id":%d,"method":"gcode/script","params":{"script":"%s"}}' % (r['k'], r['c'])
        self.client_socket.sendall(("%s\x03" % (message)).encode())

    def subscribe_output(self):
        message = '{"id":%d,"method":"gcode/subscribe_output","params":{"response_template":{"key":%d}}}' % (1, 1)
        self.client_socket.sendall(("%s\x03" % (message)).encode())

# r = RobotApiHandler()
# r.command(RobotApiCommand.Command, "G28")
# r.command(RobotApiCommand.Command, "G1 Y360 Z0 F5000")
# r.command(RobotApiCommand.Command, "SET_SERVO servo=servo_arm angle=90")
# r.command(RobotApiCommand.Command, "SET_SERVO servo=servo_arm angle=120")
# r.command(RobotApiCommand.Command, "G1 Y300 Z0 F5000")
# r.command(RobotApiCommand.Command, "SET_SERVO servo=servo_arm angle=90")
# r.command(RobotApiCommand.Command, "SET_SERVO servo=servo_arm angle=120")
# r.command(RobotApiCommand.Command, "G28")
# r.command(RobotApiCommand.Command, "M84")