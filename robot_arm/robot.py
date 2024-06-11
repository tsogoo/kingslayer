from enum import Enum
import socket
import time
import paho.mqtt.client as mqtt
import yaml
import threading
import json
import os

class RobotTask(Enum):
    Take = "Take"   # take figure from board
    Place = "Place" # place figure on board
    Out = "Out"     # same as place but place outside of board
    Move = "Move"   # default
    In = "In"       # same as take but take from outside of board

class RobotMove:
    
    def __init__(self, task=RobotTask.Move, x=0, y=0, z=0, xy_speed=0, z_speed=0):
        self.task = task    # move, eat
        self.x = x
        self.y = y
        self.z = z
        self.xy_speed = xy_speed
        self.z_speed = z_speed

class Robot:

    CMD_SERVO_FMT = "SET_SERVO servo=servo_arm angle={}"

    def __init__(self, config):
        self.config = config
        self.robotApiHandler = RobotApiHandler(config=config)

    def move_handle(self, moves):
        self.commands_handle(self.before_move_code())
        for move in moves:
            gcodes = self.move_code(move)
            self.commands_handle(gcodes)
        self.commands_handle(self.timer_code())
        self.commands_handle(self.after_move_code())

    def move_code(self, move):
        # Queen position
        if move.task == RobotTask.In:
            move.x = self.config['in_x']
            move.y = self.config['in_y']
        if move.task == RobotTask.Out:
            move.x = self.config['out_x']
            move.y = self.config['out_y']
        
        gcode = []
        gcode.append("G1 Z{} F{}".format(self.config['travel_z'], self.xy_speed(move)))
        gcode.append("G1 X{} Y{} F{}".format(move.x, move.y, self.xy_speed(move)))
        gcode.append("G1 Z{} F{}".format(self.config['take_z'], self.z_speed(move)))
        if move.task in (RobotTask.Place, RobotTask.Out):
            gcode.append(Robot.CMD_SERVO_FMT.format(self.config['release_angle']))
        else:
            gcode.append(Robot.CMD_SERVO_FMT.format(self.config['take_angle']))
        gcode.append("G1 Z{} F{}".format(self.config['travel_z'], self.z_speed(move)))

        return gcode

    def after_move_code(self):
        # home and disable motors
        return [
            "G1 X{}".format(0-self.config['offset_x']),
            "G28",
            "M84",
            "SET_SERVO servo=servo_arm width=0"
        ]
    
    def before_move_code(self):
        return [
            "G28",
            "SET_GCODE_OFFSET X={} Y={}".format(self.config['offset_x'], self.config['offset_y']), # can execute before move
            Robot.CMD_SERVO_FMT.format(self.config['release_angle'])
        ]
    
    def timer_code(self):
        # push down timer
        return [
            "G1 Z{} F{}".format(self.config['travel_z'], self.config['z_speed']),
            "G1 X{} Y{} F{}".format(self.config['timer_x'], self.config['timer_y'], self.config['xy_speed']),
            Robot.CMD_SERVO_FMT.format(self.config['take_angle']),
            "G1 Z{} F{}".format(self.config['take_z'], self.config['z_speed']),
            "G1 Z{} F{}".format(self.config['travel_z'], self.config['z_speed']),
        ]
    
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

    def xy_speed(self, move: RobotMove):
        return move.xy_speed if move.xy_speed > 0 else self.config['xy_speed']
    
    def z_speed(self, move: RobotMove):
        return move.z_speed if move.z_speed > 0 else self.config['z_speed']

class RobotHandler:

    def __init__(self):
        self.robot = Robot()
    
    def move(self, moves):
        self.robot.move(moves)


class RobotApiCommand(Enum):
    Command = "Command"


class RobotApiHandler:

    def __init__(self, config: dict=None):

        self.klipperApiHandler = KlipperApiHandler(
            config=config,
            on_response=self.on_response
        )

        if 'broker' in config and 'enabled' in config['broker'] and config['broker']['enabled']:
            self.conf = config['broker']
            client = mqtt.Client()
            self.client = client
            client.on_connect = self.on_connect
            client.on_message = self.on_message
            client.connect(self.conf['url'], self.conf['port'], 60)
            client.loop_start()


    def command(self, command, data: str=None):
        if command == RobotApiCommand.Command:
            self.klipperApiHandler.request(data)
            
    def on_connect(self, client, userdata, flags, rc):
        client.subscribe(self.conf['topic']['request'])

    def on_message(self, client, userdata, msg):
        self.command(RobotApiCommand.Command, msg.payload.decode())

    def on_response(self, msg: str):
        if self.client:
            self.client.publish(topic=self.conf['topic']['response'], payload=msg)


class KlipperApiHandler:
    
    def __init__(self, config: dict, on_response=None):
        
        self.config = config
        self.requests = []
        self.key = 1
        self.lock = False   # prevent request to send without previous completed
        self.r = ''
        self.on_response = on_response
        
        # request, response thread
        self.request_thread = None
        self.response_thread = None

        self._connect(immediate=True)

    def _connect(self, immediate:bool=False):
        try:
            if not immediate:
                time.sleep(5)
            client_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self.client_socket = client_socket
            self.client_socket.connect(self.config['klipper']['socket'])
            self.lock = False
            self.subscribe_output()
        except Exception as e:
            pass

    def _get_socket(self):
        return self.client_socket

    def request(self, command: str):
        self.requests.append({
            "k": self.key,
            "c": command
        })
        self.key = self.key + 1
        if self.key == 10000000:
            self.key = 1
        
        if not self.request_thread:
            request_thread = threading.Thread(target=self.request_handle)
            self.request_thread = request_thread
            request_thread.start()
        if not self.response_thread:
            response_thread = threading.Thread(target=self.response_handle)
            self.response_thread = response_thread
            response_thread.start()

    def request_handle(self):

        while True:
            time.sleep(0.1)
            if self.lock or len(self.requests) == 0:
                continue
            try:
                self.r = self.requests[0]
                self.run_code()
                self.requests.pop(0)
                self.lock = True
            except OSError as e:
                self._connect()
            except Exception as e:
                pass


    def response_handle(self):
        while True:
            time.sleep(0.1)
            
            try:
                data = self._get_socket().recv(1024)
                if data:
                    responses = data.split(b'\x03')
                    for response in responses:
                        self._response_handle(response)
                else:
                    self._connect()
            except Exception as e:
                pass

    def _response_handle(self, response):
        msg = response.decode()
        
        try:
            res = json.loads(msg)
        except ValueError as e:
            res = {}

        if 'error' in res:
            msg = res['error']['message']
            self.lock = False
        else:
            if 'G1' in self.r or 'SERVO' in self.r:
                if 'toolhead:move' in msg:
                    self.lock = False
            else:
                if 'result' in res:
                    self.lock = False

        if not self.lock and self.on_response:
            self.on_response(msg) 

    
    fmt_c = '''{
        "id":%d,
        "method":"gcode/script",
        "params":{
            "script":"%s"
        }
    }'''
    def run_code(self):
        r = self.r
        message = KlipperApiHandler.fmt_c % (r['k'], r['c'])
        self.client_socket.sendall(("%s\x03" % (message)).encode())
    
    fmt_o = '''{
        "id":%d,
        "method":
        "gcode/subscribe_output",
        "params":{
            "response_template":{"key":%d}
        }
    }'''
    def subscribe_output(self):
        message = KlipperApiHandler.fmt_o % (1, 1)
        self.client_socket.sendall(("%s\x03" % (message)).encode())

# sample
# r = RobotApiHandler()
# r.command(RobotApiCommand.Command, "G28")
# r.command(RobotApiCommand.Command, "G1 Z20 F5000")
# r.command(RobotApiCommand.Command, "G1 Y360 F5000")
# r.command(RobotApiCommand.Command, "SET_SERVO servo=servo_arm angle=80")
# r.command(RobotApiCommand.Command, "G1 Z-100 F5000")
# r.command(RobotApiCommand.Command, "SET_SERVO servo=servo_arm angle=125")
# r.command(RobotApiCommand.Command, "G1 Z20 F5000")
# r.command(RobotApiCommand.Command, "G1 Y300 F5000")
# r.command(RobotApiCommand.Command, "G1 Z-100 F5000")
# r.command(RobotApiCommand.Command, "SET_SERVO servo=servo_arm angle=80")
# r.command(RobotApiCommand.Command, "G1 Z20 F5000")
# r.command(RobotApiCommand.Command, "SET_SERVO servo=servo_arm angle=125")
# r.command(RobotApiCommand.Command, "G28")
# r.command(RobotApiCommand.Command, "M84")