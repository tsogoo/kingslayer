from enum import Enum
import socket
import time
import paho.mqtt.client as mqtt
import yaml
import threading
import json
import os

def get_config(conf:dict, config:str):
    confs = config.split(":")
    v = conf
    if len(confs) > 0:
        for conf in confs:
            if conf in v:
                v = v[conf]
            else:
                return None
    return v

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
        self.config = get_config(config, 'robot')
        self.robotApiHandler = RobotApiHandler(config=config)

    def move_handle(self, moves, turn):
        self.commands_handle(self.before_move_code())
        for move in moves:
            gcodes = self.move_code(move)
            self.commands_handle(gcodes)
        self.commands_handle(self.timer_code(turn))
        self.commands_handle(self.after_move_code())

    def move_code(self, move):
        # Queen position
        if move.task == RobotTask.In:
            move.x = get_config(self.config, 'in:x')
            move.y = get_config(self.config, 'in:y')
        if move.task == RobotTask.Out:
            move.x = get_config(self.config, 'out:x')
            move.y = get_config(self.config, 'out:y')
        
        is_take = move.task not in (RobotTask.Place, RobotTask.Out)
        gcode = []
        gcode.append(
            "G1 Z{} F{}".format(
                get_config(self.config, 'board:safe_z'), self.z_speed(move)
            )
        )
        gcode.append(
            "G1 X{} Y{} F{}".format(
                move.x, move.y, self.xy_speed(move)
            )
        )
        gcode.append(
            "G1 Z{} F{}".format(
                get_config(self.config, 'board:z'), self.z_speed(move, is_take)
            )
        )
        if not is_take:
            gcode.append(
                Robot.CMD_SERVO_FMT.format(
                    get_config(self.config, 'gripper:release_angle')
                )
            )
        else:
            gcode.append(
                Robot.CMD_SERVO_FMT.format(
                    get_config(self.config, 'gripper:take_angle')
                )
            )
        gcode.append(
            "G1 Z{} F{}".format(
                get_config(self.config, 'board:safe_z'),
                self.z_speed(move, not is_take)
            )
        )

        return gcode

    def after_move_code(self):
        # home and disable motors
        return [
            "G1 X{}".format(0 - get_config(self.config, 'board:x')),
            "G28",
            "M84",
            "SET_SERVO servo=servo_arm width=0"
        ]
    
    def before_move_code(self):
        return [
            "G28",
            "SET_GCODE_OFFSET X={} Y={}".format(
                get_config(self.config, 'board:x'),
                get_config(self.config, 'board:y')
            ), # can execute before move
            Robot.CMD_SERVO_FMT.format(
                get_config(self.config, 'gripper:release_angle')
            )
        ]
    
    def timer_code(self, turn):
        print(turn)
        # push down timer
        return [
            "G1 Z{} F{}".format(
                get_config(self.config, 'board:safe_z'),
                get_config(self.config, 'speed:z_slow')
            ),
            Robot.CMD_SERVO_FMT.format(
                get_config(self.config, 'gripper:take_angle')
            ),
            "G1 X{} Y{} F{}".format(
                get_config(self.config, 'timer:x'),
                get_config(self.config, 'timer:y' if turn else 'timer:y_b'),
                self.xy_speed()
            ),
            "G1 Z{} F{}".format(
                get_config(self.config, 'timer:z'),
                self.z_speed()
            ),
            "G1 Z{} F{}".format(
                get_config(self.config, 'board:safe_z'),
                self.z_speed()
            ),
        ]
    
    def command_handle(self, gcode):
        self.robotApiHandler.command(RobotApiCommand.Command, gcode)
    
    def commands_handle(self, gcodes):
        for gcode in gcodes:
            self.command_handle(gcode)

    # e = chess_engine_helper, m = main/kingslayer/
    def move(self, e, m, best_move, turn):

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

        self.move_handle(moves, turn)

    def xy_speed(self, move: RobotMove=None):
        if move and move.xy_speed > 0:
            return move.xy_speed
        return get_config(self.config, 'speed:xy')
    
    def z_speed(self, move: RobotMove=None, is_slow:bool=False):
        if is_slow:
            return get_config(self.config, 'speed:z_slow')
        else:
            if move and move.z_speed > 0:
                return move.z_speed
            return get_config(self.config, 'speed:z')
        
    

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
            config=get_config(config, 'klipper'),
            on_response=self.on_response
        )

        if get_config(config, 'broker:enabled'):
            self.config = get_config(config, 'broker')
            client = mqtt.Client()
            self.client = client
            client.on_connect = self.on_connect
            client.on_message = self.on_message
            client.connect(get_config(self.config, 'url'), get_config(self.config, 'port'), 60)
            client.loop_start()


    def command(self, command, data: str=None):
        if command == RobotApiCommand.Command:
            self.klipperApiHandler.request(data)
            
    def on_connect(self, client, userdata, flags, rc):
        client.subscribe(get_config(self.config, 'topic:request'))

    def on_message(self, client, userdata, msg):
        self.command(RobotApiCommand.Command, msg.payload.decode())

    def on_response(self, msg: str):
        if self.client:
            self.client.publish(topic=get_config(self.config, 'topic:response'), payload=msg)


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
            self.client_socket.connect(get_config(self.config, 'socket'))
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