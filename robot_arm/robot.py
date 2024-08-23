from enum import Enum
import socket
import time
import paho.mqtt.client as mqtt
import threading
import json
import math
from common.config import get_config
import requests


class RobotTask(Enum):
    Take = "Take"  # take figure from board
    Place = "Place"  # place figure on board opposite of take
    Out = "Out"  # same as place but place outside of board
    Move = "Move"  # default
    In = "In"  # same as take but take from outside of board
    Buzzer = "Buzzer"


class RobotMove:

    def __init__(self, task=RobotTask.Move, x=0, y=0):
        self.task = task  # move, eat
        self.x = x
        self.y = y


class Robot:

    def __init__(self, config):
        self.config = get_config(config, "robot")
        self.robotApiHandler = RobotApiHandler(config=config)
        self.board_square_size = get_config(self.config, "board:square_size")
        self.board_margin_size = get_config(self.config, "board:margin_size")
        self.commands = []

    def calibrate_board(self):
        self.commands = [
            "G28",
            # "SET_GCODE_OFFSET X={} Y={}".format(
            #     get_config(self.config, 'board:x'),
            #     get_config(self.config, 'board:y')
            # ),
            "G1 X{} Y{} Z{} F{}".format(
                (
                    int(get_config(self.config, "board:x"))
                    - int(get_config(self.config, "board:margin_size"))
                ),
                (
                    int(get_config(self.config, "board:y"))
                    - int(get_config(self.config, "board:margin_size"))
                ),
                get_config(self.config, "board:figure_z"),
                self.xy_speed(),
            ),
            "G1 X{} Y{} Z{} F{}".format(
                (
                    int(get_config(self.config, "board:x"))
                    + int(get_config(self.config, "board:margin_size"))
                    + 8 * int(get_config(self.config, "board:square_size"))
                ),
                (
                    int(get_config(self.config, "board:y"))
                    - int(get_config(self.config, "board:margin_size"))
                ),
                get_config(self.config, "board:figure_z"),
                self.xy_speed(),
            ),
            "G1 X{} Y{} Z{} F{}".format(
                (
                    int(get_config(self.config, "board:x"))
                    + int(get_config(self.config, "board:margin_size"))
                    + 8 * int(get_config(self.config, "board:square_size"))
                ),
                (
                    int(get_config(self.config, "board:y"))
                    + int(get_config(self.config, "board:margin_size"))
                    + 8 * int(get_config(self.config, "board:square_size"))
                ),
                get_config(self.config, "board:figure_z"),
                self.xy_speed(),
            ),
            "G1 X{} Y{} Z{} F{}".format(
                (
                    int(get_config(self.config, "board:x"))
                    - int(get_config(self.config, "board:margin_size"))
                ),
                (
                    int(get_config(self.config, "board:y"))
                    + int(get_config(self.config, "board:margin_size"))
                    + 8 * int(get_config(self.config, "board:square_size"))
                ),
                get_config(self.config, "board:figure_z"),
                self.xy_speed(),
            ),
            "G1 X{} Y{} Z{} F{}".format(
                (
                    int(get_config(self.config, "board:x"))
                    - int(get_config(self.config, "board:margin_size"))
                ),
                (
                    int(get_config(self.config, "board:y"))
                    - int(get_config(self.config, "board:margin_size"))
                ),
                get_config(self.config, "board:figure_z"),
                self.xy_speed(),
            ),
            "G1 z{}".format(get_config(self.config, "board:safe_z")),
            "G1 X{}".format(0),
            "G28",
            "M84",
        ]
        print(get_config(self.config, "urls:klipper"))
        print(self.commands)

        requests.post(
            get_config(self.config, "urls:klipper"), json={"commands": self.commands}
        )
        print("-======================responded")

    def move_handle(self, moves, turn):
        self.commands_handle(self.before_move_code())
        # initial point /offset calculated/
        last_move = RobotMove(x=0 - get_config(self.config, "board:x"), y=0)
        for move in moves:
            gcodes = self.move_code(move, last_move)
            last_move = move
            self.commands_handle(gcodes)
        if get_config(self.config, "timer:enabled"):
            move = RobotMove(
                x=get_config(self.config, "timer:x"),
                y=get_config(self.config, "timer:y" if turn else "timer:y_b"),
            )
            t = last_move
            last_move = move
            self.commands_handle(self.timer_code(turn, move, t))
        self.commands_handle(self.after_move_code(last_move))

    def move_code(self, move: RobotMove, last_move: RobotMove = None):
        # Queen position
        if move.task == RobotTask.In:
            move.x = get_config(self.config, "in:x")
            move.y = get_config(self.config, "in:y")
        if move.task == RobotTask.Out:
            move.x = get_config(self.config, "out:x")
            move.y = get_config(self.config, "out:y")

        is_take = move.task not in (RobotTask.Place, RobotTask.Out)
        gcode = []
        # go to safe z to travel if needed
        gcode.append(
            "G1 Z{} F{}".format(
                get_config(self.config, "board:safe_z"), self.z_speed(is_slow=True)
            )
        )

        # optimize move
        gcode.extend(self.optimize_move_xy(move, last_move))

        if move.task == RobotTask.Out:
            # no need to be slowly, drop it from a height
            gcode.append(
                "G1 Z{} F{}".format(
                    get_config(self.config, "board:out_z"), self.z_speed()
                )
            )
        else:
            # speed up to a figure, then slow down if needed
            gcode.append(
                "G1 Z{} F{}".format(
                    get_config(self.config, "board:figure_z"), self.z_speed()
                )
            )
            gcode.append(
                "G1 Z{} F{}".format(
                    get_config(self.config, "board:z"), self.z_speed(is_slow=is_take)
                )
            )
        gcode.extend(self.gripper_code(is_take=is_take))
        if move.task == RobotTask.Place:
            # slow down
            gcode.append(
                "G1 Z{} F{}".format(
                    get_config(self.config, "board:figure_z"),
                    self.z_speed(is_slow=True),
                )
            )
        gcode.append(
            "G1 Z{} F{}".format(get_config(self.config, "board:safe_z"), self.z_speed())
        )

        return gcode

    def perform_code_if_won(self):
        gcode = self.gripper_code(is_take=False, should_delay=False, is_check=True)
        gcode.extend(
            [
                "G4 P50",
                "G1 X{} Y{} Z{} ".format(
                    get_config(self.config, "tissue:x"),
                    get_config(self.config, "tissue:y"),
                    get_config(self.config, "board:safe_z"),
                ),
                "G1 Z{}".format(
                    get_config(self.config, "tissue:z"),
                ),
            ]
        )
        gcode.extend(self.gripper_code(is_take=True, should_delay=False, is_check=True))
        gcode.extend(
            [
                "G1 Z{}".format(
                    get_config(self.config, "board:safe_z") + 100,
                ),
                "G4 P50",
                "G1 X{} Y{} Z{} F{}".format(
                    (0 - get_config(self.config, "board:x")),
                    get_config(self.config, "board:y") + 50,
                    get_config(self.config, "board:safe_z") + 240,
                    get_config(self.config, "speed:xy"),
                ),
                "G4 P4450",
            ]
        )

        gcode.extend(
            self.gripper_code(is_take=False, should_delay=False, is_check=True)
        )
        gcode.append("G4 P1450")
        gcode.extend(self.gripper_code(is_take=True, should_delay=False, is_check=True))
        return gcode

    def after_move_code(self, last_move: RobotMove = None):
        # home and disable motors
        gcode = []
        # if True:
        #     gcode.extend(self.perform_code_if_won())
        gcode.extend(
            [
                "G1 X{}".format(0 - get_config(self.config, "board:x")),
                "G28",
                "M84",
            ]
        )
        gcode.extend(self.gripper_code(disable=True))
        return gcode

    def before_move_code(self):
        gcode = []
        gcode.extend(
            [
                "G28",
                "SET_GCODE_OFFSET X={} Y={}".format(
                    get_config(self.config, "board:x"),
                    get_config(self.config, "board:y"),
                ),  # can execute before move
            ]
        )
        # check gripper working
        if get_config(self.config, "gripper:check_phase"):
            gcode.append("M106 S100")
            gcode.extend(
                self.gripper_code(is_take=False, should_delay=False, is_check=True)
            )

            gcode.append("G4 P50")
            gcode.extend(
                self.gripper_code(is_take=True, should_delay=False, is_check=True)
            )
            gcode.append("G4 P50")
        gcode.extend(self.gripper_code(is_take=False, should_delay=False))
        gcode.append("M107")
        return gcode

    def timer_code(self, turn: bool, move: RobotMove, last_move: RobotMove):
        # push down timer
        gcode = []
        gcode.append(
            "G1 Z{} F{}".format(
                get_config(self.config, "board:safe_z"),
                get_config(self.config, "speed:z_slow"),
            )
        )
        gcode.extend(self.gripper_code(is_take=True, should_delay=False))
        gcode.extend(self.optimize_move_xy(move, last_move))
        gcode.extend(
            [
                "G1 Z{} F{}".format(get_config(self.config, "timer:z"), self.z_speed()),
                "G1 Z{} F{}".format(
                    get_config(self.config, "board:safe_z"), self.z_speed()
                ),
            ]
        )
        return gcode

    def gripper_code(
        self,
        is_take: bool = True,
        disable: bool = False,
        should_delay: bool = True,
        is_check: bool = False,
    ):
        gcode = []
        servo = "servo_arm"
        if disable:
            gcode.append("SET_SERVO servo={} width=0".format(servo))
            return gcode

        release_angle = get_config(self.config, "gripper:release_angle")
        take_angle = get_config(self.config, "gripper:take_angle")
        step = get_config(self.config, "gripper:step")
        delay = get_config(self.config, "gripper:delay")
        if not should_delay:
            step = 1
        if is_check:
            release_angle = release_angle - 0
            take_angle = take_angle
        if not is_take:
            start_angle = take_angle
            end_angle = release_angle
        else:
            start_angle = release_angle
            end_angle = take_angle
        step_angle = (end_angle - start_angle) * 1 / step
        for i in range(step):
            start_angle = start_angle + step_angle
            gcode.append("SET_SERVO servo={} angle={}".format(servo, start_angle))
            if step > 1:
                gcode.append("G4 P{}".format(delay))
        return gcode

    def command_handle(self, gcode):
        self.commands.append(gcode)
        self.robotApiHandler.command(RobotApiCommand.Command, gcode)

    def commands_handle(self, gcodes):
        for gcode in gcodes:
            self.command_handle(gcode)

    def empty_commands(self):
        self.commands = []

    # e = chess_engine_helper, m = main/kingslayer/
    def move(self, e, m, best_move, turn):
        self.empty_commands()

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
            x_s, y_s = m.get_figure_actual_position(
                xs + 3 if king_castling else xs - 4, ys, True
            )
            x_d, y_d = m.get_figure_actual_position(
                xs + 1 if king_castling else xs - 1, ys, True
            )
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
        requests.post("http://192.168.1.45:8000/", json={"commands": self.commands})
        # response = requests.get(
        #     "http://192.168.1.19:8000/", {"commands": self.commands}
        # )
        print("-======================responded")

    def xy_speed(self, is_slow: bool = False):
        return get_config(self.config, "speed:xy" if not is_slow else "speed:xy_slow")

    def z_speed(self, is_slow: bool = False):
        if is_slow:
            return get_config(self.config, "speed:z_slow")
        else:
            return get_config(self.config, "speed:z")

    def optimize_move_xy(self, move: RobotMove, last_move: RobotMove = None):
        gcode = []
        # optimization:ratio=10:80:10 = 100% travel whole path like that(slow_move:fast_move:slow_move)
        if get_config(self.config, "optimization:enabled"):
            y = move.y
            x = move.x
            _y = last_move.y
            _x = last_move.x
            delta_y = y - _y
            delta_x = x - _x
            is_short_distance = math.sqrt(delta_x**2 + delta_y**2) < float(
                get_config(self.config, "optimization:short_distance")
            )
            ratio = (
                get_config(
                    self.config,
                    "optimization:"
                    + ("short_distance_ratio" if is_short_distance else "ratio"),
                )
            ).split(":")
            ratio_t = 0
            for i, r in enumerate(ratio):
                ratio_t += float(r)
                gcode.append(
                    "G1 X{} Y{} F{}".format(
                        _x + delta_x * ratio_t / 100,
                        _y + delta_y * ratio_t / 100,
                        self.xy_speed(is_slow=i % 2 == 0),
                    )
                )
        else:
            gcode.append("G1 X{} Y{} F{}".format(move.x, move.y, self.xy_speed()))
        return gcode

    def task_handle(self, task: RobotTask):
        gcode = []
        if task == RobotTask.Buzzer:
            #
            # in printer.cfg add these
            # [output_pin buzzer]
            # pin: PE1 # near servo in aux1 section D1=red, GND=black, ramps1.4
            # pwm: True
            # cycle_time: 0.001
            duration = get_config(self.config, "buzzer:beep_duration")
            pwm = get_config(self.config, "buzzer:pwm")
            times = get_config(self.config, "buzzer:beep_times")
            fmt_pin_code = "SET_PIN PIN=buzzer VALUE={}"
            for t in range(times):
                gcode.extend(
                    [
                        fmt_pin_code.format(pwm),
                        "G4 P{}".format(duration),
                        fmt_pin_code.format(0),
                    ]
                )

        if len(gcode) > 0:
            self.commands_handle(gcode)


class RobotApiCommand(Enum):
    Command = "Command"


class RobotApiHandler:

    def __init__(self, config: dict = None):

        self.klipperApiHandler = KlipperApiHandler(
            config=get_config(config, "klipper"), on_response=self.on_response
        )

        if get_config(config, "broker:enabled"):
            self.config = get_config(config, "broker")
            client = mqtt.Client()
            self.client = client
            client.on_connect = self.on_connect
            client.on_message = self.on_message
            # client.connect(get_config(self.config, 'url'), get_config(self.config, 'port'), 60)
            # client.loop_start()

    def command(self, command, data: str = None):
        if command == RobotApiCommand.Command:
            self.klipperApiHandler.request(data)

    def on_connect(self, client, userdata, flags, rc):
        client.subscribe(get_config(self.config, "topic:request"))

    def on_message(self, client, userdata, msg):
        self.command(RobotApiCommand.Command, msg.payload.decode())

    def on_response(self, msg: str):
        if self.client:
            self.client.publish(
                topic=get_config(self.config, "topic:response"), payload=msg
            )

    def ready(self):
        return self.klipperApiHandler.request_empty()


class KlipperApiHandler:

    def __init__(self, config: dict, on_response=None):

        self.config = config
        self.requests = []
        self.key = 1
        self.lock = False  # prevent request to send without previous completed
        self.r = ""
        self.on_response = on_response

        # request, response thread
        self.request_thread = None
        self.response_thread = None

        self._connect(immediate=True)

    def _connect(self, immediate: bool = False):
        try:
            if not immediate:
                time.sleep(5)
            client_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self.client_socket = client_socket
            self.client_socket.connect(get_config(self.config, "socket"))
            self.lock = False
            self.subscribe_output()
        except Exception as e:
            pass

    def _get_socket(self):
        return self.client_socket

    def request(self, command: str):
        self.requests.append({"k": self.key, "c": command})
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
                    responses = data.split(b"\x03")
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

        if "error" in res:
            msg = res["error"]["message"]
            self.lock = False
        else:
            if "G1" in self.r or "SERVO" in self.r:
                if "toolhead:move" in msg:
                    self.lock = False
            else:
                if "result" in res:
                    self.lock = False

        if not self.lock and self.on_response:
            self.on_response(msg)

    fmt_c = """{
        "id":%d,
        "method":"gcode/script",
        "params":{
            "script":"%s"
        }
    }"""

    def run_code(self):
        r = self.r
        message = KlipperApiHandler.fmt_c % (r["k"], r["c"])
        self.client_socket.sendall(("%s\x03" % (message)).encode())

    fmt_o = """{
        "id":%d,
        "method":
        "gcode/subscribe_output",
        "params":{
            "response_template":{"key":%d}
        }
    }"""

    def subscribe_output(self):
        message = KlipperApiHandler.fmt_o % (1, 1)
        self.client_socket.sendall(("%s\x03" % (message)).encode())

    def request_empty(self):
        return len(self.requests) == 0 and not self.lock


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
