from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import argparse
import time
import os
import yaml
from common.config import get_config
from robot_arm.robot import RobotApiHandler
from robot_arm.robot import RobotApiCommand


with open(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "app.yaml"), "r"
) as file:
    config = yaml.safe_load(file)

r = RobotApiHandler(config)


class SimpleHTTPRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.wfile.write(b"Hello, this is a GET response!")

    def do_POST(self):
        if self.path == "/":
            content_length = int(self.headers["Content-Length"])  # Get the size of data
            post_data = self.rfile.read(content_length)  # Get the data
            data = json.loads(post_data)  # Parse the JSON data
            print(data)
            for i in range(len(data["commands"])):
                r.command(RobotApiCommand.Command, data["commands"][i])

            while not r.ready():
                time.sleep(0.1)
            # Process the data (this example just prints it)

        # Send a response
        self.send_response(200)
        self.send_header("Content-type", "application/json")
        self.end_headers()
        response = {
            "status": "success",
        }
        self.wfile.write(b"Hello, this is a POST response!")
        # self.wfile.write(json.dumps(response).encode('utf-8'))


def run(server_class=HTTPServer, handler_class=SimpleHTTPRequestHandler, port=8000):
    server_address = ("", port)
    httpd = server_class(server_address, handler_class)
    print(f"Starting HTTP server on port {port}")
    httpd.serve_forever()


if __name__ == "__main__":
    run()
