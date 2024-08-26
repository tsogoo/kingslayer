from http.server import BaseHTTPRequestHandler, HTTPServer
import requests
import os
import yaml
from common.config import get_config

with open(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "app.yaml"), "r"
) as file:
    config = yaml.safe_load(file)


class SimpleHTTPRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        # run subprocess command
        if self.path.startswith("/photo"):
            self.send_response(200)
            self.send_header("Content-type", "image/jpeg")
            self.end_headers()
            # return image of warped_image.jpg and augmented.jpg
            try:
                query = self.path.split("?")[1]
                image = query.split("=")[1]
                with open(image, "rb") as image_file:
                    self.wfile.write(image_file.read())

            except FileNotFoundError:
                self.send_response(404)
                self.end_headers()
                self.wfile.write(b"Image not found")
            return
        self.send_response(200)
        self.send_header("Content-type", "application/json")
        self.end_headers()
        if self.path == "/service_start":
            os.system("sudo systemctl start kingslayer_service.service")
            self.wfile.write(b"Started service")
        elif self.path == "/service_stop":
            os.system("sudo systemctl stop kingslayer_service.service")
            self.wfile.write(b"Stopped service")
        elif self.path == "/service_restart":
            os.system("sudo systemctl restart kingslayer_service.service")
            self.wfile.write(b"Restarted service")
        elif self.path.startswith("/move"):
            # get the webcam ip from query string
            query = self.path.split("?")[1]
            webcam_ip = query.split("=")[1]
            os.system(f"python3 change_to_started.py --webcam_ip={webcam_ip}")
            self.wfile.write(b"Started moving")
        elif self.path.startswith("/calibrate"):
            os.system("python3 change_to_calibrate.py")
            self.wfile.write(b"Started calibration")

        else:
            self.wfile.write(b"Kingslayer in the house!")


def run(server_class=HTTPServer, handler_class=SimpleHTTPRequestHandler, port=8000):
    server_address = ("", port)
    httpd = server_class(server_address, handler_class)
    print(f"Starting HTTP server on port {port}")
    httpd.serve_forever()


if __name__ == "__main__":
    run()
