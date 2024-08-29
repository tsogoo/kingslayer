from http.server import BaseHTTPRequestHandler, HTTPServer
import os
import time


class SimpleHTTPRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        # run subprocess command
        if self.path.startswith("/photo"):

            # return image of warped_image.jpg and augmented.jpg
            try:
                query = self.path.split("?")[1]
                image = query.split("=")[1]
                with open(image, "rb") as image_file:
                    self.send_response(200)
                    self.send_header("Content-type", "image/jpeg")
                    self.end_headers()
                    self.wfile.write(image_file.read())

            except FileNotFoundError:
                self.send_response(404)
                self.end_headers()
                self.wfile.write(b"Image not found")
            return
        if self.path == "/service_start":
            os.system("sudo systemctl start kingslayer_service.service")
            response = b"Started service"
        elif self.path == "/service_stop":
            os.system("sudo systemctl stop kingslayer_service.service")
            response = b"Stopped service"
        elif self.path == "/service_restart":
            os.system("sudo systemctl restart kingslayer_service.service")
            response = b"Restarted service"
        elif self.path.startswith("/move") or self.path.startswith("/init_camera"):
            # get the webcam ip from query string
            query = self.path.split("?")[1]
            webcamera_url = query.split("=")[1].split("&")[0]
            status = self.path.split("?")[0].split("/")[1]
            print(f"=========={status}==========")
            if self.path.startswith("/init_camera"):
                light_contour_number = query.split("=")[3].split("&")[0]
                command = "python3 change_status.py --webcam_ip={} --status={} --light_contour_number={}".format(
                    webcamera_url, status, light_contour_number
                )

            else:
                command = "python3 change_status.py --webcam_ip={} --status={} --is_white={}".format(
                    webcamera_url, status, query.split("=")[3].split("&")[0]
                )
            os.system(command)
            print(command)
            time.sleep(4)
            try:
                image = query.split("=")[2].split("&")[0]

                with open(image, "rb") as image_file:
                    self.send_response(200)
                    self.send_header("Content-type", "image/jpeg")
                    self.end_headers()
                    self.wfile.write(image_file.read())

            except FileNotFoundError:
                self.send_response(404)
                self.end_headers()
                self.wfile.write(b"Image not found")
            return
        elif self.path.startswith("/calibrate"):
            os.system("python3 change_to_calibrate.py")
            response = b"Started calibration"

        else:
            response = b"Kingslayer in the house!"
        self.send_response(200)
        self.send_header("Content-type", "application/json")
        self.end_headers()
        self.wfile.write(response)


def run(server_class=HTTPServer, handler_class=SimpleHTTPRequestHandler, port=8000):
    server_address = ("0.0.0.0", port)
    httpd = server_class(server_address, handler_class)
    print(f"Starting HTTP server on port {port}")
    httpd.serve_forever()


if __name__ == "__main__":
    run()
