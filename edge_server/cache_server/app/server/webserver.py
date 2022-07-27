# from http.server import BaseHTTPRequestHandler, HTTPServer
import numpy as np
# class handler(BaseHTTPRequestHandler):
#     def do_GET(self):
#         self.send_response(200)
#         self.send_header('Content-type','text/html')
#         self.end_headers()
#
#         message = np.random.randn()
#         self.wfile.write(message)
#
# with HTTPServer(('', 8000), handler) as server:
#     server.serve_forever()

# Importing flask module in the project is mandatory
# An object of Flask class is our WSGI application.
from flask import Flask

# Flask constructor takes the name of
# current module (__name__) as argument.
app = Flask(__name__)


# The route() function of the Flask class is a decorator,
# which tells the application which URL should call
# the associated function.
@app.route('/')
# ‘/’ URL is bound with hello_world() function.
def hello_world():
    return {"Cache_status":1}

@app.route('/<file>')
def print_name(file):
    return {"Cache_status":np.random.rand()}
# main driver function
if __name__ == '__main__':
    # run() method of Flask class runs the application
    # on the local development server.
    app.run(threaded=True)