from flask import Flask, request
from flask_restful import Resource, Api

app = Flask(__name__)
api = Api(app)


class Version(Resource):
    def get(self):
        return {'Product Name': 'HyperBorea Lab', 'Version': '0.01', 'Api version': '0.01'}


api.add_resource(Version, '/api/version/')

if __name__ == '__main__':
    app.run(port=5030, debug=True)
