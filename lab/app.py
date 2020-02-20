from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_marshmallow import Marshmallow
import os

# Init app
app = Flask(__name__)
basedir = os.path.abspath(os.path.dirname(__file__))
# Database
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(basedir, 'database/db.sqlite')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
# Init db
db = SQLAlchemy(app)
# Init ma
ma = Marshmallow(app)


# Project Class/Model
class Project(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), unique=True)
    description = db.Column(db.String(200))

    def __init__(self, name, description):
        self.name = name
        self.description = description


# Project Schema
class ProjectSchema(ma.Schema):
    class Meta:
        fields = ('id', 'name', 'description')


# Init schema
project_schema = ProjectSchema()
projects_schema = ProjectSchema(many=True)


# get version
@app.route('/version', methods=['GET'])
def get_version():
    return {'Product Name': 'HyperBorea Lab', 'Version': '0.01', 'Api version': '0.01'}


# Create a Project
@app.route('/project', methods=['POST'])
def add_project():
    name = request.json['name']
    description = request.json['description']

    new_project = Project(name, description)

    db.session.add(new_project)
    db.session.commit()

    return project_schema.jsonify(new_project)


# Get All projects
@app.route('/project', methods=['GET'])
def get_projects():
    all_projects = Project.query.all()
    result = projects_schema.dump(all_projects)
    if result:
        print(result)
        return jsonify(result)
    else:
        return {'answer': 'There is no data'}


# Get Single projects
@app.route('/project/<id>', methods=['GET'])
def get_project(id):
    project = Project.query.get(id)
    return project_schema.jsonify(project)


# Update a project
@app.route('/project/<id>', methods=['PUT'])
def update_project(id):
    project = Project.query.get(id)

    name = request.json['name']
    description = request.json['description']

    project.name = name
    project.description = description

    db.session.commit()

    return project_schema.jsonify(project)


# Delete project
@app.route('/project/<id>', methods=['DELETE'])
def delete_project(id):
    project = Project.query.get(id)
    db.session.delete(project)
    db.session.commit()

    return project_schema.jsonify(project)


# Run Server
if __name__ == '__main__':
    db.create_all()
    app.run(port=5030, debug=True)
