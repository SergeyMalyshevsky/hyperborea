import os
import shutil

from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_marshmallow import Marshmallow

# Init app
app = Flask(__name__)
basedir = os.path.abspath(os.path.dirname(__file__))

# Database
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(basedir, 'database/db.sqlite')
app.config['PROJECTS_DIRECTORY'] = os.path.join(basedir, 'projects')
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
    """
    Example of body

    {
	"name": "Titanic competition",
	"description": "Find survive passengers"
    }
    """
    name = request.json['name']
    description = request.json['description']

    project_directory = app.config['PROJECTS_DIRECTORY'] + '/' + name
    if not os.path.exists(project_directory):
        os.mkdir(project_directory)
    else:
        return jsonify({'error': 'Directory already exists'})

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
    name = project.name
    db.session.delete(project)
    db.session.commit()

    project_directory = app.config['PROJECTS_DIRECTORY'] + '/' + name
    try:
        shutil.rmtree(project_directory)
    except:
        print('Error while deleting directory')

    return project_schema.jsonify(project)


# Run Server
if __name__ == '__main__':
    db.create_all()
    app.run(port=5030, debug=True)
