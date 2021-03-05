import flask
import json
import os
from hosted_model import MedicalBertModel
import logging

# The flask app for serving predictions
app = flask.Flask(__name__)

@app.route('/ping', methods=['GET'])
def ping():
    """Determine if the container is working and healthy. In this sample container, we will declare
    it healthy if we can load the model successfully."""
    health = ScoringService.get_model() is not None  # You can insert a health check here
    status = 200 if health else 404
    return flask.Response(response='\n', status=status, mimetype='application/json')

@app.route('/invocations', methods=['POST'])
def transformation():
    """ this specifies the transformation that will be performed with the raw data from the moment the request was receieved to the moment the response was returned"""
    data = None
    #get input data from flask request
    data = flask.request.data.decode('utf-8')
    logging.debug(f"raw data supplied: {data}")
    #instantiate the MedicalBertModel class
    mbm=MedicalBertModel(run_null_model=False)
    #run model
    result=mbm.run_model_and_null(data)
    result=json.dumps(result)
    logging.debug(f"result of model: {result}")
    return flask.Response(response=result, status=200)
