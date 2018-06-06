
from flask import Flask, Blueprint, abort
from flask import request
from flask import jsonify
from flask_cors import CORS, cross_origin

# start
import numpy as np
import tensorflow as tf
import os

from trainer import Trainer
from config import get_config
from data_loader import get_loader
from utils import prepare_dirs_and_logger, save_config

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

config, unparsed = get_config()

app = Flask(__name__)
CORS(app)  #


prepare_dirs_and_logger(config)

rng = np.random.RandomState(config.random_seed)
tf.set_random_seed(config.random_seed)

if config.is_train:
    data_path = config.data_path
    batch_size = config.batch_size
    do_shuffle = True
else:
    setattr(config, 'batch_size', 1)
    if config.test_data_path is None:
        data_path = config.data_path
    else:
        data_path = config.test_data_path
    batch_size = config.sample_per_image
    do_shuffle = False
data_loader = get_loader(
        data_path, config.batch_size, config.input_scale_size,
        config.data_format, config.split)
trainer = Trainer(config, data_loader)

# end


@app.route('/')
def index():
    return "This is fashion detection"


@app.route('/api/detect', methods=['POST'])
@cross_origin()
def create_task():

    if not request.json or not 'url' in request.json:
        abort(400)
    try:
	print 'test'
        result = myobj.detect_obj.demo(request.json['url'])
	return_data = {}
	return_data['result'] = 'OK'
	return_data['data'] = result
        return jsonify(return_data), 201
    except Exception as e:
	return_data = {}
        return_data['result'] = 'ERROR'
        return_data['json'] = str(request.json)
        return_data['errorMessage'] = str(e)
        return jsonify(return_data), 201


@app.route('/api/base64_gan', methods=['POST'])
@cross_origin()
def create_base64_task():
    if not request.json or not 'base64a' in request.json or not 'base64b' in request.json:
        abort(400)
    try:
	print 'base64'
        result = trainer.test2(
            request.json['base64a'], request.json['base64b'])
	return_data = {}
	return_data['result'] = 'OK'
	return_data['data'] = "data:image/jpeg;base64," + result
        return jsonify(return_data), 201
    except Exception as e:
	return_data = {}
        return_data['result'] = 'ERROR'
        return_data['json'] = str(request.json)
        return_data['errorMessage'] = str(e)
        return jsonify(return_data), 201


@app.route('/api/base64_gan_single_image', methods=['POST'])
@cross_origin()
def create_base64__single_image_task():
    return_data = {}
    if not request.json or not 'base64' in request.json or not 'dim' in request.json or not 'value' in request.json:
        abort(400)
    try:
        dim = request.json['dim']
        print dim
        value = request.json['value']
        print value
        result = trainer.test3(request.json['base64'],dim,value)
        return_data = {}
        return_data['result'] = 'OK'
        return_data['data'] = "data:image/jpeg;base64,"+result
        return jsonify(return_data), 201
    except Exception as e:
        print str(e)
        return_data['result'] = 'ERROR'
        return_data['json'] = str(request.json)
        return_data['errorMessage'] = str(e)
        return jsonify(return_data), 201

if __name__ == '__main__':
	print "aa"
	app.run(port=3333 ,debug=True, use_reloader=False, host='0.0.0.0')
