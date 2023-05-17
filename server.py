import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import json
import os
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten

from flask import Flask, render_template, request
from flask_socketio import SocketIO
import threading
import socketio
from flask_cors import CORS

import sqlite3
import db

socketClient = socketio.Client()

app = Flask(__name__)
socketServer = SocketIO(app, cors_allowed_origins="*")

# model = keras.models.load_model("models/_default/box_model_89.45.2_london_checkpoint_6")
model = None
model_id = None
train_x = None
train_y = None
eval_x = None
eval_y = None
sacred_text = None

# ================
# Socket Functions
# ================


@socketServer.on("connect")
def io_test_connect():
    print("Client connected")
    socketServer.emit("status", "terkoneksi")


@socketServer.on("disconnect")
def io_test_disconnect():
    print("Client disconnected")


@socketServer.on("load_model")
def io_load_model(data):
    print("client requested to load_model", data)
    global model, sacred_text
    # does the model exist?
    dir_list = os.listdir("models")
    exists = False
    for folder in dir_list:
        if folder == data["nama"]:
            exists = True
            break
    if exists:
        socketServer.emit("status", "Loading model")
        if load_model(data["nama"], data["id"]):
            socketServer.emit("status", "Loading dataset")
            res = load_dataset(data)
            if res:
                sacred_text = f"JST siap digunakan. {np.shape(train_x)[0]} data latih. {np.shape(eval_x)[0]} data uji"
                socketServer.emit(
                    "status",
                    sacred_text,
                )
                socketServer.emit("jst_readyness", True)
    else:
        socketServer.emit("status", "Membuat model")
        res = create_model(data["nama"], data["susunan"])
        if res:
            socketServer.emit("status", "Loading dataset")
            res = load_dataset(data)
            if res:
                sacred_text = f"JST siap digunakan. {np.shape(train_x)[0]} data latih. {np.shape(eval_x)[0]} data uji"
                socketServer.emit(
                    "status",
                    sacred_text,
                )
                socketServer.emit("jst_readyness", True)


@socketServer.on("request_iterations")
def io_get_iterations(data):
    print("client requested iterations")
    res = db.get_all_iteration_by_model_id(data["id"])
    # print("got",res)
    socketServer.emit("iterations", res)


@socketServer.on("train_model")
def io_train_model(data):
    train_model(data["id"], data["epoch"])


def create_model(nama, susunan):
    global model
    model = Sequential()
    model.add(Flatten(input_shape=(200,)))
    print(susunan, type(susunan))
    for susu in json.loads(susunan):
        print("susu", susu)
        model.add(Dense(susu, activation="tanh"))
    model.add(Dense(2, activation="softmax"))
    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    model.save(f"models/{nama}/model")
    return True


def load_model(nama, id):
    global model, model_id
    if not model_id == id:
        model = keras.models.load_model(f"models/{nama}/model")
    return True


def train_model(id, epoch):
    print('Training model')
    global model, train_x, train_y, eval_x, eval_y, sacred_text
    socketServer.emit("status", "Melatih...")
    socketServer.emit('jst_readyness',False)
    train_history = model.fit(train_x, train_y, epochs=1).history
    eval_history = model.evaluate(eval_x, eval_y, verbose=1)
    db.iterasi_add(
        (
            id,
            epoch,
            train_history["loss"][0],
            train_history["accuracy"][0],
            eval_history[0],
            eval_history[1]
        )
    )
    socketServer.emit("status",sacred_text)
    socketServer.emit('jst_readyness',True)
    res = db.get_all_iteration_by_model_id(id)
    socketServer.emit("iterations", res)



# ===============
# Flask Functions
# ===============


@app.route("/")
def flask_home():
    return "Hello World!"


@app.route("/get_models_all")
def flask_get_models_all():
    models = db.get_models_all()
    models_baru = []
    # print('models',models)
    for row in models:
        row_baru = list(row)
        print('row',row)
        iterasi = db.get_latest_iteration_by_model_id(row[0])
        # print('iterasi')
        if iterasi:
            row_baru.append(iterasi[5])
            row_baru.append(iterasi[6])
        models_baru.append(row_baru)
    return models_baru


@app.route("/get_model", methods=["POST"])
def flask_get_model():
    model = db.get_model_by_id(request.json["id"])
    print(model, type(model))
    return [model]


@app.route("/create_model", methods=["POST"])
def flask_create_model():
    print("creating new model")
    print(request.json)
    data = request.json
    res = db.create_model(
        (
            data["nama"],
            data["latih_mulai"],
            data["latih_akhir"],
            int(data["latih_london"]),
            data["uji_mulai"],
            data["uji_akhir"],
            int(data["uji_london"]),
            json.dumps(data["neurons"]),
        )
    )
    # res = db.create_model((
    #     1,1,1,1,1,1,1,1,1
    # ))
    print(res)
    return str(res)


@app.route("/delete_model", methods=["POST"])
def flask_delete_model():
    res = db.delete_model(request.json["id"])
    return "ok"


@app.route("/get_model_detail")
def flask_get_model_detail():
    layers = []
    for i, layer in enumerate(model.layers):
        try:
            layers.append({
                'neurons' : layer.output_shape[-1],
                'activation': layer.activation.__name__
            })
        except:
            layers.append({
                'neurons' : layer.output_shape[-1],
                'activation': 'ok'
            })
    return layers

# ===============
# Other functions
# ===============


def load_dataset(data):
    global train_x, train_y, eval_x, eval_y, model_id
    if model_id == data["id"]:
        return True
    pd_train_x, pd_train_y, pd_eval_x, pd_eval_y = (
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
    )
    # train_dates, eval_dates = [], []
    train_london = True if data["latih_london"] == 1 else False
    eval_london = True if data["uji_london"] == 1 else False
    # insert to train dates
    train_year, train_month = int(data["latih_mulai"][:4]), int(
        data["latih_mulai"][-2:]
    )
    train_year_end, train_month_end = int(data["latih_akhir"][:4]), int(
        data["latih_akhir"][-2:]
    )
    while True:
        # year is more
        if train_year > train_year_end:
            break
        # year is same, month is more
        if train_year == train_year_end and train_month > train_month_end:
            break
        # append
        if train_london:
            filename_x = f"dataset/london/x/box_feature_london_{train_year}.{str(train_month).zfill(2)}.01_x.csv"
            filename_y = f"dataset/london/y/box_feature_london_{train_year}.{str(train_month).zfill(2)}.01_y.csv"

        else:
            filename_x = f"dataset/x/box_feature_{train_year}.{str(train_month).zfill(2)}.01_x.csv"
            filename_y = f"dataset/y/box_feature_{train_year}.{str(train_month).zfill(2)}.01_y.csv"
        pd_train_x = pd.concat([pd_train_x, pd.read_csv(filename_x)])
        pd_train_y = pd.concat([pd_train_y, pd.read_csv(filename_y)])
        # change value
        if train_month == 12:
            train_month = 1
            train_year += 1
        else:
            train_month += 1
    print("dataframe shape:", pd_train_x.shape)
    train_x = np.array(pd_train_x, dtype="float32")
    train_y = np.array(pd_train_y, dtype="int8")
    print("train data loaded. shape: ", np.shape(train_x), np.shape(train_y))
    # print("sample: ", train_x[0], train_y[0])
    # insert to eval dates
    eval_year, eval_month = int(data["uji_mulai"][:4]), int(data["uji_mulai"][-2:])
    eval_year_end, eval_month_end = int(data["uji_akhir"][:4]), int(
        data["uji_akhir"][-2:]
    )
    while True:
        # year is more
        if eval_year > eval_year_end:
            break
        # year is same, month is more
        if eval_year == eval_year_end and eval_month > eval_month_end:
            break
        # append
        if eval_london:
            filename_x = f"dataset/london/x/box_feature_london_{eval_year}.{str(eval_month).zfill(2)}.01_x.csv"
            filename_y = f"dataset/london/y/box_feature_london_{eval_year}.{str(eval_month).zfill(2)}.01_y.csv"

        else:
            filename_x = (
                f"dataset/x/box_feature_{eval_year}.{str(eval_month).zfill(2)}.01_x.csv"
            )
            filename_y = (
                f"dataset/y/box_feature_{eval_year}.{str(eval_month).zfill(2)}.01_y.csv"
            )
        pd_eval_x = pd.concat([pd_eval_x, pd.read_csv(filename_x)])
        pd_eval_y = pd.concat([pd_eval_y, pd.read_csv(filename_y)])
        # change value
        if eval_month == 12:
            eval_month = 1
            eval_year += 1
        else:
            eval_month += 1
    print("dataframe shape:", pd_eval_x.shape)
    eval_x = np.array(pd_eval_x, dtype="float32")
    eval_y = np.array(pd_eval_y, dtype="int8")
    print("eval data loaded. shape: ", np.shape(eval_x), np.shape(eval_y))
    # print("sample: ", eval_x[0], eval_y[0])

    model_id = data["id"]
    return True

# ===
# end
# ===

CORS(app, resources={r"*": {"origins": "*"}})
socketServer.run(app, debug=True)