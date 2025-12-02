import json
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
import tensorflow as tf
from tensorflow.keras import models, layers, optimizers, regularizers
from keras.datasets import mnist
import threading
import os
import time
from datetime import datetime

# TODO fix the bug that repeatedly sends sample if the save drawing as sample button is pressed
# TODO check server no-pull to truenas


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CLIENT_DIR = os.path.join(BASE_DIR, "..", "client")

DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

MODEL_FILE = os.path.join(DATA_DIR, "saved_model.keras")
LOG_FILE = os.path.join(DATA_DIR, "evaluation_log.json")


global_model = None
global_model_initialized = False
weight_shapes = None

round_index = 0
pending_updates = []
lock = threading.Lock()
round_freq = 60

eval_log = []
community_feedback = {}


(_, _), (x_test, y_test) = mnist.load_data()
x_test = x_test.astype("float32") / 255.0
x_test = x_test.reshape(-1, 28*28)
y_test = tf.keras.utils.to_categorical(y_test, 10)


def create_model():
    model = models.Sequential([
        layers.Input(shape=(784,)),
        layers.Dense(
            128,
            activation='relu',
            kernel_initializer='glorot_uniform',
            kernel_regularizer=regularizers.l2(1e-4)
        ),
        layers.Dense(
            64,
            activation='relu',
            kernel_initializer='glorot_uniform',
            kernel_regularizer=regularizers.l2(1e-4)
        ),
        layers.Dense(
            10,
            activation='softmax',
            kernel_initializer='glorot_uniform'
        )
    ])

    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.0015),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def set_model_weights(model, weights_arrays, shapes):
    tensors = [np.array(arr, dtype=np.float32).reshape(shape)
               for arr, shape in zip(weights_arrays, shapes)]
    model.set_weights(tensors)


def extract_weights(model):
    weights = model.get_weights()
    shapes = [w.shape for w in weights]
    arrays = [w.flatten().tolist() for w in weights]
    return arrays, shapes


def load_global_model_if_exists():
    global global_model, global_model_initialized, weight_shapes

    if os.path.exists(MODEL_FILE):
        print("Loading saved global model")
        global_model = tf.keras.models.load_model(MODEL_FILE, compile=False)
        _, weight_shapes = extract_weights(global_model)
        global_model.compile(
            optimizer=optimizers.Adam(learning_rate=0.0015),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        global_model_initialized = True
        return True

    print("No saved global model found. Creating a new one")
    global_model = create_model()
    weight_shapes = [w.shape for w in global_model.get_weights()]
    global_model_initialized = True
    global_model.save(MODEL_FILE, include_optimizer=True)
    return False


def save_eval_log():
    with open(LOG_FILE, "w") as f:
        json.dump(eval_log, f, indent=2)


def load_eval_log():
    global eval_log, community_feedback, round_index

    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r") as f:
            try:
                eval_log = json.load(f)
            except json.JSONDecodeError:
                print("Warning: Log file is corrupted. Starting with empty log.")
                eval_log = []
    else:
        eval_log = []
        with open(LOG_FILE, "w") as f:
            json.dump(eval_log, f, indent=2)
        print(f"Created new evaluation log at {LOG_FILE}")
    community_feedback.clear()
    for entry in eval_log:
        if "community_votes" in entry:
            community_feedback[entry["round"]] = entry["community_votes"]
    round_index = eval_log[-1]["round"] if eval_log else 0


load_global_model_if_exists()
load_eval_log()

next_round = eval_log[-1]["round"] + 1 if eval_log else 1

app = Flask(__name__, static_folder=CLIENT_DIR, static_url_path="")


@app.route("/get_model")
def get_model():
    if not global_model_initialized:
        return jsonify({"initialized": False})

    arrays, shapes = extract_weights(global_model)
    return jsonify({"initialized": True, "weights": arrays, "shapes": shapes})


@app.route("/submit_update", methods=["POST"])
def submit_update():
    global pending_updates, weight_shapes, global_model_initialized, global_model, next_round

    data = request.json or {}

    client_deltas = data.get("deltas")
    client_shapes = data.get("shapes")
    client_size = data.get("client_size", 1)
    is_delta = bool(data.get("is_delta", False))

    if not client_shapes or not isinstance(client_shapes, list):
        return jsonify({"status": "error", "message": "Missing or invalid shapes"}), 400

    if client_deltas is None or not isinstance(client_deltas, list):
        return jsonify({"status": "error", "message": "Missing or invalid deltas"}), 400

    with lock:
        if not global_model_initialized:
            print("Global model not initialized, using incoming update to initialize.")
            global_model = create_model()
            weight_shapes = client_shapes
            try:
                set_model_weights(global_model, client_deltas, weight_shapes)
            except Exception as e:
                print("Error setting initial model weights:", e)
                return jsonify({"status": "error", "message": str(e)}), 500
            global_model_initialized = True
            global_model.save(MODEL_FILE)
            is_delta = False

        pending_updates.append({
            "deltas": client_deltas,
            "shapes": client_shapes,
            "client_size": client_size,
            "is_delta": is_delta
        })

        print(f"Received update for round {next_round} "
              f"Updates waiting: {len(pending_updates)} (most recent is_delta={is_delta})")

    return jsonify({"status": "stored", "pending_updates": len(pending_updates)})


def perform_fedavg_round():
    global pending_updates, global_model, weight_shapes

    if not pending_updates:
        print("No updates to aggregate this round.")
        return False

    print(f"Aggregating {len(pending_updates)} delta updates")

    current_weights = global_model.get_weights()
    accum = [np.zeros_like(w) for w in current_weights]
    total_size = 0
    valid_updates = []

    for update in pending_updates:
        if not update.get("is_delta", False):
            print("Skipping invalid update.")
            continue
        valid_updates.append(update)
        deltas = [np.array(w, dtype=np.float32).reshape(shape)
                  for w, shape in zip(update["deltas"], update["shapes"])]
        size = update["client_size"]
        for i in range(len(deltas)):
            accum[i] += deltas[i] * size
        total_size += size

    if not valid_updates:
        print("No valid updates to aggregate this round.")
        return False

    avg_delta = [a / total_size for a in accum]
    new_weights = [cw + d for cw, d in zip(current_weights, avg_delta)]
    global_model.set_weights(new_weights)

    global_model.save(MODEL_FILE, include_optimizer=False)

    for vu in valid_updates:
        pending_updates.remove(vu)

    print("FedAvg applied successfully and model saved.")
    return True


@app.route("/finish_round")
def finish_round():
    with lock:
        if not pending_updates:
            return jsonify({"status": "no_updates"})

        global next_round
        perform_fedavg_round()
        entry = evaluate_model(round_override=next_round)
        next_round += 1
        return entry


def compute_global_community_accuracy():
    all_votes = []
    for votes in community_feedback.values():
        all_votes.extend(votes)
    return float(np.mean(all_votes)) if all_votes else None


def evaluate_model(round_override=None):

    if not global_model_initialized:
        return jsonify({"initialized": False})

    loss, acc = global_model.evaluate(x_test, y_test, verbose=0)

    if round_override is not None:
        current_round = round_override
    else:
        current_round = next_round

    community_acc = compute_global_community_accuracy()
    votes = community_feedback.get(current_round, [])

    entry = {
        "round": current_round,
        "loss": float(loss),
        "accuracy_model": float(acc),
        "accuracy_community": community_acc,
        "community_votes": votes,
        "timestamp": datetime.utcnow().isoformat()
    }

    eval_log.append(entry)
    save_eval_log()

    return entry


def evaluation_worker():
    global next_round
    with app.app_context():
        while True:
            with lock:
                if pending_updates:
                    print("Checking pending updates for FL round")
                    updated = perform_fedavg_round()
                    if updated:
                        entry = evaluate_model(round_override=next_round)
                        print(f"Round {next_round} evaluation: "
                              f"Model Acc={entry['accuracy_model']:.4f}, "
                              f"Community Acc={entry['accuracy_community']}")
                        next_round += 1
            time.sleep(round_freq)


@app.route("/submit_user_feedback", methods=["POST"])
def submit_user_feedback():
    global community_feedback, next_round
    data = request.json
    user_accuracy = float(data["user_accuracy"])
    with lock:
        target_round = next_round
        community_feedback.setdefault(target_round, []).append(user_accuracy)
        avg_acc = compute_global_community_accuracy()
        if eval_log:
            eval_log[-1]["accuracy_community"] = avg_acc
            save_eval_log()
    return jsonify({"status": "updated", "community_accuracy": avg_acc})


@app.route("/evaluation_log")
def evaluation_log_route():
    return jsonify(eval_log)


@app.route("/")
def serve_index():
    return send_from_directory(CLIENT_DIR, "index.html")


if __name__ == "__main__":
    threading.Thread(target=evaluation_worker, daemon=True).start()
    print("Fed_avg server started")
    app.run(host="0.0.0.0", port=5000, debug=False)
