import json
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
import tensorflow as tf
from tensorflow.keras import layers, models
from keras.datasets import mnist
import threading
from threading import Lock
import os, time
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CLIENT_DIR = os.path.join(BASE_DIR, "..", "client")
LOG_FILE = "evaluation_log.json"


global_model = None
global_model_initialized = False
weight_shapes = None

round_index = 0
pending_updates = []
lock = Lock()
round_freq = 60

eval_log = []
community_feedback = {}


(_, _), (x_test, y_test) = mnist.load_data()
x_test = x_test.astype("float32") / 255.0
x_test = x_test.reshape(-1, 28*28)
y_test = tf.keras.utils.to_categorical(y_test, 10)


def create_model():
    model = models.Sequential([
        layers.Dense(64, activation='relu', input_shape=(784,)),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.01),
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
    if os.path.exists("saved_model.keras"):
        print("Loading saved global model…")
        global_model = tf.keras.models.load_model("saved_model.keras")
        _, weight_shapes = extract_weights(global_model)
        global_model_initialized = True
        return True
    return False


def save_eval_log():
    with open(LOG_FILE, "w") as f:
        json.dump(eval_log, f, indent=2)


def load_eval_log():
    global eval_log, community_feedback, round_index
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r") as f:
            eval_log = json.load(f)
        community_feedback.clear()
        for entry in eval_log:
            if "community_votes" in entry:
                community_feedback[entry["round"]] = entry["community_votes"]
        round_index = eval_log[-1]["round"] if eval_log else 0


load_global_model_if_exists()
load_eval_log()

app = Flask(__name__, static_folder=CLIENT_DIR, static_url_path="")


@app.route("/get_model")
def get_model():
    if not global_model_initialized:
        return jsonify({"initialized": False})

    arrays, shapes = extract_weights(global_model)
    return jsonify({"initialized": True, "weights": arrays, "shapes": shapes})


@app.route("/submit_update", methods=["POST"])
def submit_update():
    global pending_updates, weight_shapes, global_model_initialized

    data = request.json
    client_weights = data["weights"]
    client_shapes = data["shapes"]
    client_size = data["client_size"]

    with lock:

        if not global_model_initialized:
            global_model = create_model()
            weight_shapes = client_shapes
            global_model_initialized = True
            global_model.save("saved_model.keras")

        pending_updates.append({
            "weights": client_weights,
            "shapes": client_shapes,
            "client_size": client_size
        })

        print(f"Received update for round {round_index+1}. "
              f"Updates waiting: {len(pending_updates)}")

    return jsonify({"status": "stored", "pending_updates": len(pending_updates)})


def perform_fedavg_round():
    global pending_updates, global_model, weight_shapes, round_index

    if not pending_updates:
        print("No updates to aggregate this round.")
        return None

    print(f"Aggregating {len(pending_updates)} updates (FedAvg)…")

    accum = [np.zeros(shape, dtype=np.float32) for shape in weight_shapes]
    total_size = 0

    for update in pending_updates:
        weights = [np.array(w, dtype=np.float32).reshape(shape)
                   for w, shape in zip(update["weights"], update["shapes"])]
        size = update["client_size"]

        for i in range(len(weights)):
            accum[i] += weights[i] * size
        total_size += size

    averaged = [wsum / total_size for wsum in accum]

    global_model.set_weights(averaged)
    global_model.save("saved_model.keras")

    pending_updates = []

    print("FedAvg applied successfully.")
    return True


@app.route("/finish_round")
def finish_round():
    with lock:
        ok = perform_fedavg_round()
        if ok:
            return evaluate_model()
        else:
            return jsonify({"status": "no_updates"})


def compute_global_community_accuracy():
    all_votes = []
    for votes in community_feedback.values():
        all_votes.extend(votes)
    return float(np.mean(all_votes)) if all_votes else None


@app.route("/evaluate_model")
def evaluate_model():
    global round_index

    if not global_model_initialized:
        return jsonify({"initialized": False})

    loss, acc = global_model.evaluate(x_test, y_test, verbose=0)

    round_index += 1
    community_acc = compute_global_community_accuracy()

    entry = {
        "round": round_index,
        "loss": float(loss),
        "accuracy_model": float(acc),
        "accuracy_community": community_acc,
        "community_votes": community_feedback.get(round_index, []),
        "timestamp": datetime.utcnow().isoformat()
    }

    eval_log.append(entry)
    save_eval_log()

    return entry


def evaluation_worker():
    with app.app_context():
        while True:
            time.sleep(round_freq)
            with lock:
                if pending_updates:
                    print("[AUTO] Finishing FL round…")
                    perform_fedavg_round()
                    evaluate_model()


@app.route("/submit_user_feedback", methods=["POST"])
def submit_user_feedback():
    global community_feedback

    data = request.json
    user_accuracy = float(data["user_accuracy"])
    key = round_index

    community_feedback.setdefault(key, []).append(user_accuracy)

    avg_acc = compute_global_community_accuracy()

    if eval_log:
        eval_log[-1]["accuracy_community"] = avg_acc
        eval_log[-1]["community_votes"] = community_feedback.get(round_index, [])
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
    app.run(host="0.0.0.0", port=5000, debug=True)
