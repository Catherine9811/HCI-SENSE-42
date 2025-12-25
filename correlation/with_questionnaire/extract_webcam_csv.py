import numpy as np
import pickle
import os
import pandas as pd
import jellyfish
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns
import addcopyfighandler
from collections import defaultdict
from data_parser import DataParser
from data_definition import psydat_files

SMOOTH_WINDOW = 5 * 60

# --- helper functions ---
def euler_deg_to_quat(roll, pitch, yaw):
    """
    Convert roll, pitch, yaw (degrees) to quaternion (w, x, y, z).
    Assumes roll, pitch, yaw are arrays or scalars and uses the common
    roll (x), pitch (y), yaw (z) order.
    """
    r = np.deg2rad(roll) * 0.5
    p = np.deg2rad(pitch) * 0.5
    y = np.deg2rad(yaw) * 0.5

    cr = np.cos(r); sr = np.sin(r)
    cp = np.cos(p); sp = np.sin(p)
    cy = np.cos(y); sy = np.sin(y)

    # quaternion components
    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy

    # shape -> (N,4) if inputs are arrays
    return np.stack([qw, qx, qy, qz], axis=-1)


def quat_dot(q1, q2):
    """Dot product between quaternions along last axis."""
    return np.sum(q1 * q2, axis=-1)


def quat_angle_between(q1, q2):
    """
    Quaternion angular distance: theta = 2 * arccos(|dot(q1,q2)|)
    Returns angle in radians.
    """
    d = np.clip(np.abs(quat_dot(q1, q2)), -1.0, 1.0)
    return 2.0 * np.arccos(d)


def forward_vector_from_euler(pitch_deg, yaw_deg):
    """
    Compute unit forward vector from pitch and yaw (degrees).
    Uses:
      x = cos(pitch) * cos(yaw)
      y = cos(pitch) * sin(yaw)
      z = sin(pitch)
    Works with arrays.
    """
    p = np.deg2rad(pitch_deg)
    y = np.deg2rad(yaw_deg)
    x = np.cos(p) * np.cos(y)
    yy = np.cos(p) * np.sin(y)
    z = np.sin(p)
    vec = np.stack([x, yy, z], axis=-1)
    # normalize (just in case)
    norms = np.linalg.norm(vec, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vec / norms


def filter_time_series(times, values):
    return times, values
    # times = np.array(times)
    # values = np.array(values)
    # condition = (times < 2 * 60 * 60)
    # print(f"Time filtered {np.sum(condition)} items.")
    # return times[condition], values[condition]


def filter_nan_indices(processed):
    """Remove all indices where any value in the dictionary contains NaN."""
    # Find indices where any column contains NaN
    nan_indices = {i for values in processed.values() for i, v in enumerate(values) if np.isnan(v)}

    # Keep only the indices that are NOT in nan_indices
    processed = {key: [v for i, v in enumerate(values) if i not in nan_indices] for key, values in processed.items()}

    print(f"{len(set(nan_indices))} filtered.")

    return processed


def to_title_case(s):
    return s.replace("_", " ").title()


class PerformanceAnswerExtractor:
    name = "performance"

    def process(self, parser):
        questionnaires = parser['browser_content']
        question_name = "performance:_how_successful_were_you_in_accomplishing_what_you_were_asked_to_do_slider"
        x_values = [entry["browser_content.started"] + entry[f"{question_name}.rt"] for entry in questionnaires]
        y_values = [entry[f"{question_name}.rating"] for entry in questionnaires]
        return x_values, y_values


class TemporalDemandAnswerExtractor:
    name = "temporal_demand"

    def process(self, parser):
        questionnaires = parser['browser_content']
        question_name = "temporal_demand:_how_hurried_or_rushed_was_the_pace_of_the_task_slider"
        x_values = [entry["browser_content.started"] + entry[f"{question_name}.rt"] for entry in questionnaires]
        y_values = [entry[f"{question_name}.rating"] for entry in questionnaires]
        return x_values, y_values


class AttentivenessAnswerExtractor:
    name = "attentiveness"

    def process(self, parser):
        questionnaires = parser['browser_content']
        question_name = "attentiveness:_how_focused_were_you_on_performing_the_task_slider"
        x_values = [entry["browser_content.started"] + entry[f"{question_name}.rt"] for entry in questionnaires]
        y_values = [entry[f"{question_name}.rating"] for entry in questionnaires]
        return x_values, y_values


class SleepinessAnswerExtractor:
    name = "sleepiness"

    def process(self, parser):
        questionnaires = parser['browser_content']
        question_name = "sleepiness:_how_sleepy_are_you_slider"
        x_values = [entry["browser_content.started"] + entry[f"{question_name}.rt"] for entry in questionnaires]
        y_values = [entry[f"{question_name}.rating"] for entry in questionnaires]
        return x_values, y_values


class BlinkTimesExtractor:
    name = "blink_times"
    base_path = os.path.join("..", "..", "data", "Webcam")

    def process(self, parser):
        x_values = []
        y_values = []
        camera_name = "camera.filename"
        camera_task = parser[camera_name]
        # Extracting values for plotting
        for entry in camera_task:
            key_name = "style_randomizer"
            file_name = entry[camera_name]
            if f"{key_name}.started" in entry:
                _, video_filename = os.path.split(file_name)
                csv_filename = video_filename.replace(".mp4", ".csv")
                csv_path = os.path.join(self.base_path, csv_filename)
                if os.path.exists(csv_path):
                    csv = pd.read_csv(csv_path)
                    # ensure columns exist (robust)
                    for col in ("blink", ):
                        if col not in csv.columns:
                            raise ValueError(f"Missing required column '{col}' in {csv_path}")

                    blink = csv["blink"].to_numpy()
                    x_values.append(entry[f"{key_name}.started"])
                    y_values.append(blink[-1])
                else:
                    print(f"{csv_path} not found!")
        return x_values, y_values


class LookDownTimesExtractor:
    name = "look_down_times"
    base_path = os.path.join("..", "..", "data", "Webcam")

    def process(self, parser):
        x_values = []
        y_values = []
        camera_name = "camera.filename"
        camera_task = parser[camera_name]
        # Extracting values for plotting
        for entry in camera_task:
            key_name = "style_randomizer"
            file_name = entry[camera_name]
            if f"{key_name}.started" in entry:
                _, video_filename = os.path.split(file_name)
                csv_filename = video_filename.replace(".mp4", ".csv")
                csv_path = os.path.join(self.base_path, csv_filename)
                if os.path.exists(csv_path):
                    csv = pd.read_csv(csv_path)
                    # ensure columns exist (robust)
                    for col in ("look_down", ):
                        if col not in csv.columns:
                            raise ValueError(f"Missing required column '{col}' in {csv_path}")

                    blink = csv["look_down"].to_numpy()
                    x_values.append(entry[f"{key_name}.started"])
                    y_values.append(blink[-1])
                else:
                    print(f"{csv_path} not found!")
        return x_values, y_values


class HeadPoseMovementExtractor:
    name = "head_pose_movement"
    base_path = os.path.join("..", "..", "data", "Webcam")

    def process(self, parser):
        x_values = []
        y_values = []
        camera_name = "camera.filename"
        camera_task = parser[camera_name]
        # Extracting values for plotting
        for entry in camera_task:
            key_name = "style_randomizer"
            file_name = entry[camera_name]
            if f"{key_name}.started" in entry:
                _, video_filename = os.path.split(file_name)
                csv_filename = video_filename.replace(".mp4", ".csv")
                csv_path = os.path.join(self.base_path, csv_filename)
                if os.path.exists(csv_path):
                    csv = pd.read_csv(csv_path)
                    # ensure columns exist (robust)
                    for col in ("pitch", "yaw", "roll"):
                        if col not in csv.columns:
                            raise ValueError(f"Missing required column '{col}' in {csv_path}")

                    # drop rows with NaN in the Euler columns
                    sub = csv[["pitch", "yaw", "roll"]].dropna().to_numpy()
                    pitch = sub[:, 0]
                    yaw = sub[:, 1]
                    roll = sub[:, 2]

                    # 1) forward vectors (ignores roll) + spherical variance
                    # fvecs = forward_vector_from_euler(pitch, yaw)  # shape (N,3)
                    # mean_vec = np.mean(fvecs, axis=0)
                    # R = np.linalg.norm(mean_vec) / 1.0  # resultant length (0..1) since vectors are unit
                    # spherical_variance = 1.0 - R

                    # 2) angular steps between successive forward vectors (vector method)
                    # dots = np.clip(np.sum(fvecs[1:] * fvecs[:-1], axis=1), -1.0, 1.0)
                    # ang_steps_vec = np.arccos(dots)  # radians

                    # mean_ang_vec = np.mean(ang_steps_vec)
                    # var_ang_vec = np.var(ang_steps_vec)
                    # total_rot_vec = np.sum(ang_steps_vec)
                    # max_step_vec = np.max(ang_steps_vec)
                    # rms_ang_vec = np.sqrt(np.mean(ang_steps_vec ** 2))

                    # 3) quaternion method (accounts for roll properly)
                    quats = euler_deg_to_quat(roll, pitch, yaw)  # (w,x,y,z)
                    ang_steps_quat = quat_angle_between(quats[1:], quats[:-1])  # radians
                    mean_ang_quat = np.mean(ang_steps_quat)
                    # var_ang_quat = np.var(ang_steps_quat)
                    # total_rot_quat = np.sum(ang_steps_quat)
                    # max_step_quat = np.max(ang_steps_quat)
                    # rms_ang_quat = np.sqrt(np.mean(ang_steps_quat ** 2))

                    # 4) jitter (magnitude of derivative of forward vector)
                    # dvec = fvecs[1:] - fvecs[:-1]
                    # jitter_mag = np.linalg.norm(dvec, axis=1)
                    # mean_jitter = np.mean(jitter_mag)
                    # var_jitter = np.var(jitter_mag)
                    # rms_jitter = np.sqrt(np.mean(jitter_mag ** 2))

                    # --- Append a set of meaningful metrics to x_values/y_values ---
                    # Use degrees for angular values (more interpretable)
                    rad2deg = 180.0 / np.pi
                    x_values.append(entry[f"{key_name}.started"])
                    y_values.append(mean_ang_quat * rad2deg)
                else:
                    print(f"{csv_path} not found!")
        return x_values, y_values


class HeadPoseVariationExtractor:
    name = "head_pose_variation"
    base_path = os.path.join("..", "..", "data", "Webcam")

    def process(self, parser):
        x_values = []
        y_values = []
        camera_name = "camera.filename"
        camera_task = parser[camera_name]
        # Extracting values for plotting
        for entry in camera_task:
            key_name = "style_randomizer"
            file_name = entry[camera_name]
            if f"{key_name}.started" in entry:
                _, video_filename = os.path.split(file_name)
                csv_filename = video_filename.replace(".mp4", ".csv")
                csv_path = os.path.join(self.base_path, csv_filename)
                if os.path.exists(csv_path):
                    csv = pd.read_csv(csv_path)
                    # ensure columns exist (robust)
                    for col in ("pitch", "yaw", "roll"):
                        if col not in csv.columns:
                            raise ValueError(f"Missing required column '{col}' in {csv_path}")

                    # drop rows with NaN in the Euler columns
                    sub = csv[["pitch", "yaw", "roll"]].dropna().to_numpy()
                    pitch = sub[:, 0]
                    yaw = sub[:, 1]
                    roll = sub[:, 2]

                    # 1) forward vectors (ignores roll) + spherical variance
                    # fvecs = forward_vector_from_euler(pitch, yaw)  # shape (N,3)
                    # mean_vec = np.mean(fvecs, axis=0)
                    # R = np.linalg.norm(mean_vec) / 1.0  # resultant length (0..1) since vectors are unit
                    # spherical_variance = 1.0 - R

                    # 2) angular steps between successive forward vectors (vector method)
                    # dots = np.clip(np.sum(fvecs[1:] * fvecs[:-1], axis=1), -1.0, 1.0)
                    # ang_steps_vec = np.arccos(dots)  # radians

                    # mean_ang_vec = np.mean(ang_steps_vec)
                    # var_ang_vec = np.var(ang_steps_vec)
                    # total_rot_vec = np.sum(ang_steps_vec)
                    # max_step_vec = np.max(ang_steps_vec)
                    # rms_ang_vec = np.sqrt(np.mean(ang_steps_vec ** 2))

                    # 3) quaternion method (accounts for roll properly)
                    quats = euler_deg_to_quat(roll, pitch, yaw)  # (w,x,y,z)
                    ang_steps_quat = quat_angle_between(quats[1:], quats[:-1])  # radians
                    std_ang_quat = np.std(ang_steps_quat)
                    # var_ang_quat = np.var(ang_steps_quat)
                    # total_rot_quat = np.sum(ang_steps_quat)
                    # max_step_quat = np.max(ang_steps_quat)
                    # rms_ang_quat = np.sqrt(np.mean(ang_steps_quat ** 2))

                    # 4) jitter (magnitude of derivative of forward vector)
                    # dvec = fvecs[1:] - fvecs[:-1]
                    # jitter_mag = np.linalg.norm(dvec, axis=1)
                    # mean_jitter = np.mean(jitter_mag)
                    # var_jitter = np.var(jitter_mag)
                    # rms_jitter = np.sqrt(np.mean(jitter_mag ** 2))

                    # --- Append a set of meaningful metrics to x_values/y_values ---
                    # Use degrees for angular values (more interpretable)
                    rad2deg = 180.0 / np.pi
                    x_values.append(entry[f"{key_name}.started"])
                    y_values.append(std_ang_quat * rad2deg)
                else:
                    print(f"{csv_path} not found!")
        return x_values, y_values


class HeadRollVariationExtractor:
    name = "head_roll_variation"
    base_path = os.path.join("..", "..", "data", "Webcam")

    def process(self, parser):
        x_values = []
        y_values = []
        camera_name = "camera.filename"
        camera_task = parser[camera_name]
        # Extracting values for plotting
        for entry in camera_task:
            key_name = "style_randomizer"
            file_name = entry[camera_name]
            if f"{key_name}.started" in entry:
                _, video_filename = os.path.split(file_name)
                csv_filename = video_filename.replace(".mp4", ".csv")
                csv_path = os.path.join(self.base_path, csv_filename)
                if os.path.exists(csv_path):
                    csv = pd.read_csv(csv_path)
                    # ensure columns exist (robust)
                    for col in ("pitch", "yaw", "roll"):
                        if col not in csv.columns:
                            raise ValueError(f"Missing required column '{col}' in {csv_path}")

                    # drop rows with NaN in the Euler columns
                    sub = csv[["pitch", "yaw", "roll"]].dropna().to_numpy()
                    pitch = sub[:, 0]
                    yaw = sub[:, 1]
                    roll = sub[:, 2]

                    std_ang_roll = np.std(roll)

                    x_values.append(entry[f"{key_name}.started"])
                    y_values.append(std_ang_roll)
                else:
                    print(f"{csv_path} not found!")
        return x_values, y_values


class HeadPitchVariationExtractor:
    name = "head_pitch_variation"
    base_path = os.path.join("..", "..", "data", "Webcam")

    def process(self, parser):
        x_values = []
        y_values = []
        camera_name = "camera.filename"
        camera_task = parser[camera_name]
        # Extracting values for plotting
        for entry in camera_task:
            key_name = "style_randomizer"
            file_name = entry[camera_name]
            if f"{key_name}.started" in entry:
                _, video_filename = os.path.split(file_name)
                csv_filename = video_filename.replace(".mp4", ".csv")
                csv_path = os.path.join(self.base_path, csv_filename)
                if os.path.exists(csv_path):
                    csv = pd.read_csv(csv_path)
                    # ensure columns exist (robust)
                    for col in ("pitch", "yaw", "roll"):
                        if col not in csv.columns:
                            raise ValueError(f"Missing required column '{col}' in {csv_path}")

                    # drop rows with NaN in the Euler columns
                    sub = csv[["pitch", "yaw", "roll"]].dropna().to_numpy()
                    pitch = sub[:, 0]
                    yaw = sub[:, 1]
                    roll = sub[:, 2]

                    std_ang_pitch = np.std(pitch)

                    x_values.append(entry[f"{key_name}.started"])
                    y_values.append(std_ang_pitch)
                else:
                    print(f"{csv_path} not found!")
        return x_values, y_values


class HeadYawVariationExtractor:
    name = "head_yaw_variation"
    base_path = os.path.join("..", "..", "data", "Webcam")

    def process(self, parser):
        x_values = []
        y_values = []
        camera_name = "camera.filename"
        camera_task = parser[camera_name]
        # Extracting values for plotting
        for entry in camera_task:
            key_name = "style_randomizer"
            file_name = entry[camera_name]
            if f"{key_name}.started" in entry:
                _, video_filename = os.path.split(file_name)
                csv_filename = video_filename.replace(".mp4", ".csv")
                csv_path = os.path.join(self.base_path, csv_filename)
                if os.path.exists(csv_path):
                    csv = pd.read_csv(csv_path)
                    # ensure columns exist (robust)
                    for col in ("pitch", "yaw", "roll"):
                        if col not in csv.columns:
                            raise ValueError(f"Missing required column '{col}' in {csv_path}")

                    # drop rows with NaN in the Euler columns
                    sub = csv[["pitch", "yaw", "roll"]].dropna().to_numpy()
                    pitch = sub[:, 0]
                    yaw = sub[:, 1]
                    roll = sub[:, 2]

                    std_ang_yaw = np.std(yaw)

                    x_values.append(entry[f"{key_name}.started"])
                    y_values.append(std_ang_yaw)
                else:
                    print(f"{csv_path} not found!")
        return x_values, y_values


if __name__ == '__main__':
    predictor_definitions = [
        HeadPoseVariationExtractor(),
        HeadPoseMovementExtractor(),
        HeadPitchVariationExtractor(),
        HeadRollVariationExtractor(),
        HeadYawVariationExtractor(),
        BlinkTimesExtractor(),
        LookDownTimesExtractor()
    ]

    outcome_definition = SleepinessAnswerExtractor()

    processed = {}

    for psydat_file in tqdm(psydat_files[1:]):
        participant_id = int(psydat_file.split("_")[0])
        parser = DataParser(os.path.join("..", "..", "data", psydat_file))
        outcome_times, outcome_values = outcome_definition.process(parser)
        outcome_times, outcome_values = filter_time_series(outcome_times, outcome_values)
        for extractor in predictor_definitions:
            predictor_name = extractor.name
            predictor_times, predictor_values = extractor.process(parser)
            predictor_associated_values = []
            predictor_associated_vars = []
            predictor_times = np.array(predictor_times)
            predictor_values = np.array(predictor_values)
            for outcome_time in outcome_times:
                associated_values = predictor_values[(predictor_times > outcome_time - SMOOTH_WINDOW) & (predictor_times <= outcome_time)]
                predictor_associated_values.append(np.mean(associated_values))
                predictor_associated_vars.append(np.std(associated_values))
            if f"{predictor_name}_mean" not in processed:
                processed[f"{predictor_name}_mean"] = []
            processed[f"{predictor_name}_mean"].extend(predictor_associated_values)
            # if f"{predictor_name}_var" not in processed:
            #     processed[f"{predictor_name}_var"] = []
            # processed[f"{predictor_name}_var"].extend(predictor_associated_vars)
        if "time" not in processed:
            processed["time"] = []
        processed["time"].extend(outcome_times)
        if outcome_definition.name not in processed:
            processed[outcome_definition.name] = []
        processed[outcome_definition.name].extend(outcome_values)
        if "participant" not in processed:
            processed["participant"] = []
        processed["participant"].extend([participant_id] * len(outcome_values))

    processed = filter_nan_indices(processed)

    # Convert processed dictionary to DataFrame
    df = pd.DataFrame(processed)

    save_directory = os.path.join("processed_data", "webcam")

    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    # Save to CSV file
    df.to_csv(os.path.join(save_directory, f"{len(psydat_files)}-{outcome_definition.name}.csv"), index=False)

    print("Processed data saved.")

