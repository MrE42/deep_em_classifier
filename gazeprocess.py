import os
import logging
import subprocess
import tempfile
import pandas as pd

logging.basicConfig(level=logging.INFO)

def process_gaze_data(gaze_data):
    # Step 1: Prepare gaze data in .arff format
    arff_file_path, vt = convert_gaze_data_to_arff(gaze_data)

    # Step 2: Extract features using MATLAB script (if needed)
    features_arff_path = extract_features_from_arff(arff_file_path)

    # Step 3: Run the BLSTM model on the extracted features
    output_arff_path = run_blstm_model(features_arff_path)

    # Step 4: Parse the output ARFF to classify fixations, saccades, and smooth pursuits
    fixations, saccades, smooth_pursuits, total = parse_output_arff(output_arff_path, vt)

    return fixations, saccades, smooth_pursuits, total

def convert_gaze_data_to_arff(gaze_data):
    vt = []
    # Create .arff file from gaze data
    temp_arff = tempfile.NamedTemporaryFile(delete=False, suffix='.arff')
    with open(temp_arff.name, 'w') as arff_file:
        arff_file.write("@RELATION gaze_labels\n")
        arff_file.write("\n")
        arff_file.write("%@METADATA width_px 1280.0\n")
        arff_file.write("%@METADATA height_px 720.0\n")
        arff_file.write("%@METADATA width_mm 400.0\n")
        arff_file.write("%@METADATA height_mm 225.0\n")
        arff_file.write("%@METADATA distance_mm 450.0\n")
        arff_file.write("\n")
        arff_file.write("@ATTRIBUTE time NUMERIC\n")
        arff_file.write("@ATTRIBUTE x NUMERIC\n")
        arff_file.write("@ATTRIBUTE y NUMERIC\n")
        arff_file.write("@ATTRIBUTE confidence NUMERIC\n")
        arff_file.write("@DATA\n")

        for point in gaze_data:
            arff_file.write(f"{point['timestamp'] * 1000},{point['x']},{point['y']},1\n")
            vt.append(point['videoTime'])

    return temp_arff.name, vt

def extract_features_from_arff(arff_file_path):
    # Call MATLAB script to extract features
    logging.info("Features generating.")
    features_arff = tempfile.NamedTemporaryFile(delete=False, suffix='.arff')
    logging.info(f"{arff_file_path}, {features_arff.name}")
    subprocess.run(['matlab', '-batch', f"AnnotateData('{arff_file_path}', '{features_arff.name}')"])
    logging.info("Features generated.")
    return features_arff.name

def run_blstm_model(arff_file_path):
    # Run the BLSTM model using the pre-trained model
    logging.info("Running BLSTM.")
    output_arff = tempfile.NamedTemporaryFile(delete=False, suffix='.arff')
    model_path = "example_data/model.h5"
    subprocess.run([
        "venv\Scripts\python.exe", 'blstm_model_run.py',
        '--feat', 'speed', 'direction',
        '--model', model_path,
        '--in', arff_file_path,
        '--out', output_arff.name,
    ])
    logging.info("BLTSM complete.")
    return output_arff.name

def parse_output_arff(output_arff_path, vt):
    # Read the ARFF file and extract classifications
    logging.info("Parsing output.")
    fixations = []
    saccades = []
    smooth_pursuits = []
    total = []
    with open(output_arff_path, 'r') as arff_file:
        data_section = False
        for line in arff_file:
            logging.info(f"{line}")
            if line.strip().lower() == "@data":
                data_section = True
                continue
            if data_section and not line.startswith('%'):
                fields = line.strip().split(',')
                timestamp, x, y, c, s1, d1, a1, s2, d2, a2, s4, d4, a4, s8, d8, a8, s16, d16, a16, classification = fields
                event = {'videoTime': float(vt.pop(0)), 'x': float(x), 'y': float(y), 'classification': str(classification)}

                if classification == 'FIX':
                    fixations.append(event)
                elif classification == 'SACCADE':
                    saccades.append(event)
                elif classification == 'SP':
                    smooth_pursuits.append(event)
                total.append(event)
    logging.info("Output parsed.")
    return fixations, saccades, smooth_pursuits, total

if __name__ == "__main__":

    # Paths for video and data
    unprocessed_gaze_csv_path = "unprocessed_gaze_data.csv"

    # Save unprocessed gaze data to CSV
    # pd.DataFrame(gaze_data).to_csv(unprocessed_gaze_csv_path, index=False)
    gaze_data = pd.read_csv(unprocessed_gaze_csv_path).to_dict(orient='records')
    logging.info(f"Received {len(gaze_data)} gaze points.")

    # Process gaze data to classify fixations, saccades, and smooth pursuit
    fixations, saccades, smooth_pursuits, total = process_gaze_data(gaze_data)


    # Save processed gaze data to CSV
    processed_data = total
    pd.DataFrame(processed_data).to_csv("processed_gaze_data.csv", index=False)