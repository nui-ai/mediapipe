import os
import sys

# pb2 classes are generated code, generated from their original .proto definitions during the pip install,
# by our setup.py, into the target venv's paths. they are therefore not known to the IDE.
from mediapipe.examples.desktop import pipeline_output_pb2
from mediapipe.framework.formats import landmark_pb2, classification_pb2, rect_pb2, detection_pb2
from google.protobuf.json_format import MessageToDict

CURRENT = "output_data_python.pb"
REFERENCE = "output_data_v0.10.13.pb"  # "output_data_python.pb"

# Helper to read a varint from a file
def read_varint(file):
    shift = 0
    result = 0
    while True:
        b = file.read(1)
        if not b:
            return None  # EOF
        i = b[0]
        result |= ((i & 0x7F) << shift)
        if not (i & 0x80):
            break
        shift += 7
    return result

# Helper to read a stream of delimited PipelineOutputData messages from a file.
# Each message is written as: [varint length][message bytes].
# no python library implementation for this in python https://chatgpt.com/s/t_68de7b5db2bc8191a472dc39f7af2c4a
# (only in cpp, so we roll our chatgpt own for python)
def read_pipeline_output_data(filename):
    messages = []
    with open(filename, "rb") as f:
        while True:
            length = read_varint(f)
            if length is None:
                break
            data = f.read(length)
            if len(data) < length:
                print(f"Error: expected {length} bytes, got {len(data)} bytes. File may be truncated.")
                break
            msg = pipeline_output_pb2.PipelineOutputData()
            try:
                msg.ParseFromString(data)
                messages.append(msg)
            except Exception as e:
                print(f"Error parsing message at offset {f.tell() - length}: {e}")
                break
            else:
                pass
                # print(f'frame number is {msg.frame_number}')
    return messages

def compare_messages(msg1, msg2):
    dict1 = MessageToDict(msg1, preserving_proto_field_name=True)
    dict2 = MessageToDict(msg2, preserving_proto_field_name=True)
    differences = {}
    all_keys = set(dict1.keys()) | set(dict2.keys())
    for key in all_keys:
        if dict1.get(key) != dict2.get(key):
            differences[key] = {'reference': dict1.get(key), 'python': dict2.get(key)}
    return differences

def main():
    print("note: the c++ equivalent main provides an easier to consume elucidation of the same differences. try it.\n")
    if not os.path.exists(CURRENT) or not os.path.exists(REFERENCE):
        print(f"Missing required files: {CURRENT} or {REFERENCE}")
        sys.exit(1)
    py_msgs = read_pipeline_output_data(REFERENCE)
    reference_msgs = read_pipeline_output_data(CURRENT)
    n_reference = len(reference_msgs)
    n_py = len(py_msgs)
    print(f"Read {n_reference} records from {CURRENT}")
    print(f"Read {n_py} records from {REFERENCE}")
    n = min(n_reference, n_py)
    all_equal = True
    different_pairs = 0
    for i in range(n):
        if reference_msgs[i] == py_msgs[i]:
            continue
        all_equal = False
        different_pairs += 1
        print(f"Record {i+1} differs:")
        diffs = compare_messages(reference_msgs[i], py_msgs[i])
        for field, vals in diffs.items():
            print(f"  Field '{field}': reference={vals['reference']} python={vals['python']}")
    if n_reference != n_py:
        print(f"Number of records differ: reference={n_reference}, python={n_py}")
        all_equal = False
    if all_equal:
        print("All records are equal.")
    else:
        print(f"{different_pairs} records are different.")

if __name__ == "__main__":
    main()
