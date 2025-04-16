#!/usr/bin/env python3
# -----------------------------------------------------------------------------
#  mm5_ls_utils.py
#
#  LICENSE:
#    This file is distributed under the Creative Commons
#    Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0).
#    https://creativecommons.org/licenses/by-nc/4.0/
#
#    You are free to:
#      • Share — copy and redistribute the material in any medium or format
#      • Adapt — remix, transform, and build upon the material
#    under the following terms:
#      • Attribution — You must give appropriate credit, provide a link to
#        the license, and indicate if changes were made.
#      • NonCommercial — You may not use the material for commercial purposes.
#
#  SOURCES & ATTRIBUTIONS:
#    Portions of this code are adapted or derived from the following:
#      1) "Original RLE JS code" from thi-ng/umbrella:
#         https://github.com/thi-ng/umbrella/blob/develop/packages/rle-pack/src/index.ts
#      2) label-studio-converter:
#         https://github.com/HumanSignal/label-studio-converter/blob/master/label_studio_converter/brush.py
#      3) StackOverflow answer:
#         https://stackoverflow.com/a/32681075/6051733
#
#    Such portions may be subject to their respective licenses, as found in the
#    original repositories or sources. The maintainers of this file make no claims
#    of granting any additional rights beyond what those original licenses allow.
#
#  DISCLAIMER:
#    This code is provided “AS IS,” without warranties or conditions of any kind,
#    express or implied. Use it at your own risk.
# -----------------------------------------------------------------------------
import cv2
import numpy as np
import json
from typing import List
import requests
import math
from label_studio_sdk.client import LabelStudio
from PIL import Image as PILImage
from skimage.segmentation import random_walker

# from PIL import Image as PILImage
# from IPython.display import display
# import io
# from skimage.segmentation import random_walker
# import glob
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches
# from pycocotools.mask import encode
# import pycocotools.mask as maskUtils  # Import maskUtils for RLE encoding
# import base64
# from scipy import ndimage

"""
Original RLE JS code from https://github.com/thi-ng/umbrella/blob/develop/packages/rle-pack/src/index.ts
Code taken from https://github.com/HumanSignal/label-studio-converter/blob/master/label_studio_converter/brush.py
"""
def is_clockwise(contour):
    value = 0
    num = len(contour)
    for i, point in enumerate(contour):
        p1 = contour[i]
        if i < num - 1:
            p2 = contour[i + 1]
        else:
            p2 = contour[0]
        value += (p2[0][0] - p1[0][0]) * (p2[0][1] + p1[0][1])
    return value < 0


def get_merge_point_idx(contour1, contour2):
    idx1 = 0
    idx2 = 0
    distance_min = -1
    for i, p1 in enumerate(contour1):
        for j, p2 in enumerate(contour2):
            distance = pow(p2[0][0] - p1[0][0], 2) + pow(p2[0][1] - p1[0][1], 2)
            if distance_min < 0:
                distance_min = distance
                idx1 = i
                idx2 = j
            elif distance < distance_min:
                distance_min = distance
                idx1 = i
                idx2 = j
    return idx1, idx2


def merge_contours(contour1, contour2, idx1, idx2):
    contour = []
    for i in list(range(0, idx1 + 1)):
        contour.append(contour1[i])
    for i in list(range(idx2, len(contour2))):
        contour.append(contour2[i])
    for i in list(range(0, idx2 + 1)):
        contour.append(contour2[i])
    for i in list(range(idx1, len(contour1))):
        contour.append(contour1[i])
    contour = np.array(contour)
    return contour


def merge_with_parent(contour_parent, contour):
    if not is_clockwise(contour_parent):
        contour_parent = contour_parent[::-1]
    if is_clockwise(contour):
        contour = contour[::-1]
    idx1, idx2 = get_merge_point_idx(contour_parent, contour)
    return merge_contours(contour_parent, contour, idx1, idx2)


def mask_to_polygon(image):
    contours, hierarchies = cv2.findContours(image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_KCOS)
    if len(contours) == 0:
        return []
    contours_approx = []
    for contour in contours:
        epsilon = 0.001 * cv2.arcLength(contour, True)
        contour_approx = cv2.approxPolyDP(contour, epsilon, True)
        contours_approx.append(contour_approx)
    contours_parent = []
    for i, contour in enumerate(contours_approx):
        parent_idx = hierarchies[0][i][3]
        if parent_idx < 0 and len(contour) >= 3:
            contours_parent.append(contour)
        else:
            contours_parent.append([])
    for i, contour in enumerate(contours_approx):
        parent_idx = hierarchies[0][i][3]
        if parent_idx >= 0 and len(contour) >= 3:
            contour_parent = contours_parent[parent_idx]
            if len(contour_parent) == 0:
                continue
            contours_parent[parent_idx] = merge_with_parent(contour_parent, contour)
    contours_parent_tmp = []
    for contour in contours_parent:
        if len(contour) == 0:
            continue
        contours_parent_tmp.append(contour)
    polygons = []
    for contour in contours_parent_tmp:
        polygon = contour.flatten().tolist()
        polygons.append(polygon)
    return polygons


class InputStream:
    def __init__(self, data):
        self.data = data
        self.i = 0

    def read(self, size):
        out = self.data[self.i:self.i + size]
        self.i += size
        return int(out, 2)


def access_bit(data, num):
    """from bytes array to bits by num position"""
    base = int(num // 8)
    shift = 7 - int(num % 8)
    return (data[base] & (1 << shift)) >> shift


def bytes2bit(data):
    """get bit string from bytes data"""
    return ''.join([str(access_bit(data, i)) for i in range(len(data) * 8)])


def bits2byte(arr_str, n=8):
    """Convert bits back to byte

    :param arr_str:  string with the bit array
    :type arr_str: str
    :param n: number of bits to separate the arr string into
    :type n: int
    :return rle:
    :type rle: list
    """
    rle = []
    numbers = [arr_str[i : i + n] for i in range(0, len(arr_str), n)]
    for i in numbers:
        rle.append(int(i, 2))
    return rle

# to visualise the random_walker areas for label growth
def create_colored_map(seeds, masked_edges):
    # Create a blank 3-channel image
    colored_map = np.zeros((seeds.shape[0], seeds.shape[1], 3), dtype=np.uint8)
    
    # Assign colors
    red = [0, 0, 255]   # Grow area (seeds == 0)
    green = [0, 255, 0] # Seed area (seeds == 2)
    blue = [255, 0, 0]  # Masked edges
    black = [0, 0, 0]   # Block area (seeds == 1)
    
    # Assign colors to the 3-channel image based on the masks
    colored_map[seeds == 0] = red    # Grow area
    colored_map[seeds == 2] = green  # Seed area
    colored_map[seeds == 1] = black  # Block area
    colored_map[masked_edges == 255] = blue  # Masked edges

    return colored_map
    
def rle_to_mask(rle: List[int], height: int, width: int) -> np.array:
    """
    Converts rle to image mask
    Args:
        rle: your long rle
        height: original_height
        width: original_width

    Returns: np.array
    """

    rle_input = InputStream(bytes2bit(rle))

    num = rle_input.read(32)
    word_size = rle_input.read(5) + 1
    rle_sizes = [rle_input.read(4) + 1 for _ in range(4)]
    # print('RLE params:', num, 'values,', word_size, 'word_size,', rle_sizes, 'rle_sizes')

    i = 0
    out = np.zeros(num, dtype=np.uint8)
    while i < num:
        x = rle_input.read(1)
        j = i + 1 + rle_input.read(rle_sizes[rle_input.read(2)])
        if x:
            val = rle_input.read(word_size)
            out[i:j] = val
            i = j
        else:
            while i < j:
                val = rle_input.read(word_size)
                out[i] = val
                i += 1

    image = np.reshape(out, [height, width, 4])[:, :, 3]
    return image
    

# Taken from https://stackoverflow.com/a/32681075/6051733
def base_rle_encode(inarray):
    """run length encoding. Partial credit to R rle function.
    Multi datatype arrays catered for including non Numpy
    returns: tuple (runlengths, startpositions, values)"""
    ia = np.asarray(inarray)  # force numpy
    n = len(ia)
    if n == 0:
        return None, None, None
    else:
        y = ia[1:] != ia[:-1]  # pairwise unequal (string safe)
        i = np.append(np.where(y), n - 1)  # must include last element posi
        z = np.diff(np.append(-1, i))  # run lengths
        p = np.cumsum(np.append(0, z))[:-1]  # positions
        return z, p, ia[i]
#

def encode_rle(arr, wordsize=8, rle_sizes=[3, 4, 8, 16]):
    """Encode a 1d array to rle


    :param arr: flattened np.array from a 4d image (R, G, B, alpha)
    :type arr: np.array
    :param wordsize: wordsize bits for decoding, default is 8
    :type wordsize: int
    :param rle_sizes:  list of ints which state how long a series is of the same number
    :type rle_sizes: list
    :return rle: run length encoded array
    :type rle: list

    """
    # Set length of array in 32 bits
    num = len(arr)
    numbits = f'{num:032b}'

    # put in the wordsize in bits
    wordsizebits = f'{wordsize - 1:05b}'

    # put rle sizes in the bits
    rle_bits = ''.join([f'{x - 1:04b}' for x in rle_sizes])

    # combine it into base string
    base_str = numbits + wordsizebits + rle_bits

    # start with creating the rle bite string
    out_str = ''
    for length_reeks, p, value in zip(*base_rle_encode(arr)):
        # TODO: A nice to have but --> this can be optimized but works
        if length_reeks == 1:
            # we state with the first 0 that it has a length of 1
            out_str += '0'
            # We state now the index on the rle sizes
            out_str += '00'

            # the rle size value is 0 for an individual number
            out_str += '000'

            # put the value in a 8 bit string
            out_str += f'{value:08b}'
            state = 'single_val'

        elif length_reeks > 1:
            state = 'series'
            # rle size = 3
            if length_reeks <= 8:
                # Starting with a 1 indicates that we have started a series
                out_str += '1'

                # index in rle size arr
                out_str += '00'

                # length of array to bits
                out_str += f'{length_reeks - 1:03b}'

                out_str += f'{value:08b}'

            # rle size = 4
            elif 8 < length_reeks <= 16:
                # Starting with a 1 indicates that we have started a series
                out_str += '1'
                out_str += '01'

                # length of array to bits
                out_str += f'{length_reeks - 1:04b}'

                out_str += f'{value:08b}'

            # rle size = 8
            elif 16 < length_reeks <= 256:
                # Starting with a 1 indicates that we have started a series
                out_str += '1'

                out_str += '10'

                # length of array to bits
                out_str += f'{length_reeks - 1:08b}'

                out_str += f'{value:08b}'

            # rle size = 16 or longer
            else:
                length_temp = length_reeks
                while length_temp > 2**16:
                    # Starting with a 1 indicates that we have started a series
                    out_str += '1'

                    out_str += '11'
                    out_str += f'{2 ** 16 - 1:016b}'

                    out_str += f'{value:08b}'
                    length_temp -= 2**16

                # Starting with a 1 indicates that we have started a series
                out_str += '1'

                out_str += '11'
                # length of array to bits
                out_str += f'{length_temp - 1:016b}'

                out_str += f'{value:08b}'

    # make sure that we have an 8 fold lenght otherwise add 0's at the end
    nzfill = 8 - len(base_str + out_str) % 8
    total_str = base_str + out_str
    total_str = total_str + nzfill * '0'

    rle = bits2byte(total_str)

    return rle


def contour2rle(contours, contour_id, img_width, img_height):
    """
    :param contours:  list of contours
    :type contours: list
    :param contour_id: id of contour which you want to translate
    :type contour_id: int
    :param img_width: image shape width
    :type img_width: int
    :param img_height: image shape height
    :type img_height: int
    :return: list of ints in RLE format
    """
    import cv2  # opencv

    mask_im = np.zeros((img_width, img_height, 4))
    mask_contours = cv2.drawContours(
        mask_im, contours, contour_id, color=(0, 255, 0, 100), thickness=-1
    )
    rle_out = encode_rle(mask_contours.ravel().astype(int))
    return rle_out


def mask2rle(mask):
    """Convert mask to RLE

    :param mask: uint8 or int np.array mask with len(shape) == 2 like grayscale image
    :return: list of ints in RLE format
    """
    assert len(mask.shape) == 2, 'mask must be 2D np.array'
    assert mask.dtype == np.uint8 or mask.dtype == int, 'mask must be uint8 or int'
    array = mask.ravel()
    array = np.repeat(array, 4)  # must be 4 channels
    rle = encode_rle(array)
    return rle

def subtract_annotation_in_label_studio(
    server_url,
    ls_api_key,
    project_id,
    task_id,
    label_a_meta_key,  # Label A: the annotation to update (the mask from which we subtract)
    label_b_meta_key,  # Label B: the annotation whose mask area will be removed from label A
    image_shape        # Tuple for the mask dimensions
):
    """
    Subtracts the mask of label B from label A (both in RLE format) and updates label A's annotation.
    """
    # Setup API details
    api_url = server_url + '/api/'
    headers = {
        'Authorization': 'Token ' + ls_api_key,
        'Content-Type': 'application/json'
    }
    
    # Initialize Label Studio client, project, and task
    client = LabelStudio(base_url=server_url, api_key=ls_api_key)
    project = client.projects.get(project_id)
    task = client.tasks.get(task_id)
    
    # Retrieve annotations for both labels from the task
    label_a_result = None
    label_b_result = None
    for annotation in task.annotations:
        results = annotation.result if hasattr(annotation, 'result') else []
        for result in results:
            if result.get('id') == label_a_meta_key:
                label_a_result = result
            elif result.get('id') == label_b_meta_key:
                label_b_result = result
        if label_a_result and label_b_result:
            break

    if not label_a_result:
        print(f"No annotation with label {label_a_meta_key} found in task {task_id}")
        return
    if not label_b_result:
        print(f"No annotation with label {label_b_meta_key} found in task {task_id}")
        return

    # Ensure both annotations use RLE format
    if label_a_result.get('value', {}).get('format') != 'rle' or label_b_result.get('value', {}).get('format') != 'rle':
        print("One or both annotations are not in 'rle' format.")
        return

    # Retrieve the RLE lists from the annotations
    rle_a = label_a_result.get('value', {}).get('rle')
    rle_b = label_b_result.get('value', {}).get('rle')

    try:
        # Decode the RLE masks using your helper function.
        mask_a = rle_to_mask(rle_a, image_shape[0], image_shape[1])
        mask_b = rle_to_mask(rle_b, image_shape[0], image_shape[1])
    except Exception as e:
        print("Error decoding RLE:", e)
        return

    # Subtract mask_b from mask_a.
    # Here, wherever mask_b has a nonzero value, we remove that pixel from mask_a.
    new_mask = np.where(mask_b > 0, 0, mask_a)

    # Encode the new mask back into RLE using your helper function.
    try:
        new_rle = mask2rle(new_mask)
    except Exception as e:
        print("Error encoding new mask to RLE:", e)
        return

    # Update label A's annotation with the new RLE mask.
    updated_result = label_a_result.copy()
    updated_result['value']['rle'] = new_rle

    # Locate the annotation that contains label A so we can update it.
    annotation_to_update = None
    for annotation in task.annotations:
        if hasattr(annotation, 'result') and annotation.result:
            for idx, res in enumerate(annotation.result):
                if res.get('id') == label_a_meta_key:
                    annotation_to_update = annotation
                    break
        if annotation_to_update:
            break

    try:
        if annotation_to_update:
            # Replace the existing label A result with the updated one.
            current_results = annotation_to_update.result.copy()
            for idx, res in enumerate(current_results):
                if res.get('id') == label_a_meta_key:
                    current_results[idx] = updated_result
                    break
            payload = {"result": current_results}
            update_url = f'{api_url}annotations/{annotation_to_update.id}/'
            response = requests.patch(update_url, json=payload, headers=headers)
            response.raise_for_status()
            print("Annotation updated successfully with subtracted mask.")
        else:
            # If no annotation was found to update, create a new annotation.
            payload = {"task": task_id, "result": [updated_result]}
            create_url = f'{api_url}tasks/{task_id}/annotations/'
            response = requests.post(create_url, json=payload, headers=headers)
            response.raise_for_status()
            print("Annotation created successfully with subtracted mask.")
    except requests.exceptions.HTTPError as e:
        print("HTTP Error:", e)
        print("Response status:", response.status_code)
        print("Response text:", response.text)
    except json.decoder.JSONDecodeError:
        print("Failed to decode JSON:", response.text)
    except Exception as e:
        print("An error occurred:", e)
        
import requests

def get_ml_backend_id_for_project(ls_base_url, ls_api_key, project_id, debug=False):
    """
    Get the first ML backend for a specific project by making a GET request to /api/ml,
    passing the project ID as a query parameter.

    :param ls_base_url: str, e.g. "http://localhost:8080"
    :param ls_api_key:  str, your Label Studio API key
    :param project_id:  int, the project ID
    :param debug:       bool, if True prints debug info
    :return:            ML backend id, or None if error
    """
    
    # e.g. http://localhost:8080/api/ml?project=2
    url = f"{ls_base_url}/api/ml?project={project_id}"
    headers = {
        "Authorization": f"Token {ls_api_key}"
    }
    if debug:
        print(f"[DEBUG] GET {url}")
    
    try:
        resp = requests.get(url, headers=headers)
        resp.raise_for_status()
    except requests.HTTPError as e:
        print(f"HTTP Error listing ML backends for project {project_id}: {e.response.status_code} {e.response.text}")
        return None
    except Exception as e:
        print(f"Error listing ML backends for project {project_id}: {e}")
        return None
    
    data = resp.json()
    if debug:
        print(f"[DEBUG] ML backends for project {project_id}: {data}")

    if isinstance(data, list) and len(data) > 0:
        # Return the 'id' of the first backend
        first_backend = data[0]
        return first_backend.get("id")

    return None

        
def copy_annotation_to_label_studio(
    server_url,
    ls_api_key,
    project_id,
    source_task_id,
    source_label_meta_key,
    target_task_id,
    target_label_meta_key
):
    # Setup API details
    api_url = server_url + '/api/'
    headers = {
        'Authorization': 'Token ' + ls_api_key,
        'Content-Type': 'application/json'
    }
    
    # Initialize the client
    client = LabelStudio(
        base_url=server_url,
        api_key=ls_api_key
    )
    
    # Retrieve the project (required for tasks belonging to the project)
    project = client.projects.get(project_id)
    
    # Retrieve the source and target tasks
    source_task = client.tasks.get(source_task_id)
    target_task = client.tasks.get(target_task_id)
    
    # -----------------------------
    # EXTRACT THE DESIRED ANNOTATION RESULT FROM THE SOURCE TASK
    # -----------------------------
    extracted_result = None
    for annotation in source_task.annotations:
        # Use attribute access: assume annotation.result is a list of dicts.
        results = annotation.result if hasattr(annotation, 'result') else []
        for result in results:
            # We assume the result's 'id' is the label meta key.
            if result.get('id') == source_label_meta_key:
                extracted_result = result
                break
        if extracted_result:
            break

    if not extracted_result:
        print(f"No annotation with label {source_label_meta_key} found in task {source_task_id}")
        return

    # Prepare a new result by copying the extracted result and updating its 'id'
    new_result = extracted_result.copy()
    new_result['id'] = target_label_meta_key  # Update to the target label meta key

    # -----------------------------
    # UPDATE THE TARGET TASK'S ANNOTATION, MERGING THE NEW RESULT
    # -----------------------------
    # Retrieve existing annotations for the target task.
    annotation_to_update = None
    for annotation in target_task.annotations:
        if hasattr(annotation, 'result') and annotation.result:
            first_result = annotation.result[0]
            if 'value' in first_result:  # heuristic to determine a "manual" annotation
                annotation_to_update = annotation
                break

    try:
        if annotation_to_update:
            # Merge the new result into the existing results list.
            current_results = annotation_to_update.result.copy()  # copy existing list of results
            found = False
            for idx, res in enumerate(current_results):
                if res.get('id') == target_label_meta_key:
                    # Update the matching result.
                    current_results[idx] = new_result
                    found = True
                    break
            if not found:
                # Append the new result if it wasn't found.
                current_results.append(new_result)

            payload = {"result": current_results}
            update_url = f'{api_url}annotations/{annotation_to_update.id}/'
            response = requests.patch(update_url, json=payload, headers=headers)
            response.raise_for_status()
            print(f"Annotation updated successfully")
        else:
            # No existing annotation, so create a new one with the new result.
            payload = {"task": target_task_id, "result": [new_result]}
            create_url = f'{api_url}tasks/{target_task_id}/annotations/'
            response = requests.post(create_url, json=payload, headers=headers)
            response.raise_for_status()
            print(f"Annotation created successfully")
    except requests.exceptions.HTTPError as e:
        print("HTTP Error:", e)
        print("Response status:", response.status_code)
        print("Response text:", response.text)
    except json.decoder.JSONDecodeError:
        print("Failed to decode JSON:", response.text)
    except Exception as e:
        print("An error occurred:", e)
        
        
        
def find_task_id_by_label_id(client, project_id, target_label_id):
    """
    Search for a task in a given project that contains an annotation with the specified unique label ID.
    
    Parameters:
      client: Label Studio client instance.
      project_id (int): The project ID to search in.
      target_label_id (str): The unique label ID to look for.
    
    Returns:
      The task ID if found, else None.
    """
    tasks = client.tasks.list(project=project_id, fields='all')
    for task in tasks:
        # Ensure we can access the annotation results
        results = task.annotations if hasattr(task, 'annotations') else []
        for annotation in results:
            # If annotation is not a dictionary, try converting it.
            if not isinstance(annotation, dict):
                try:
                    annotation = annotation.__dict__
                except Exception:
                    continue
            for res in annotation.get("result", []):
                if res.get("id") == target_label_id:
                    return task.id
    return None

def find_task_and_annotation_by_label_id(
    client,
    project_id,
    target_label_id,
    debug=False
):
    """
    Searches through tasks in the specified project for an annotation result
    that includes the given unique label ID. Uses fallback logic to find 'result' 
    among possible keys and prints debug info if debug=True.

    :param client:        Label Studio client instance
    :param project_id:    The project ID to search in
    :param target_label_id: The unique label ID to look for (e.g. "297_5_4")
    :param debug:         bool, if True print debug statements
    :return: (task_id, annotation_id) if found, else (None, None)
    """
    tasks = client.tasks.list(project=project_id, fields='all')
    if not tasks:
        if debug:
            print(f"[DEBUG] No tasks found in project {project_id}")
        return (None, None)

    for task in tasks:
        annotations = task.annotations if hasattr(task, 'annotations') else []
        if debug:
            print(f"[DEBUG] Task ID={task.id} has {len(annotations)} annotation(s). Checking...")

        for annotation_obj in annotations:
            # Convert to dict if not already
            if not isinstance(annotation_obj, dict):
                annotation_obj = annotation_obj.__dict__

            # Potential annotation ID or PK
            annotation_id = annotation_obj.get('id') or annotation_obj.get('pk')
            if debug:
                print(f"[DEBUG] Annotation keys: {list(annotation_obj.keys())}, annotation_id={annotation_id}")

            # A fallback logic to find 'result'
            results = annotation_obj.get('result', [])
            if not results:
                # Some LSD versions store data in 'data' or 'serialized_data'
                data_node = annotation_obj.get('data', {})
                if isinstance(data_node, dict) and 'result' in data_node:
                    results = data_node['result']
                    if debug:
                        print(f"[DEBUG] Found annotation results under annotation_obj['data']['result'] with {len(results)} items.")
                if not results and 'serialized_data' in annotation_obj:
                    ser_node = annotation_obj['serialized_data']
                    if isinstance(ser_node, dict) and 'result' in ser_node:
                        results = ser_node['result']
                        if debug:
                            print(f"[DEBUG] Found annotation results under annotation_obj['serialized_data']['result'] with {len(results)} items.")
            
            if debug and not results:
                print("[DEBUG] Could not find 'result' array in this annotation object. Skipping.")
                # Proceed to next annotation
                continue

            # Now loop over the results array
            for res in results:
                # If the result's 'id' matches our target label
                if res.get("id") == target_label_id:
                    if debug:
                        print(f"[DEBUG] Found target_label_id={target_label_id} in task {task.id}, annotation {annotation_id}")
                    return (task.id, annotation_id)

    if debug:
        print(f"[DEBUG] No task/annotation found for label {target_label_id}")
    return (None, None)


def find_task_and_annotation_by_image_id(client, project_id, img_id_val):
    """
    Searches the given project for a task whose data['IMG_ID'] == img_id_val,
    then returns the first annotation in that task that matches your snippet logic:
      - 'result' is non-empty
      - the first item in 'result' contains 'value'
    Returns (task_id, annotation_id) if found, else (None, None).
    """

    tasks = client.tasks.list(project=project_id, fields='all')
    if not tasks:
        print(f"No tasks found in project {project_id}")
        return (None, None)

    # Step 1: locate the single task for the given IMG_ID
    target_task = None
    for t in tasks:
        if 'IMG_ID' in t.data:
            try:
                if int(t.data['IMG_ID']) == img_id_val:
                    target_task = t
                    break
            except ValueError:
                pass

    if not target_task:
        # print(f"No task found for img_id={img_id_val}")
        return (None, None)

    # Step 2: among that task's annotations,
    # pick the first annotation that meets your snippet logic
    annotation_to_update = None
    # Each annotation might be a dict or LSD4 object
    annotations = target_task.annotations if hasattr(target_task, 'annotations') else []
    for annotation in annotations:
        if not isinstance(annotation, dict):
            annotation = annotation.__dict__  # Convert LSD object to dict

        results = annotation.get('result', [])
        if not results:
            continue

        first_result = results[0]
        if first_result and 'value' in first_result:
            annotation_to_update = annotation
            break

    if not annotation_to_update:
        print(f"No annotation in task {target_task.id} has a 'value' in its first result.")
        return (None, None)

    # Step 3: Return (task_id, annotation_id)
    annotation_id = annotation_to_update.get('id')
    return (target_task.id, annotation_id)

def subtract_all_from_label_a(
    ls_base_url,
    ls_api_key,
    project_id,
    label_a,
    debug=False
):
    """
    Searches the project for the task containing the specified brush label ID (label_a),
    then subtracts all other brush label IDs from label_a. Uses the existing
    `subtract_annotation_in_label_studio` function for the actual subtraction step.

    :param ls_base_url: str, e.g. "http://localhost:8080"
    :param ls_api_key:  str, Label Studio API key
    :param project_id:  int, project ID
    :param label_a:     str, the brush label ID from which all others will be subtracted
    :param debug:       bool, if True print debug statements
    :return: None
    """
    import sys
    from label_studio_sdk.client import LabelStudio
    from utils import subtract_annotation_in_label_studio, find_task_id_by_label_id, get_image_shape_for_project

    # 1) Create Label Studio client
    client = LabelStudio(base_url=ls_base_url, api_key=ls_api_key)

    # 2) Find the task that contains label_a
    task_id = find_task_id_by_label_id(client, project_id, label_a)
    if task_id is None:
        if debug:
            print(f"No task found containing label_id={label_a}")
        return

    # 3) Retrieve the task (which includes its annotations)
    try:
        task = client.tasks.get(task_id)
    except Exception as e:
        if debug:
            print(f"Error retrieving task {task_id}: {e}")
        sys.exit(1)

    # 4) Collect all brush label IDs from the task's annotations
    all_brush_label_ids = set()
    for annotation in task.annotations:
        results = annotation.result if hasattr(annotation, 'result') else []
        for result in results:
            # We assume brush labels use type "brushlabels"
            if result.get("type") == "brushlabels":
                label_id_found = result.get("id")
                if label_id_found:
                    all_brush_label_ids.add(label_id_found)

    if debug:
        print(f"Task ID for label_a={label_a}: {task_id}")
        print("Found brush label IDs in task:", all_brush_label_ids)

    # 5) Subtract each label except label_a
    labels_to_subtract = [lbl for lbl in all_brush_label_ids if lbl != label_a]
    if debug:
        print("Label A (target):", label_a)
        print("Label IDs to subtract from Label A:", labels_to_subtract)

    # 6) Determine image shape for the project
    image_shape = get_image_shape_for_project(project_id)

    # 7) Loop over each label_b and call the subtraction function
    for label_b_meta_key in labels_to_subtract:
        if debug:
            print(f"Subtracting label '{label_b_meta_key}' from label '{label_a}' for task {task_id}...")
        subtract_annotation_in_label_studio(
            server_url=ls_base_url,
            ls_api_key=ls_api_key,
            project_id=project_id,
            task_id=task_id,
            label_a_meta_key=label_a,
            label_b_meta_key=label_b_meta_key,
            image_shape=image_shape
        )
    if debug:
        print("All subtractions complete.")

###############################################
# Function to find a task's ID by its IMG_ID
###############################################
def find_task_id_by_img_id(client, project_id, img_id_val):
    """
    Search for a task in the specified project that has data['IMG_ID'] == img_id_val.
    Returns the task ID if found, else None.
    """
    tasks = client.tasks.list(project=project_id, fields='all')
    for t in tasks:
        if 'IMG_ID' in t.data:
            try:
                if int(t.data['IMG_ID']) == img_id_val:
                    return t.id
            except ValueError:
                pass
    return None
   
######################################################################################################    
######################################################################################################
### End of label-studio functions
######################################################################################################

def get_image_shape_for_project(project_id):
    """
    Example helper that returns an (img_width, img_height) tuple
    depending on the project ID.
    """
    if project_id == 1:
        image_shape = (720, 1280)
    elif project_id == 2:
        image_shape = (512, 640)
    else:
        image_shape = (576, 720)

    return image_shape

def display_image_from_array(img_array, title=""):
    if title:
        print(title)  # Print the title if it is not empty
    if img_array.dtype != np.uint8:
        img_array = img_array.astype(np.uint8)  # Ensure the image array is of type uint8
    is_color = len(img_array.shape) >= 3
    if is_color:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
    pil_img = PILImage.fromarray(img_array)
    display(pil_img)  # Directly display PIL image

def convert_bin2gray(binary_map):
    return (binary_map * 255).astype(np.uint8) if binary_map.max() == 1 else binary_map

def convert_mat2greybin(image):
    """
    Converts any image to a binary format with pixel values as 0 or 255.
    
    Args:
    image (numpy.ndarray): Input image which can be in BGR, BGRA, grayscale, or already binary.
    
    Returns:
    numpy.ndarray: Binary image with pixel values only 0 or 255.
    """
    # Convert image to grayscale if it's in BGR or BGRA format
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply thresholding to convert image to binary format
    # Here we use 127 as a fixed threshold for simplicity, but this can be adjusted if needed
    _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    return binary_image
    
def apply_color_to_binary_map(binary_map, color):    
    # Create an empty RGB image
    color_image = np.zeros((binary_map.shape[0], binary_map.shape[1], 3), dtype=np.uint8)

    # Find the indices where the binary map is not zero
    indices = np.where(binary_map != 0)

    # Apply color to the non-zero pixels of the binary map
    color_image[indices[0], indices[1], :] = color  # Ensure color is broadcasted correctly

    # Debugging outputs
    total_pixels = binary_map.size
    non_zero_pixels = len(indices[0])
    zero_pixels = total_pixels - non_zero_pixels
    #print("Image Dimensions:", binary_map.shape)
    #print("Total Pixels:", total_pixels)
    #print("Zero Pixels:", zero_pixels)
    #print("Non-Zero Pixels (Colored):", non_zero_pixels)

    return color_image

def create_inverse_map(map_x, map_y):
    h, w = map_x.shape[:2]
    inv_map_x = np.full((h, w), -1, dtype=np.float32)
    inv_map_y = np.full((h, w), -1, dtype=np.float32)

    # Assign the inverse mapping based on forward map coordinates
    for y in range(h):
        for x in range(w):
            mapped_x, mapped_y = int(map_x[y, x]), int(map_y[y, x])
            if 0 <= mapped_x < w and 0 <= mapped_y < h:
                # Prefer the first mapping encountered (naive approach)
                if inv_map_x[mapped_y, mapped_x] == -1:
                    inv_map_x[mapped_y, mapped_x] = x
                    inv_map_y[mapped_y, mapped_x] = y

    # Fill in gaps by nearest neighbor expansion (useful for discrete label maps)
    fill_gaps(inv_map_x, inv_map_y)

    return inv_map_x, inv_map_y

def fill_gaps(map_x, map_y):
    """Fill gaps in the inverse map using nearest valid values."""
    h, w = map_x.shape
    for y in range(h):
        for x in range(w):
            if map_x[y, x] == -1 or map_y[y, x] == -1:
                # Find nearest valid pixel
                nearest_x, nearest_y = find_nearest_valid_pixel(map_x, map_y, x, y)
                map_x[y, x] = nearest_x
                map_y[y, x] = nearest_y

def find_nearest_valid_pixel(map_x, map_y, x, y):
    """Find the nearest pixel with a valid mapping."""
    h, w = map_x.shape
    search_radius = 1
    while search_radius < max(h, w):
        for dy in range(-search_radius, search_radius + 1):
            ny = y + dy
            if 0 <= ny < h:
                for dx in range(-search_radius, search_radius + 1):
                    nx = x + dx
                    if 0 <= nx < w and map_x[ny, nx] != -1 and map_y[ny, nx] != -1:
                        return map_x[ny, nx], map_y[ny, nx]
        search_radius += 1
    return x, y  # Default to the original coordinates if no valid neighbor is found


def smooth_gt_map(gt_map, kernel_size=3, max_iterations=1):
    """
    Smooth the edges and close gaps in a ground truth map using morphological operations.

    Parameters:
    gt_map (numpy.ndarray): The input ground truth map where different objects are marked with different integer values.
    kernel_size (int): The size of the square kernel used for morphological operations.
    max_iterations (int): The number of times dilation and erosion are applied.

    Returns:
    numpy.ndarray: The smoothed ground truth map.
    """
    # Create the kernel for morphological operations
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # Perform the closing operation multiple times
    for _ in range(max_iterations):
        # Dilation followed by Erosion to close small gaps and smooth edges
        gt_map = cv2.dilate(gt_map, kernel, iterations=1)
        gt_map = cv2.erode(gt_map, kernel, iterations=1)

    return gt_map

def find_center_of_mask(instance_mask):
    """
    Find the center pixel position of the binary mask area (non-zero values).
    
    Args:
    instance_mask (numpy.ndarray): A binary mask where the masked area has non-zero pixel values.
    
    Returns:
    tuple: (center_x, center_y) representing the center pixel position of the mask.
    """
    # Find the indices of non-zero pixels (the mask area)
    y_indices, x_indices = np.nonzero(instance_mask)

    if len(y_indices) == 0 or len(x_indices) == 0:
        return None  # Return None if the mask has no non-zero pixels

    # Calculate the mean of the x and y coordinates to find the center
    center_x = int(np.mean(x_indices))
    center_y = int(np.mean(y_indices))

    return center_x, center_y

# size image down to thermal/UV size
def resize_and_crop(src, trg):
    # Original dimensions of src
    src_height, src_width = src.shape[:2]
    
    # Target dimensions
    trg_height, trg_width = trg.shape[:2]
    
    # Calculate the scaling factor to match the target height
    scaling_factor = src_height / trg_height
    
    # New dimensions for cropping centered area from src
    new_width = int(trg_width * scaling_factor)
    
    # Calculate the starting x coordinate to crop the center
    start_x = (src_width - new_width) // 2
    
    # Crop the centered area
    src_cropped = src[:, start_x:start_x + new_width]
    
    # Resize cropped image back to the target size
    trg_resized = cv2.resize(src_cropped, (trg_width, trg_height), interpolation=cv2.INTER_NEAREST) # INTER_AREA is used for images
#     print(f"Scaling factor: {scaling_factor}")
#     print(f"Original dimensions: ({src_width}, {src_height})")
#     print(f"Target dimensions: ({trg_width}, {trg_height})")
#     print(f"New width for cropping: {new_width}")
#     print(f"Start x-coordinate for cropping: {start_x}")
    return trg_resized
    
def refine_edges_gt_map(gt_map, kernel_size=3, operation='close'):
    """
    Refine edges of a GT map using morphological operations on each channel separately.
    
    Args:
    gt_map (numpy.ndarray): The ground truth map.
    kernel_size (int): Size of the morphological operation kernel.
    operation (str): Type of operation ('open', 'close', 'both').
    
    Returns:
    numpy.ndarray: The refined ground truth map.
    """
    # Create the kernel
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # Check if the image is single-channel or multi-channel
    if len(gt_map.shape) == 3 and gt_map.shape[2] == 3:
        channels = cv2.split(gt_map)
        refined_channels = []

        for ch in channels:
            if operation == 'open':
                refined_ch = cv2.morphologyEx(ch, cv2.MORPH_OPEN, kernel)
            elif operation == 'close':
                refined_ch = cv2.morphologyEx(ch, cv2.MORPH_CLOSE, kernel)
            elif operation == 'both':
                refined_ch = cv2.morphologyEx(ch, cv2.MORPH_OPEN, kernel)
                refined_ch = cv2.morphologyEx(refined_ch, cv2.MORPH_CLOSE, kernel)
            else:
                refined_ch = ch
            refined_channels.append(refined_ch)

        refined_gt_map = cv2.merge(refined_channels)
    else:
        # Perform the chosen morphological operation for single-channel images
        if operation == 'open':
            refined_gt_map = cv2.morphologyEx(gt_map, cv2.MORPH_OPEN, kernel)
        elif operation == 'close':
            refined_gt_map = cv2.morphologyEx(gt_map, cv2.MORPH_CLOSE, kernel)
        elif operation == 'both':
            refined_gt_map = cv2.morphologyEx(gt_map, cv2.MORPH_OPEN, kernel)
            refined_gt_map = cv2.morphologyEx(refined_gt_map, cv2.MORPH_CLOSE, kernel)
        else:
            refined_gt_map = gt_map

    return refined_gt_map
    
def read_calibration_params(filename, ftype):
    fs = cv2.FileStorage(filename, cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise IOError("File not opened: " + filename)

    calibration_data = {}    
    if ftype=="cam":
        matrix_keys = ['CM', 'CMT'] # , 'R', 'T'
        distortion_keys = ['D','R','T']
    else: 
        matrix_keys = ['CM1', 'CM2', 'R', 'E', 'F', 'R1', 'R2', 'P1', 'P2', 'Q']
        distortion_keys = ['D1', 'D2','T']
        
    for key in matrix_keys:
        node = fs.getNode(key)
        if not node.empty():
            calibration_data[key] = node.mat().astype(np.float64)
        else:
            print(f"Warning: {key} is not found or is empty in file.")
    
    # Handling D1 and D2 which are stored as lists of values
    for key in distortion_keys:
        node = fs.getNode(key)
        if not node.empty():
            # Assuming the node contains a list of floats
            distortion_coefficients = []
            if node.isSeq():
                for i in range(node.size()):
                    distortion_coefficients.append(node.at(i).real())
                calibration_data[key] = np.array(distortion_coefficients, dtype=np.float64)
            else:
                print(f"Error: Failed to read data for {key} as sequence")
        else:
            print(f"Warning: {key} is not found or is empty in file.")
    fs.release()
    return calibration_data
    
def ensure_even_dimensions_and_aspect(roi, input_shape):
    x, y, w, h = roi
    aspect_ratio = input_shape[1] / input_shape[0]  # width / height

    # Expand to maintain aspect ratio
    if w % 2 != 0:
        w += 1
    if h % 2 != 0:
        h += 1

    # Adjust width to maintain aspect ratio based on the height
    new_width = int(h * aspect_ratio)
    if new_width % 2 != 0:
        new_width += 1

    # Center the ROI around the original center
    center_x = x + w // 2
    center_y = y + h // 2

    new_x = max(center_x - new_width // 2, 0)
    new_y = max(center_y - h // 2, 0)

    # Ensure the new ROI does not exceed the image dimensions
    new_x = min(new_x, input_shape[1] - new_width)
    new_y = min(new_y, input_shape[0] - h)

    return (new_x, new_y, new_width, h)

def adjust_camera_matrix_for_new_size(camera_matrix, original_size, new_size, good_pixel):
    aspect_ratio_original = original_size[0] / original_size[1]
    aspect_ratio_new = new_size[0] / new_size[1]

    scale = new_size[1] / original_size[1] if aspect_ratio_original > aspect_ratio_new else new_size[0] / original_size[0]

    adjusted_matrix = camera_matrix.copy()
    adjusted_matrix[0, 0] *= scale  # fx
    adjusted_matrix[1, 1] *= scale*1.05  # fy
    adjusted_matrix[0, 2] = (new_size[0] - original_size[0] * scale) / 2 + camera_matrix[0, 2] * scale
    adjusted_matrix[1, 2] = (new_size[1] - original_size[1] * scale) / 2 + camera_matrix[1, 2] * scale

    return adjusted_matrix
    
# size image up to trg size
def resize_thermal_to_kinect_mat(input_image, target_width, target_height):
    """
    Resizes 'input_image' to (target_width, target_height).

    Behavior:
      - Chooses interpolation automatically based on the bit depth:
          * If input_image.dtype == np.uint8, uses INTER_LINEAR (smoother)
          * If input_image.dtype == np.uint16, uses INTER_NEAREST (preserves exact data)
      - Preserves the original number of channels:
          * Single-channel remains (H', W')
          * Multi-channel remains (H', W', channels)
      - Centers the resized image on a black (zero) background if the aspect ratio changes.
    """

    if input_image is None:
        raise ValueError("Input image is not loaded.")
    if len(input_image.shape) < 2:
        raise ValueError("Input image does not have enough dimensions (expected at least 2).")

    # Decide interpolation based on bit depth
    if input_image.dtype == np.uint8:
        interpolation = cv2.INTER_LINEAR   # typical for 8-bit
    elif input_image.dtype == np.uint16:
        interpolation = cv2.INTER_NEAREST  # preserve data for 16-bit
    else:
        interpolation = cv2.INTER_LINEAR   # fallback (could raise an error or handle float32, etc.)

    # Detect if single-channel or multi-channel
    if len(input_image.shape) == 2:
        # Single-channel
        height, width = input_image.shape
        num_channels = 1
    else:
        # Multi-channel (e.g. (H, W, 2), (H, W, 3), or (H, W, 4))
        height, width, num_channels = input_image.shape

    # Maintain aspect ratio
    scale_height = target_height / height
    scale_width = target_width / width
    scale = min(scale_height, scale_width)

    new_height = int(height * scale)
    new_width = int(width * scale)

    # Resize
    # If single-channel, shape is (H, W); if multi-channel, shape is (H, W, c).
    resized_image = cv2.resize(
        input_image,
        (new_width, new_height),
        interpolation=interpolation
    )

    # Prepare output ("borderedImage") with black background
    top_border = (target_height - new_height) // 2
    left_border = (target_width - new_width) // 2

    if num_channels == 1:
        # -- Single-channel case: keep 2D shape (H', W') --
        borderedImage = np.zeros((target_height, target_width), dtype=input_image.dtype)
        borderedImage[:] = 0  # black background
        borderedImage[top_border : top_border + new_height,
                      left_border : left_border + new_width] = resized_image
    else:
        # -- Multi-channel case: shape (H', W', c) --
        # Decide background color: zero per channel
        background_color = [0]*num_channels  # e.g. (0, 0) for 2ch, (0, 0, 0) for 3ch, ...
        borderedImage = np.zeros((target_height, target_width, num_channels), dtype=input_image.dtype)
        borderedImage += np.array(background_color, dtype=input_image.dtype)

        borderedImage[top_border : top_border + new_height,
                      left_border : left_border + new_width, :] = resized_image

    return borderedImage
    
def auto_tone(image, low_clip=0.5, high_clip=99.5, gamma=None, gBase=2.0):
    """
    Apply contrast enhancement directly on a 16-bit grayscale image while preserving the 16-bit depth
    and lighten darker areas using gamma correction.
    
    Args:
    image (numpy.ndarray): Input 16-bit grayscale image.
    low_clip (float): Lower percentile to clip (default 1.5).
    high_clip (float): Upper percentile to clip (default 95).
    gamma (float): Gamma value for gamma correction (default 1.2).
    
    Returns:
    numpy.ndarray: Enhanced 16-bit image, converted to 8-bit.
    """
    # Clip extremes to remove outliers
    lower_bound = np.percentile(image, low_clip)
    upper_bound = np.percentile(image, high_clip)
    img_clipped = np.clip(image, lower_bound, upper_bound)

    # Check for empty or zero-size arrays
    valid_pixels = img_clipped[image > 0]
    if valid_pixels.size == 0:
        print("No valid non-zero pixels found in the image.")
        return np.zeros_like(image, dtype=np.uint8)
        
    # Clip to the min and max of the image
    min_val = np.min(img_clipped[image > 0])  # Ignore zero values for min (assuming 0 is no data)
    max_val = np.max(img_clipped)
    
    # Stretch the histogram: set darkest pixel as 0 and lightest pixel as 65535
    img_stretched = np.clip(image, min_val, max_val)
    
    # Set the lowest pixel value to 0 and the highest to 65535 (full 16-bit range)
    img_stretched = (img_stretched - min_val) * (65535 / (max_val - min_val))
    
    # Calculate gamma dynamically based on image brightness
    if gamma is None:
        mean_intensity = np.mean(img_stretched) / 65535.0  # Normalize to [0, 1] range
        gamma = gBase - 1.5 * mean_intensity  # Higher gamma for dim images, lower for bright ones
    
    
    # Apply gamma correction to lighten darker areas
    inv_gamma = 1.0 / gamma
    img_stretched = ((img_stretched / 65535.0) ** inv_gamma) * 65535
    img_stretched = np.clip(img_stretched, 0, 65535).astype(np.uint16)
    
    # Convert the stretched image to 8-bit
    img_normalized = cv2.normalize(img_stretched, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    return img_normalized
    
def rectify_image(input_image, camera_matrix, dist_coeffs, target_size):
    optimal_new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, input_image.shape[1::-1], 0.2, input_image.shape[1::-1], centerPrincipalPoint=1
    )

    # Ensure even dimensions for ROI and maintain aspect ratio
    x, y, w, h = ensure_even_dimensions_and_aspect(roi, input_image.shape)

    # Undistort the original image - process 8 and 16-bit since lowe-upper byte conversion causes issues
    ud_image = cv2.undistort(input_image, camera_matrix, dist_coeffs, None, optimal_new_camera_matrix)
    
    # Crop the undistorted image to the area of good pixels
    gp_image = ud_image[y:y+h, x:x+w]
    
    #if input_image.dtype == np.uint16:
    #    gp_image = conv16to8(gp_image)

    # Resize the image to the target size
    output_image = resize_thermal_to_kinect_mat(gp_image, target_size[0], target_size[1])

    # Adjust camera matrix to match the new image size
    adjusted_camera_matrix = adjust_camera_matrix_for_new_size(
        optimal_new_camera_matrix, (w, h), target_size, (x, y, w, h)
    )

    return output_image, adjusted_camera_matrix

def redistort_image(img, camera_matrix, dist_coeffs, x_factor=1.0, y_factor=1.0):
    h, w = img.shape[:2]
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    x, y = x.flatten(), y.flatten()

    # Normalize coordinates
    x_normalized = (x - camera_matrix[0, 2]) / camera_matrix[0, 0]
    y_normalized = (y - camera_matrix[1, 2]) / camera_matrix[1, 1]

    r2 = x_normalized**2 + y_normalized**2

    # Apply a customized distortion model with asymmetry adjustment on both axes
    x_distorted = x_normalized * (1 + x_factor * dist_coeffs[0] * r2**3 + x_factor * dist_coeffs[1] * r2**4)
    y_distorted = y_normalized * (1 + y_factor * dist_coeffs[0] * r2**3 + y_factor * dist_coeffs[1] * r2**4)

    # Convert back to pixel coordinates
    x_distorted = (x_distorted * camera_matrix[0, 0]) + camera_matrix[0, 2]
    y_distorted = (y_distorted * camera_matrix[1, 1]) + camera_matrix[1, 2]

    map_x, map_y = x_distorted.reshape(h, w).astype(np.float32), y_distorted.reshape(h, w).astype(np.float32)
    distorted_img = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    return distorted_img
    
def align_images_cv(cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, aAdjust, T, img1, img2, udistAlpha):
    imageSize = img1.shape[:2][::-1]  # Get the size of the image (note: shape returns height, width)

    # Compute the rectification transformation
    R1, R2, P1, P2, Q = cv2.stereoRectify(cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, imageSize, R, T, alpha=udistAlpha, flags=cv2.CALIB_ZERO_DISPARITY)[0:5]

    # create Ra matrix for adjustment
    theta = np.radians(aAdjust)  # Convert angle from degrees to radians
    c, s = np.cos(theta), np.sin(theta)
    Ra= np.array([
        [c, -s, 0],
        [s, c, 0],
        [0, 0, 1]
    ])

    # Compute the inverse of R1
    R1_inv = cv2.invert(R1)[1]  # cv2.invert returns a tuple (retval, inverse_matrix)
#     print("Inverse of R1 (R1_inv):\n", R1_inv)

    # Compute the relative rotation matrix
    R_relative = R2 @ R1_inv @ Ra # Using the '@' operator for matrix multiplication
    
    # Initialize maps for remapping
    map2x, map2y = cv2.initUndistortRectifyMap(cameraMatrix2, distCoeffs2, R_relative, P2, imageSize, cv2.CV_32FC1)

    # Apply remapping to the image
    interpolation = cv2.INTER_LINEAR
    if img2.dtype == np.uint16:  # Check if the image depth is 16-bit (unsigned short)
        interpolation = cv2.INTER_NEAREST  # Use nearest neighbor for 16-bit images to avoid interpolation artifacts

    img2_rectified = cv2.remap(img2, map2x, map2y, interpolation)

    return img2_rectified, map2x, map2y
    
def apply_perspective_scaling_and_skew(input_image, stretch_x, stretch_y, perspective_top, perspective_bottom, skew_top, skew_bottom):
    """
    Apply scaling, perspective transformations, and skewing to the image.
    
    Args:
    input_image (np.array): Input image in BGR, BGRA, or grayscale format.
    stretch_x (float): Factor to scale the image horizontally (>1.0 stretches, <1.0 compresses).
    stretch_y (float): Factor to scale the image vertically (>1.0 stretches, <1.0 compresses).
    perspective_top (int): Vertical adjustment for the top corners (positive moves outward, negative inward).
    perspective_bottom (int): Vertical adjustment for the bottom corners (positive moves outward, negative inward).
    skew_top (int): Horizontal skew for the top part of the image (pixels to shift rightward).
    skew_bottom (int): Horizontal skew for the bottom part of the image (pixels to shift rightward).

    Returns:
    np.array: The transformed image.
    """
    height, width = input_image.shape[:2]
    
    # Define source and destination points for the perspective and skew transformations
    src_points = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    dst_points = np.float32([
        [0 + perspective_top + skew_top, 0],
        [width - perspective_top + skew_top, 0],
        [0 + perspective_bottom + skew_bottom, height],
        [width - perspective_bottom + skew_bottom, height]
    ])

    # Calculate the perspective transformation matrix and apply it
    M_perspective = cv2.getPerspectiveTransform(src_points, dst_points)
    transformed_image = cv2.warpPerspective(input_image, M_perspective, (width, height))

    # Scale the image using OpenCV's resize function
    scaled_image = cv2.resize(transformed_image, None, fx=stretch_x, fy=stretch_y, interpolation=cv2.INTER_LINEAR)

    return scaled_image
    
def align_image(img1, img2, shift_x=0, shift_y=0, zoom_factor=1.0, rot=0, anti_alias=False):
    """
    Align img2 to img1 using zoom, rotation, and translation.
    Ensures that a 16-bit single-channel image remains a single-channel image.

    Parameters:
        img1 (numpy.ndarray): The base image (used for dimensions).
        img2 (numpy.ndarray): The image to be transformed.
        shift_x (int): Horizontal shift for img2.
        shift_y (int): Vertical shift for img2.
        zoom_factor (float): Zoom factor for img2.
        rot (float): Rotation angle for img2 in degrees.
        anti_alias (bool): Use anti-aliasing when resizing.

    Returns:
        numpy.ndarray: The transformed (aligned) version of img2 with the same size as img1.
    """
    # Preserve bit depth and number of channels
    is_single_channel = len(img2.shape) == 2  # Check if img2 is single-channel

    # Resize (zoom) img2 based on the zoom factor
    new_width = int(img2.shape[1] * zoom_factor)
    new_height = int(img2.shape[0] * zoom_factor)
    interpolation = cv2.INTER_LINEAR if anti_alias else cv2.INTER_NEAREST
    resized_img2 = cv2.resize(img2, (new_width, new_height), interpolation=interpolation)

    # Apply rotation to img2
    if rot != 0:
        center = (new_width // 2, new_height // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, rot, 1.0)
        rotated_img2 = cv2.warpAffine(resized_img2, rotation_matrix, (new_width, new_height), flags=interpolation)
    else:
        rotated_img2 = resized_img2

    # Calculate the center offset to align the transformed img2 with the center of img1
    center_x = (img1.shape[1] - new_width) // 2 + shift_x
    center_y = (img1.shape[0] - new_height) // 2 + shift_y

    # Ensure output matches the input format
    aligned_img2 = np.zeros_like(img1)

    # Apply translation to align img2 properly
    translation_matrix = np.float32([[1, 0, center_x], [0, 1, center_y]])
    aligned_img2 = cv2.warpAffine(rotated_img2, translation_matrix, (img1.shape[1], img1.shape[0]),
                                  flags=interpolation,
                                  borderMode=cv2.BORDER_REPLICATE)

    # Ensure output remains single-channel if input was single-channel
    if is_single_channel:
        aligned_img2 = aligned_img2[:, :, 0] if len(aligned_img2.shape) == 3 else aligned_img2

    return aligned_img2

# Function to check if object instances overlap
def check_color_borders(imgAO_colored, max_border_pixels=50):
    """
    Check if colors border each other in the instance map without 0s in between.

    Parameters:
    - imgAO_colored (np.ndarray): Instance map image with 3 8-bit channels.
    - max_border_pixels (int): Maximum number of border pixels to find before returning True.

    Returns:
    - bool: True if more than max_border_pixels border pixels are found, False otherwise.
    """
    # Get the shape of the image
    height, width, _ = imgAO_colored.shape

    # Flatten image into rows and columns
    border_pixel_count = 0

    # Iterate over the image except for the borders
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            # Current pixel
            current_pixel = imgAO_colored[y, x]
            
            # Skip if the pixel is unlabeled (0, 0, 0)
            if np.array_equal(current_pixel, [0, 0, 0]):
                continue
            
            # Neighboring pixels
            neighbors = [
                imgAO_colored[y - 1, x],  # Top
                imgAO_colored[y + 1, x],  # Bottom
                imgAO_colored[y, x - 1],  # Left
                imgAO_colored[y, x + 1]   # Right
            ]
            
            # Check if any neighboring pixel has a different color (and is not 0)
            for neighbor in neighbors:
                if not np.array_equal(neighbor, [0, 0, 0]) and not np.array_equal(neighbor, current_pixel):
                    border_pixel_count += 1
                    if border_pixel_count > max_border_pixels:
                        return True
    
    return border_pixel_count > max_border_pixels
    
def img_merge(img1, img2, weight=50, shift_x=0, shift_y=0, zoom_factor=1.0, rot=0, zeroTrans1=False, zeroTrans2=False, anti_alias=True):
    """
    Merge two images by averaging their pixels with options for alignment on the center,
    zooming, shifting, rotating, and handling of transparent pixels.
    
    Parameters:
    img1 (numpy.ndarray): The first image (base image).
    img2 (numpy.ndarray): The second image, which will be translated, zoomed, and rotated.
    weight (int): The percentage weight of the first image in the blend (0-100).
    shift_x (int): Horizontal shift for img2 (positive shifts right, negative shifts left).
    shift_y (int): Vertical shift for img2 (positive shifts down, negative shifts up).
    zoom_factor (float): Zoom factor for img2; 1.0 means no zoom, >1.0 to zoom in, <1.0 to zoom out.
    rot (float): Rotation angle for img2 in degrees.
    zeroTrans1 (bool): If True, ignore zero pixels in img1 when merging.
    zeroTrans2 (bool): If True, ignore zero pixels in img2 when merging.
    anti_alias (bool): If True, use anti-aliasing in the resize operation.

    Returns:
    numpy.ndarray: The merged image.
    """
    # Check and convert from BGRA to BGR if necessary
    if len(img1.shape) == 3 and img1.shape[2] == 4:
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGRA2BGR)
    
    if len(img2.shape) == 2:  # img2 is grayscale
        img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    elif len(img2.shape) == 3 and img2.shape[2] == 4:
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGRA2BGR)
    
    if len(img1.shape) == 2:  # img1 is grayscale
        img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    elif len(img2.shape) == 3 and img2.shape[2] == 4:
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGRA2BGR)

    # Resize (zoom) img2 based on the zoom factor
    new_width = int(img2.shape[1] * zoom_factor)
    new_height = int(img2.shape[0] * zoom_factor)
    interpolation = cv2.INTER_LINEAR if anti_alias else cv2.INTER_NEAREST
    resized_img2 = cv2.resize(img2, (new_width, new_height), interpolation=interpolation)

    # Apply rotation to img2
    center = (new_width // 2, new_height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, rot, 1.0)
    rotated_img2 = cv2.warpAffine(resized_img2, rotation_matrix, (new_width, new_height))

    # Calculate the center offset to align the zoomed and rotated img2 to the center of img1
    center_x = (img1.shape[1] - new_width) // 2 + shift_x
    center_y = (img1.shape[0] - new_height) // 2 + shift_y

    # Translate img2 to the calculated position
    translated_img2 = np.zeros_like(img1)
    translation_matrix = np.float32([[1, 0, center_x], [0, 1, center_y]])
    translated_img2 = cv2.warpAffine(rotated_img2, translation_matrix, (img1.shape[1], img1.shape[0]))

    # Prepare masks for zero transparency conditions
    mask1 = np.ones_like(img1, dtype=float) if not zeroTrans1 else (img1 != 0).astype(float)
    mask2 = np.ones_like(translated_img2, dtype=float) if not zeroTrans2 else (translated_img2 != 0).astype(float)

    # Blend the two images based on the provided weight
    weight1 = weight / 100.0
    weight2 = 1 - weight1
    combined_mask = cv2.addWeighted(mask1, weight1, mask2, weight2, 0)
    combined_mask[combined_mask == 0] = 1  # Prevent division by zero in masks

    img1_float = img1.astype(float)
    img2_float = translated_img2.astype(float)
    merged_image = cv2.addWeighted(img1_float * mask1, weight1, img2_float * mask2, weight2, 0)
    merged_image /= combined_mask

    return merged_image.astype(img1.dtype)
    
def get_depth(imgID, imgD, anno_folder, metadata):
    """
    Process images to calculate average depth per class-instance and generate binary maps for each instance.
    Additionally, apply color mapping to the annotations based on metadata.

    Args:
    imgID (str): Image ID for the corresponding annotation files.
    imgD (numpy.ndarray): 16-bit depth image with depth values in mm.
    anno_folder (str): Folder path containing annotation images.
    metadata (dict): Dictionary holding metadata such as class and instance IDs, label names, and file names.

    Returns:
    dict: Dictionary with metadata keys, each containing 'average_depth', 'binary_map', 'instance_colour'.
    numpy.ndarray: The color-mapped class annotation image.
    numpy.ndarray: The color-mapped object annotation image with instance shading.
    """
    pathAO = f'{anno_folder}/{imgID}_rgb_object.bmp'
    pathAC = f'{anno_folder}/{imgID}_rgb_class.bmp'

    # Load annotation images
    imgAO = cv2.imread(pathAO, cv2.IMREAD_UNCHANGED)
    imgAC = cv2.imread(pathAC, cv2.IMREAD_UNCHANGED)

    if imgAO is None or imgAC is None:
        print(f"Error: One of the annotation images didn't load for imgID: {imgID}.")
        return None, None, None

    # Define colors for 15 classes
    color_map = {
        0: (0, 0, 0),         # Background
        1: (255, 0, 0),       # Red
        2: (0, 255, 0),       # Green
        3: (0, 0, 255),       # Blue
        4: (255, 255, 0),     # Yellow (originally labeled Cyan)
        5: (255, 0, 255),     # Magenta
        6: (0, 255, 255),     # Cyan (originally labeled Yellow)
        7: (192, 192, 192),   # Silver
        8: (128, 128, 128),   # Gray
        9: (128, 0, 0),       # Maroon
        10: (128, 128, 0),    # Olive
        11: (0, 128, 0),      # Dark Green
        12: (128, 0, 128),    # Purple
        13: (0, 128, 128),    # Teal
        14: (0, 0, 128),      # Navy
        15: (255, 165, 0),    # Orange
        16: (165, 42, 42),    # Brown
        17: (255, 192, 203),  # Pink
        18: (255, 215, 0),    # Gold
        19: (0, 255, 127),    # Spring Green
        20: (238, 130, 238),  # Violet
        21: (75, 0, 130),     # Indigo
        22: (255, 127, 80),   # Coral
        23: (64, 224, 208),   # Turquoise
        24: (240, 230, 140),  # Khaki
        25: (218, 112, 214),  # Orchid
        26: (250, 128, 114),  # Salmon
        27: (106, 90, 205),   # Slate Blue
        28: (210, 180, 140),  # Tan
        29: (245, 222, 179),  # Wheat
        30: (220, 20, 60),    # Crimson
        31: (127, 255, 212),  # Aquamarine
        32: (160, 82, 45),    # Sienna
        33: (255, 20, 147),   # Deep Pink
        34: (204, 204, 255)   # Periwinkle
    }

    # Initialize color images
    imgAC_colored = np.zeros((imgAC.shape[0], imgAC.shape[1], 3), dtype=np.uint8)
    imgAO_colored = np.zeros_like(imgAC_colored)

    results = {}

    # Process the metadata and images
    for meta_key, meta_data in metadata.items():
        #print(f'Image id:{imgID}')
        #id2=meta_data['img_id']
        #print(f'Image id:{id2}')
        if int(meta_data['img_id']) != imgID:
             continue  # Skip entries that do not match the imgID
        
        class_id = meta_data['label_id']
        instance_id = meta_data['instance_id']
        label_name = meta_data['label_name']
        ls_label_id = meta_data['ls_label_id']
        
        # Create a mask for the current instance within the current class
        class_mask = imgAC == class_id
        instance_mask = (imgAO == instance_id) & class_mask
		
        if not np.any(instance_mask):
            print(f"No non-zero values found in mask for id {instance_id}.")
        
        #find centre of mask
        centreX, centreY = find_center_of_mask(instance_mask)
        
        # Find the center of the map
        center_map_x = imgAC.shape[1] // 2
        center_map_y = imgAC.shape[0] // 2
        
        # Calculate the distances along x and y axes
        distX = centreX - center_map_x
        distY = centreY - center_map_y

        # Extract depth values at the instance locations
        depth_values = imgD[instance_mask]
        if depth_values.size == 0:
            continue  # No depth data to process

        # Create a 2D boolean mask for growing
        non_zero_mask = imgD > 0  # Use the entire image's depth data for mask growing
        instance_non_zero_mask = instance_mask & non_zero_mask

        # Grow the mask if there are no valid depth values
        if not np.any(instance_non_zero_mask):
            instance_mask = grow_mask_until_values(imgD, instance_mask,20)
            instance_non_zero_mask = instance_mask & non_zero_mask

        if not np.any(instance_non_zero_mask):
            print(f"Warning: No valid depth values after growing mask for id {instance_id}.")
            continue

        # Extract depth values using the updated mask
        depth_values = imgD[instance_non_zero_mask]

        # Calculate percentiles only on non-zero values
        valid_percentiles = np.percentile(depth_values, [20, 80])

        # Create a mask that includes only values within the 20th to 80th percentile range and excludes zeros
        valid_mask = (depth_values >= valid_percentiles[0]) & (depth_values <= valid_percentiles[1])

        # Apply the mask to filter the depths
        filtered_depths = depth_values[valid_mask]

        # Compute the average depth, ensuring that there are filtered depths to average
        average_depth = np.mean(filtered_depths) if filtered_depths.size > 0 else None

        # Apply color to the class image
        imgAC_colored[class_mask] = color_map.get(class_id, (255, 255, 255))  # Default to white if not in color_map

        # Apply shading to the object image
        max_instances = np.max(imgAO[class_mask])
        if max_instances > 0:
            shades = np.linspace(0.5, 1.5, num=max_instances)  # Create shades for each instance
            for i in range(1, max_instances + 1):
                imgAO_colored[(imgAO == i) & class_mask] = (
                    np.array(color_map.get(class_id, (255, 255, 255))) * shades[i-1]
                ).clip(0, 255).astype(np.uint8)

        # Store results
        # Construct the key using class_id and instance_id
        key = (imgID, int(class_id), int(instance_id))
        results[key] = {
            'average_depth': average_depth,
            'binary_map': np.uint8(instance_mask) * 255,  # Create a binary map
            'instance_colour': tuple(np.mean(imgAO_colored[instance_mask], axis=0).astype(int)),  # Average color
            'label_name': label_name,
            'ls_label_id': ls_label_id,
            'class_id': class_id,
            'file_name': meta_data['file_name'],
            'label_maskT' : np.zeros((512, 640), dtype=np.uint8),
            'label_maskUV' : np.zeros((576, 720), dtype=np.uint8),
            'distX' : distX,
            'distY' : distY
        }

    # Sort by average depth in descending order (farthest objects first)
    sorted_results = sorted(results.items(), key=lambda x: x[1]['average_depth'], reverse=True)
    
    # Debug: Print all entries where average_depth is None before sorting
    # for k, v in results.items():
        # if v['average_depth'] is None:
            # print("Debug: Entry with None average_depth", k, v)

    # # Sort by average depth in descending order (farthest objects first)
    # try:
        # sorted_results = sorted(results.items(), key=lambda x: x[1]['average_depth'] if x[1]['average_depth'] is not None else float('-inf'), reverse=True)
    # except TypeError as e:
        # print("TypeError encountered:", e)
        
    return sorted_results, imgAC_colored, imgAO_colored

def process_images(imgACx, imgAOx, map_x, map_y, cam_data, imgM, valx, valy):
    """
    Process images by applying inverse mapping, resizing, cropping, and redistortion.
    
    Args:
    imgACx (numpy.ndarray): 8-bit class segmentation image.
    imgAOx (numpy.ndarray): 8-bit instance segmentation image.
    map_x (numpy.ndarray): Horizontal map for remapping.
    map_y (numpy.ndarray): Vertical map for remapping.
    cam_dataT (dict): Dictionary containing camera matrix 'CM' and distortion coefficients 'D'.
    imgM (numpy.ndarray): Reference image to match size and properties after processing.

    Returns:
    tuple: Returns processed class and instance segmentation images.
    """
    # Create inverse map
    inv_map_x, inv_map_y = create_inverse_map(map_x, map_y)

    # Remap images using the inverse map
    imgAC_unrectified = cv2.remap(imgACx, inv_map_x, inv_map_y, cv2.INTER_LINEAR)
    imgAO_unrectified = cv2.remap(imgAOx, inv_map_x, inv_map_y, cv2.INTER_LINEAR)

    # Reverse upscaling by resizing and cropping to match img2
    imgAC_unrectified = resize_and_crop(imgAC_unrectified, imgM)
    imgAO_unrectified = resize_and_crop(imgAO_unrectified, imgM)

    # Redistort the images using camera data
    valx = 0.15
    valy = 0.15
    imgAC_redistorted = redistort_image(imgAC_unrectified, cam_data['CM'], cam_data['D'], valx, valy)
    imgAO_redistorted = redistort_image(imgAO_unrectified, cam_data['CM'], cam_data['D'], valx, valy) #0.50, 0.50

    # Refine edges to smooth out ground truth maps
    imgAC_refined = refine_edges_gt_map(imgAC_redistorted, 2, "both")
    imgAO_refined = refine_edges_gt_map(imgAO_redistorted, 2, "both")

    return imgAC_refined, imgAO_refined
    
def get_region_growing(image, map_img, walker_iterations=1, seedErode=3):
    """
    Simplified function that computes and returns the 'unknown' region used for random walker segmentation.
    
    It first converts the input map to binary, then dilates it and erodes it to obtain the sure
    foreground. The unknown region is defined as the difference between the dilated binary map and the sure foreground.
    
    Args:
        image (numpy.ndarray): Input image (unused in this simplified version).
        map_img (numpy.ndarray): Input map image (binary).
        walker_iterations (int): Number of iterations for dilation.
        seedErode (int): Additional iterations for erosion to compute the sure foreground.
        
    Returns:
        numpy.ndarray: The unknown region.
    """
    # Convert map image to binary using your helper function.
    map_img = convert_mat2greybin(map_img)

    # Define a 3x3 kernel for morphological operations.
    kernel = np.ones((3, 3), np.uint8)
    
    # Dilate the binary map.
    dilated_map = cv2.dilate(map_img, kernel, iterations=walker_iterations)
    # Erode the dilated map to obtain the sure foreground.
    eroded_map = cv2.erode(dilated_map, kernel, iterations=walker_iterations + seedErode)
    
    # Compute the sure foreground as a binary image.
    sure_fg = convert_mat2greybin(eroded_map)
    # Compute the unknown region as the difference between the dilated map and the sure foreground.
    unknown = cv2.subtract(convert_mat2greybin(dilated_map), sure_fg)
    
    return unknown, convert_mat2greybin(eroded_map)

def refine_map_with_region_growing(image, map_img, other_unknown, other_eroded, beta=50, mode='bf', canny_threshold1=120, canny_threshold2=190, walker_iterations=1, refined_iterations=1, seedErode=3, growDilate = 3, grayPctMin=8,  grayPctMax=50
, disableCanny=False, otherErode=3):
    """
    Refines a map with region growing and random walker segmentation using Canny edge detection and morphological operations.
    Returns a tuple containing the refined map (colored and binary), labels, gray image, and edges.
    
    Args:
        image (numpy.ndarray): Input image (grayscale or RGB/BGRA).
        map_img (numpy.ndarray): Input map image (binary).
        other_unknown (numpy.ndarray): Merged unknown mask from other instances.
        other_eroded (numpy.ndarray): Merged eroded (seed) mask from other instances.
        beta (int): Beta parameter for random walker (default=1000).
        mode (str): Mode for random walker (default='bf').
        canny_threshold1 (int): Lower threshold for Canny edge detection (default=120).
        canny_threshold2 (int): Upper threshold for Canny edge detection (default=190).
        walker_iterations (int): Number of iterations for dilation (default=1).
        refined_iterations (int): Number of iterations for further refinement (default=1).
        seedErode (int): Additional erosion iterations for seeds (default=3).
        growDilate (int): Dilation iterations for growing areas (default=3).
        grayPctMin (int): Lower percentile for gray image enhancement (default=8).
        grayPctMax (int): Upper percentile for gray image enhancement (default=50).
        disableCanny (bool): Flag to disable Canny edge detection (default=False).
    
    Returns:
        tuple: (refined_map_colored, refined_map, labels, gray_image, edges)
    """
    # Check if the image is already grayscale
    if len(image.shape) == 3 and (image.shape[2] == 3 or image.shape[2] == 4):  # Check if the image is 3 or 4-channel (BGR or BGRA)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY if image.shape[2] == 3 else cv2.COLOR_BGRA2GRAY)
    else:  # If it's already single-channel, assume it's grayscale
        gray_image = image

    # Enhance gray image for processing
    min_val = np.percentile(gray_image, grayPctMin)
    max_val = np.percentile(gray_image, grayPctMax)
    gray_image = np.clip((gray_image - min_val) * 255.0 / (max_val - min_val), 0, 255).astype(np.uint8)
    
    #display_image_from_array(gray_image,"gray_image")
    
    # Convert map image to binary
    map_img = convert_mat2greybin(map_img)

    # Apply Canny edge detection
    if not disableCanny:
        imgGrayGauss = cv2.GaussianBlur(gray_image, (5, 5), 3)
        edges = cv2.Canny(imgGrayGauss, canny_threshold1, canny_threshold2)
    else:
        edges = np.zeros_like(map_img)

    # Morphological operations to relax the seed regions - main seed (transformed label)
    kernel = np.ones((3, 3), np.uint8)
    dilated_map = cv2.dilate(map_img, kernel, iterations=walker_iterations)
 
    eroded_map = cv2.erode(dilated_map, kernel, iterations=walker_iterations + seedErode)

    # Convert the eroded map to binary for region growing
    sure_fg = convert_mat2greybin(eroded_map)  # main seed (transformed label) 
    unknown = cv2.subtract(convert_mat2greybin(dilated_map), sure_fg)
    
    # Create seeds for region growing
    seeds = np.zeros_like(sure_fg)
    seeds[sure_fg == 255] = 2  # Foreground seed
    if not disableCanny:
        seeds[edges == 255] = 2  # Foreground seed
    seeds[unknown == 255] = 0  # Background seed
    
    # Adjust the other_unknown mask: erode it by one iteration to block only a thinner region.
    combined_other = cv2.bitwise_or(other_unknown, other_eroded)
    adjusted_other = cv2.erode(combined_other, kernel, iterations=otherErode)
    
    seeds[other_eroded == 255] = 1  # Block other seed areas
    seeds[adjusted_other == 255] = 1  # Block other growth areas
    # Debug: Inspect the seeds
    # print(f"Unique values in seeds: {np.unique(seeds)}")

    # Dilate the map_img to grow the regions for the blocked area
    dilated_map = cv2.dilate(map_img, kernel, iterations=growDilate)
    seeds[dilated_map == 0] = 1  # Blocking area
    
    ######################################
    ## Optimise based on contrast in areas
    ######################################
    # Create masks for seed and growth areas
    seed_mask = (sure_fg == 255)
    growth_mask = (unknown == 255)

    # Compute average intensities in the grayscale image for each region
    avg_seed_intensity = np.mean(gray_image[seed_mask]) if np.count_nonzero(seed_mask) > 0 else 0
    avg_growth_intensity = np.mean(gray_image[growth_mask]) if np.count_nonzero(growth_mask) > 0 else 0

    print(f"Average seed intensity: {avg_seed_intensity}")
    print(f"Average growth intensity: {avg_growth_intensity}")

    # Compute the intensity difference
    intensity_diff = abs(avg_seed_intensity - avg_growth_intensity)
    print(f"Intensity difference: {intensity_diff}")

    # Define a threshold for sufficient contrast (adjust as needed)
    contrast_threshold = 50  # example threshold

    # Dynamically adjust parameters if contrast is too low
    if intensity_diff < contrast_threshold:
        print("Low contrast detected between seed and growth areas. Adjusting parameters...")
        # Example adjustments:
        # Increase beta (makes the random walker segmentation stricter)
        beta = 150
        mode = 'cg_mg'

    ######################################
    ######################################

    # Apply random walker segmentation (region growing approach)
    try:
        labels = random_walker(gray_image, seeds, beta=beta, mode=mode)
        # Debug: Inspect the labels
        #print(f"Unique values in labels: {np.unique(labels)}")
    except Exception as e:
        print(f"Error in random_walker: {e}")
        return None, None

    # Convert labels to a binary map and smooth
    refined_map = np.uint8(labels == 2) * 255
    dilated_map = cv2.dilate(refined_map, kernel, iterations=refined_iterations)
    refined_map = cv2.erode(dilated_map, kernel, iterations=refined_iterations)
    
    # Convert to a 3-channel image for visualization
    refined_map_colored = create_colored_map(seeds, edges)  # show seed and growth areas

    return refined_map_colored, refined_map, labels, gray_image, edges
