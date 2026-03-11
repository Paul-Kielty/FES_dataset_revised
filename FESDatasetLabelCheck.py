import os
from torch.utils.data import Dataset
from glob import glob
import numpy as np
from math import floor
import xml.etree.ElementTree as ET
import pandas as pd

dataset_root = "Z:\\datasets\\FES\\raw_dataset"
FES_landmarks_dict = {0: 'eye_l', 1: 'eye_r',  2: 'nose',  3: 'mouth_l',  4: 'mouth_r'}


class FESDatasetCheck(Dataset):
    def __init__(self, wild=False):
        self.setting = "wild" if wild else "lab"

        event_paths = glob(dataset_root+f"/event_streams_npy_reformat/{self.setting}/**/*.npy")
        event_paths_sort_reference = []
        for i in range(len(event_paths)):
            path = event_paths[i]
            split1, split2 = path.rsplit('_',1)
            split2, split3 = split2.split('.', 1)
            path = f"{split1}_{split2.zfill(2)}.{split3}"
            event_paths_sort_reference.append(path)

        event_paths = [x for _, x in sorted(zip(event_paths_sort_reference, event_paths))]
        self.event_paths = event_paths



    def __getitem__(self, index):
        event_path = self.event_paths[index]
        video_name = os.path.splitext(os.path.basename(event_path))[0]
        
        video_date_dir = video_name[:video_name.rfind('_')]
        video_number = video_name[video_name.rfind('_') + 1:]
        video_number = str(int(video_number))  # remove leading zeros

        print("______________________________________________________________________________________________________")
        print(f"{video_date_dir}/{video_number.zfill(3)}")
        labels = combine_annotations(video_name)
        labels = pd.DataFrame(labels).to_numpy()
    
        split_labels = split_labels_by_faults(labels, count_consecutive=True)
        split_labels = [x for x in split_labels if len(x) >= 30] # minimum 1s of video

        if len(split_labels) == 0:
                return False
        
        return True

        
    def __len__(self):
        return len(self.event_paths)

fault_thresholds = {'LMs outside bbox by > 10%'             : -1,
                    'Bbox outside image'                    : -1,
                    'Mismatched SUM displacement'           : -1,
                    'Frozen LMs (bbox change > 5%)'         : -1,
                    'Frozen bbox (LM change > 5%)'          : -1,
                    'LM indices (Nose position)'            : -1,
                    'LM indices (Left/right e/m positions)' : -1,
                    'LM indices (Top/bottom e/m positions)' : -1,
                    'LMs outside image'                     : -1,
                    'LM occluded True'                      : -1,
                    'LM outside True'                       : -1,
                    'Bbox occluded True'                    : -1,
                    'Bbox outside True'                     : -1,
                    'LM indices (swap)'                     : -1,
                    'LM count' : -1,
                    # '' : -1,
                    # '' : -1,
                    }

fault_pad = {'LMs outside bbox by > 10%'             :  2,
            #  'Bbox outside image'                    : -1,
             'Mismatched SUM displacement'           :  5,
            #  'Frozen LMs (bbox change > 5%)'         : -1,
            #  'Frozen bbox (LM change > 5%)'          : -1,
            #  'LM indices (Nose position)'            : -1,
            #  'LM indices (Left/right e/m positions)' : -1,
            #  'LM indices (Top/bottom e/m positions)' : -1,
            #  'LMs outside image'                     : -1,
            #  'LM occluded True'                      : -1,
            #  'LM outside True'                       : -1,
            #  'Bbox occluded True'                    : -1,
            #  'Bbox outside True'                     : -1,
            #  'LM indices (swap)'                     : -1,
             }

def split_labels_by_faults(labels, count_consecutive=True):


    faults_at = np.argwhere(labels[:,4]).flatten()
    faults = labels[faults_at,4]
    
    unique_faults, unique_counts = np.unique(faults, return_counts=True)
    if not count_consecutive:
        # remove consecutive faults (of same type) from unique counts
        for i,fault in enumerate(unique_faults):
            curr_fault_at = np.argwhere(fault == labels[:,4]).flatten()
            if len(curr_fault_at) > 1:
                diffs = np.diff(curr_fault_at.flatten())
                # non_consecutive_counts[i] -= np.count_nonzero(diffs==1)
                unique_counts[i] -= np.count_nonzero(diffs==1)
    # check against fault_thresholds
    for fault, count in zip(unique_faults, unique_counts):
        if fault in fault_thresholds.keys():
            if count > fault_thresholds[fault] and fault_thresholds[fault] >= 0:
                print(f"Fault \"{fault}\" exceeds threshold ( {count}/{fault_thresholds[fault]} ), skipping video")
                [print("\t",k,v) for k,v in zip(unique_faults, unique_counts)]
                # [print("\t\t",i,fault) for i, fault in zip(faults_at,faults)]
                return []
        else:
            print(f"*** Fault \"{fault}\" has no set threshold ***")
       
    [print("\t",k,v) for k,v in zip(unique_faults, unique_counts)]
    remove_indices = np.zeros(labels.shape[0], dtype=bool)
    remove_indices[faults_at] = True
    for fault, fault_at in zip(faults, faults_at):
        if fault in fault_pad.keys():
            pad = fault_pad[fault]
            for i in range(fault_at-pad, fault_at+pad+1):
                if i < 0 or i >= len(remove_indices):
                    continue
                remove_indices[i] = True
                
            
    no_bbox_at = [not isinstance(label, np.ndarray) for label in labels[:,2]]
    no_lms_at = [not isinstance(label, np.ndarray) for label in labels[:,3]]
    skip_missing_labels = np.logical_or(no_bbox_at, no_lms_at)
    remove_indices[skip_missing_labels] = True

    split_indices = np.argwhere(remove_indices).flatten()
    split_labels = np.array_split(labels, split_indices)
    split_labels = [split[1:] for split in split_labels if len(split) > 1]  # Remove empty splits

    return split_labels


def parse_lm_annotations(xml_file):
    # Parse the XML file
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Read version
    version = root.find('version').text if root.find('version') is not None else None
    
    # Read original image size
    size_node = root.find('meta/task/original_size')
    original_size = {
        'width': int(size_node.find('width').text),
        'height': int(size_node.find('height').text)
    } if size_node is not None else {}

    # Parse each track
    tracks = []
    for track in root.findall('track'):
        track_id = track.get('id')
        label = track.get('label')
        source = track.get('source')
        points_data = []

        for pts in track.findall('points'):
            frame = int(pts.get('frame'))
            outside = bool(int(pts.get('outside')))
            occluded = bool(int(pts.get('occluded')))
            keyframe = bool(int(pts.get('keyframe')))
            z_order = int(pts.get('z_order'))
            # Parse the semi-colon separated point coordinates
            coords = [tuple(map(float, p.split(','))) for p in pts.get('points').split(';')]

            points_data.append({
                'frame': frame,
                'outside': outside,
                'occluded': occluded,
                'keyframe': keyframe,
                'z_order': z_order,
                'points': coords
            })

        tracks.append({
            'id': track_id,
            'label': label,
            'source': source,
            'points': points_data
        })

    return {
        'version': version,
        'original_size': original_size,
        'tracks': tracks
    }

def parse_bbox_annotations(xml_file):
    """
    Parse annotations where coordinates are stored in <box> elements.
    Returns a dict with version, original_size, and a list of tracks with box data.
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()

    version = root.find('version').text if root.find('version') is not None else None
    size_node = root.find('meta/task/original_size') or root.find('meta/task/size')
    original_size = {}
    if size_node is not None and size_node.find('width') is not None:
        original_size = {
            'width': int(size_node.find('width').text),
            'height': int(size_node.find('height').text)
        }

    tracks = []
    for track in root.findall('track'):
        tid = track.get('id')
        label = track.get('label')
        source = track.get('source')
        box_data = []
        for box in track.findall('box'):
            frame = int(box.get('frame'))
            outside = bool(int(box.get('outside')))
            occluded = bool(int(box.get('occluded')))
            keyframe = bool(int(box.get('keyframe')))
            z_order = int(box.get('z_order'))
            xtl = float(box.get('xtl'))
            ytl = float(box.get('ytl'))
            xbr = float(box.get('xbr'))
            ybr = float(box.get('ybr'))
            box_data.append({
                'frame': frame,
                'outside': outside,
                'occluded': occluded,
                'keyframe': keyframe,
                'z_order': z_order,
                'xtl': xtl,
                'ytl': ytl,
                'xbr': xbr,
                'ybr': ybr
            })
        tracks.append({'id': tid, 'label': label, 'source': source, 'boxes': box_data})

    return {'version': version, 'original_size': original_size, 'tracks': tracks}

def parse_full_annotations(video_name):
    # video_date, video_date_id
    video_date_dir = video_name[:video_name.rfind('_')]
    video_number = video_name[video_name.rfind('_') + 1:]
    video_number = str(int(video_number))  # remove leading zeros
    # print(f"\n\nParsing & checking annotations for {video_name}")

    labels_bbox_full_path = os.path.join(dataset_root,"annotations","lab","bbox","lab_github",
                                         "bbox_xml",video_date_dir,video_number,"annotations.xml")
    labels_lm_full_path = os.path.join(dataset_root,"annotations","lab","fl","lab_github",
                                       "fld_xml",video_date_dir,video_number,"annotations.xml")

    annotations_bbox = parse_bbox_annotations(labels_bbox_full_path)
    if len(annotations_bbox['tracks']) > 1:
        print("Found more than one track in the annotations, taking last")
    annotations_lm = parse_lm_annotations(labels_lm_full_path)
    
    if len(annotations_lm['tracks']) > 1:
        print("Found more than one track in the LM annotations, taking last")
    annotations_bbox = annotations_bbox['tracks'][-1]
    annotations_lm = annotations_lm['tracks'][-1]

    return annotations_bbox, annotations_lm

def combine_annotations(video_name):

    annotations_bbox, annotations_lm = parse_full_annotations(video_name)
    bbox_df = pd.DataFrame(annotations_bbox['boxes'])
    lm_df = pd.DataFrame(annotations_lm['points'])
    max_frame = max( max(bbox_df['frame']), max(lm_df['frame']) )
    duration = 1e6/30
    
    new_labels = []
    curr_frame = 0
    misc_keys = ['outside', 'occluded', 'keyframe', 'z_order']
    for curr_frame in range(max_frame + 1):
        bbox_corners = None
        landmarks = None
        misc_labels = None

        ts = int(curr_frame * duration)
        bbox_curr_i = np.argwhere(bbox_df['frame'] == curr_frame).flatten()
        lm_curr_i = np.argwhere(lm_df['frame'] == curr_frame).flatten()

        if len(bbox_curr_i) == 0:
            bbox_part = None
        elif len(bbox_curr_i) == 1:
            bbox_part = bbox_df.iloc[bbox_curr_i[0]].to_dict()
        else:
            # raise ValueError(f"Multiple bbox entries for frame {curr_frame}")
            print(f"Warning: Multiple bbox entries for frame {curr_frame}, skipping")
            bbox_part = None
        
        if len(lm_curr_i) == 0:
            lm_part = None
        elif len(lm_curr_i) == 1:
            lm_part = lm_df.iloc[lm_curr_i[0]].to_dict()
        else:
            # raise ValueError(f"Multiple lm entries for frame {curr_frame}")
            print(f"Warning: Multiple lm entries for frame {curr_frame}, skipping")
            lm_part = None
        faults = None
        if bbox_part is not None and lm_part is not None:
            # perform checks of label contents
            lm_part, bbox_part, faults, landmarks, bbox_corners = check_frame_labels(lm_part, bbox_part, curr_frame, new_labels)


        if landmarks is not None:
            misc_labels = {}
            for key in misc_keys:
                misc_labels[key+"_lm"] = lm_part[key]

        if bbox_corners is not None:
            if misc_labels is None:
                misc_labels = {}
            for key in misc_keys:
                misc_labels[key+"_bbox"] = bbox_part[key]
        
        
        new_labels.append({
            'i': curr_frame,
            'ts': ts,
            'bbox_corners': bbox_corners,
            'landmarks': landmarks,
            'faults': faults,
            'misc': misc_labels,
        })

    return new_labels

# multiple tracks and multiple bbox/lms entries per frame are already checked for and skipped
def check_frame_labels(lm_part, bbox_part, curr_frame, new_labels):
    lms = np.array(lm_part['points'])
    
    bbox_corners = np.array([bbox_part['xtl'], bbox_part['ytl'], bbox_part['xbr'], bbox_part['ybr']])
    bbox_corners = bbox_corners.reshape(2,2) # (x1,y1), (x2,y2)
    bbox_l2 = np.linalg.norm(bbox_corners[1] - bbox_corners[0]) # length of the diagonal of the bbox
    

    if not check_bbox_shape(bbox_corners):
        return None, None, "Bbox shape", None, None
    
    
    if not check_landmark_count(lms, count=5):
        return None, bbox_part, "LM count", None, bbox_corners
    
    if not check_lms_within_img(lms):
        return None, bbox_part, "LMs outside image", None, bbox_corners
    
    if not check_lms_within_bbox(lms, bbox_corners, margin=0):
        if check_lms_within_bbox(lms, bbox_corners, margin=0.1*bbox_l2):
            # expand bbox if landmarks are within 10% of bbox size
            bbox_corners = expand_bbox_to_lms(lms, bbox_corners)
        else:
            return None, None, "LMs outside bbox by > 10%", None, None

    if not check_bbox_within_img(bbox_corners):
        return None, None, "Bbox outside image", None, None
    
    if not check_left_right_positions(lms):
        return None, bbox_part, "LM indices (Left/right e/m positions)", None, bbox_corners
    
    if not check_top_bottom_positions(lms):
        return None, bbox_part, "LM indices (Top/bottom e/m positions)", None, bbox_corners
    
    if not check_nose_position(lms):
        return None, bbox_part, "LM indices (Nose position)", None, bbox_corners
    
    if not check_misc_labels(lm_part, ['occluded']):
        # return None, bbox_part, "LM occluded True", None, bbox_corners
        print(curr_frame, "Warning: LM occluded == True but not skipping frame")
    if not check_misc_labels(lm_part, ['outside']):
        return None, bbox_part, "LM outside True", None, bbox_corners
    
    if not check_misc_labels(bbox_part, ['occluded']):
        print(curr_frame, "Warning: Bbox occluded == True but not skipping frame")
        # return lm_part, None, "Bbox occluded True", lms, None
    if not check_misc_labels(bbox_part, ['outside']):
        return lm_part, None, "Bbox outside True", lms, None
    
    if len(new_labels) > 0:
        previous_lms = new_labels[-1]['landmarks']
        if previous_lms is not None:
    
            previous_bbox = new_labels[-1]['bbox_corners']
            if previous_bbox is not None:
                
                any_lm_change = check_lms_moved(lms, previous_lms, 0)
                any_bbox_change = check_bbox_moved(bbox_corners, previous_bbox, 0)
                
                    
                if check_lms_moved(lms, previous_lms, 0.05 * bbox_l2):
                    if not any_bbox_change:   
                        return lm_part, None, f"Frozen bbox (LM change > 5%)", lms, None
                    
                if check_bbox_moved(bbox_corners, previous_bbox, 0.05 * bbox_l2):   
                    if not any_lm_change:
                        return None, bbox_part, f"Frozen LMs (bbox change > 5%)", None, bbox_corners
                
                if check_mismatched_lms_bbox_change_sum(lms, previous_lms, bbox_corners, previous_bbox, 0.2 * bbox_l2):
                    return None, None, "Mismatched SUM displacement (> 20%)", None, None
                
      
    return lm_part, bbox_part, None, lms, bbox_corners

def check_bbox_shape(bbox):
    """
    Check bbox br corner > tl corner.
    """
    if bbox[0,0] >= bbox[1,0]:
        return False
    if bbox[0,1] >= bbox[1,1]:
        return False
    return True

def check_bbox_within_img(bbox, h=360, w=480):
    """
    Check bbox corners are within image bounds.
    """
    if bbox[0,0] < 0:
        return False
    if bbox[0,1] < 0:
        return False
    if bbox[1,0] >= w:
        return False
    if bbox[1,1] >= h:
        return False
    return True

def check_landmark_count(lms, count=5):
    if lms.shape==(count, 2):
        return True
    # print(f"Landmark count mismatch (expected [{count}, 2], got {lms.shape})")
    return False
    
def check_lms_within_bbox(lms, bbox, margin=0):
    """
    Check if any landmarks are outside the bounding box.
    """
    if min(lms[:,0]) < bbox[0,0]-margin:
        # print(f"Leftmost landmark outside the bounding box ({min(lms[:,0])} < {bbox[0]})")
        return False
    if min(lms[:,1]) < bbox[0,1]-margin:
        # print(f"Top landmark outside the bounding box ({min(lms[:,1])} < {bbox[1]})")
        return False
    if max(lms[:,0]) > bbox[1,0]+margin:
        # print(f"Rightmost landmark outside the bounding box ({max(lms[:,0])} > {bbox[2]})")
        return False
    if max(lms[:,1]) > bbox[1,1]+margin:
        # print(f"Bottom landmark outside the bounding box ({max(lms[:,1])} > {bbox[3]})")
        return False
    return True

def check_lms_within_img(lms, h=360, w=480):
    """
    Check if any landmarks are outside the image bounds.
    """
    if min(lms[:,0]) < 0:
        return False
    if min(lms[:,1]) < 0:
        return False
    if max(lms[:,0]) >= w:
        return False
    if max(lms[:,1]) >= h:
        return False
    return True

def expand_bbox_to_lms(lms, bbox):
    """
    Expand bounding box to contain all lms.
    """
    if min(lms[:,0]) < bbox[0,0]:
        bbox[0,0] = min(lms[:,0])
        
    if min(lms[:,1]) < bbox[0,1]:
        bbox[0,1] = min(lms[:,1])
        
    if max(lms[:,0]) > bbox[1,0]:
        bbox[1,0] = max(lms[:,0])
        
    if max(lms[:,1]) > bbox[1,1]:
        bbox[1,1] = max(lms[:,1])

    return bbox

def check_nose_position(lms, margin_factor=1):
    
    nose_x, nose_y = lms[2] # Assuming the nose is at index 2
    e_l_x, e_l_y = lms[0]   # Left eye
    e_r_x, e_r_y = lms[1]   # Right eye
    m_l_x, m_l_y = lms[3]   # Left mouth corner
    m_r_x, m_r_y = lms[4]   # Right mouth corner
    # y should be below lms 0 and 1, but above lms 3 and 4
    # if y distance between landmarks on left side (e_l,m_l) less than x distance between eyes or mouth corners, head is tilted up or down and dont check nose Y
    
    # Head pitch check
    eye_mouth_h = max(abs(e_l_y - m_l_y), abs(e_r_y - m_r_y)) # max height difference between left eye and left mouth corner, and right eye and right mouth corner
    eye_mouth_w = max(abs(e_l_x - e_r_x), abs(m_l_x - m_r_x)) # max width difference between left eye and right eye, and left mouth corner and right mouth corner
    if eye_mouth_h > eye_mouth_w*margin_factor: # skip cases where head is tilted up or down
        if nose_y < (e_l_y + e_r_y)/2:
            # print(f"Nose landmark y-coordinate {nose_y} is above the eyes {e_l_y} and {e_r_y}")
            return False
        if nose_y > (m_l_y + m_r_y)/2:
            # print(f"Nose landmark y-coordinate {nose_y} is below the mouth corners {m_l_y} and {m_r_y}")
            return False

    if eye_mouth_w > eye_mouth_h*margin_factor:
        if nose_x < (e_l_x + m_l_x)/2:
            # print(f"Nose landmark x-coordinate {nose_x} is left of left eye and mouth {e_l_x} and {m_l_x}")
            return False
        if nose_x > (e_r_x + m_r_x)/2:
            # print(f"Nose landmark x-coordinate {nose_x} is right of right eye and mouth {e_r_x} and {m_r_x}")
            return False
    return True

def check_left_right_positions(lms):
    e_l_x, e_l_y = lms[0]  # Left eye
    e_r_x, e_r_y = lms[1]  # Right eye
    n_x, n_y = lms[2]      # Nose
    m_l_x, m_l_y = lms[3]  # Left mouth corner
    m_r_x, m_r_y = lms[4]  # Right mouth corner

    if e_l_x > e_r_x:
        return False
    if m_l_x > m_r_x:
        return False
    return True

def check_top_bottom_positions(lms):
    e_l_x, e_l_y = lms[0]  # Left eye
    e_r_x, e_r_y = lms[1]  # Right eye
    n_x, n_y = lms[2]      # Nose
    m_l_x, m_l_y = lms[3]  # Left mouth corner
    m_r_x, m_r_y = lms[4]  # Right mouth corner
    if e_l_y > m_l_y:
        return False
    if e_r_y > m_r_y:
        return False
    return True

def check_bbox_moved(curr_bbox, prev_bbox, margin):
    diffs = curr_bbox-prev_bbox
    if np.max(np.abs(diffs)) > margin:
        return True
    return False

def check_lms_moved(curr_lms, prev_lms, margin):
    
    diffs = curr_lms-prev_lms
    
    if np.max(np.abs(diffs)) > margin:
        return True
    return False

def check_mismatched_lms_bbox_change_sum(curr_lms, prev_lms, curr_bbox, prev_bbox, margin):
    # Check for changes in x & y displacement of landmarks and bounding box within margin
    # get centrepoint of bbox

    curr_bbox_center = np.mean(curr_bbox, axis=0)
    prev_bbox_center = np.mean(prev_bbox, axis=0)

    bbox_diff = curr_bbox_center-prev_bbox_center

    lms_diff = curr_lms-prev_lms
    x_disp = np.sum(np.abs(lms_diff[:,0]-bbox_diff[0]))
    y_disp = np.sum(np.abs(lms_diff[:,1]-bbox_diff[1]))
    # print(f"x_disp: {x_disp:.2f}, y_disp: {y_disp:.2f}, margin: {margin:.2f}")
    if np.abs(x_disp) > margin or np.abs(y_disp) > margin:
        return True
    return False

def check_misc_labels(labels, keys=['outside', 'occluded']):
    # misc_keys = ['outside']
    for key in keys:
        if labels[key]:
            return False
    return True

def gen_target_pip(target, meanface_indices, target_map, target_local_x, target_local_y, target_nb_x, target_nb_y):
    num_nb = len(meanface_indices[0])
    map_channel, map_height, map_width = target_map.shape
    target = target.reshape(-1, 2)
    assert map_channel == target.shape[0]

    for i in range(map_channel):
        mu_x = int(floor(target[i][0] * map_width))
        mu_y = int(floor(target[i][1] * map_height))
        mu_x = max(0, mu_x)
        mu_y = max(0, mu_y)
        mu_x = min(mu_x, map_width-1)
        mu_y = min(mu_y, map_height-1)
        target_map[i, mu_y, mu_x] = 1
        shift_x = target[i][0] * map_width - mu_x
        shift_y = target[i][1] * map_height - mu_y
        target_local_x[i, mu_y, mu_x] = shift_x
        target_local_y[i, mu_y, mu_x] = shift_y

        for j in range(num_nb):
            nb_x = target[meanface_indices[i][j]][0] * map_width - mu_x
            nb_y = target[meanface_indices[i][j]][1] * map_height - mu_y
            target_nb_x[num_nb*i+j, mu_y, mu_x] = nb_x
            target_nb_y[num_nb*i+j, mu_y, mu_x] = nb_y

    return target_map, target_local_x, target_local_y, target_nb_x, target_nb_y

if __name__ == '__main__':
    
    dataset = FESDatasetCheck()

    passed = []
    for i in range(len(dataset)):
        passed.append(dataset[i])
    passed = np.array(passed)
    passed_count = np.sum(passed)
    print(f"Passed {passed_count}/{len(passed)} samples, {passed_count/len(passed)*100:.2f}%")
