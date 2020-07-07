import torch
import numpy as np
import cv2
import os
import h5py
from collections import defaultdict 
from mvn.models.triangulation import RANSACTriangulationNet, AlgebraicTriangulationNet, VolumetricTriangulationNet
from mvn.models.loss import KeypointsMSELoss, KeypointsMSESmoothLoss, KeypointsMAELoss, KeypointsL2Loss, VolumetricCELoss
from mvn.utils import img, multiview, op, vis, misc, cfg
from mvn.utils.img import get_square_bbox, resize_image, crop_image, normalize_image, scale_bbox
from mvn.utils.multiview import Camera
from mvn.utils.multiview import project_3d_points_to_image_plane_without_distortion as project
from mvn.datasets import utils as dataset_utils
from mvn.utils.img import image_batch_to_torch

retval = {
    'subject_names': ['S1', 'S5', 'S6', 'S7', 'S8', 'S9', 'S11'],
    'camera_names': ['54138969', '55011271', '58860488', '60457274'],
    'action_names': [
        'Directions-1', 'Directions-2',
        'Discussion-1', 'Discussion-2',
        'Eating-1', 'Eating-2',
        'Greeting-1', 'Greeting-2',
        'Phoning-1', 'Phoning-2',
        'Posing-1', 'Posing-2',
        'Purchases-1', 'Purchases-2',
        'Sitting-1', 'Sitting-2',
        'SittingDown-1', 'SittingDown-2',
        'Smoking-1', 'Smoking-2',
        'TakingPhoto-1', 'TakingPhoto-2',
        'Waiting-1', 'Waiting-2',
        'Walking-1', 'Walking-2',
        'WalkingDog-1', 'WalkingDog-2',
        'WalkingTogether-1', 'WalkingTogether-2']
}
h5_file="data/human36m/extra/una-dinosauria-data/h36m/cameras.h5"
bbox_file="data/human36m/extra/bboxes-Human36M-GT.npy"

def square_the_bbox(bbox):
    top, left, bottom, right = bbox
    width = right - left
    height = bottom - top

    if height < width:
        center = (top + bottom) * 0.5
        top = int(round(center - width * 0.5))
        bottom = top + width
    else:
        center = (left + right) * 0.5
        left = int(round(center - height * 0.5))
        right = left + height

    return top, left, bottom, right

def fill_bbox(bb_file):
    # Fill bounding boxes TLBR
    bboxes = np.load(bb_file, allow_pickle=True).item()
    for subject in bboxes.keys():
        for action in bboxes[subject].keys():
            for camera, bbox_array in bboxes[subject][action].items():
                for frame_idx, bbox in enumerate(bbox_array):
                    bbox[:] = square_the_bbox(bbox)
    return bboxes

def fill_bbox_subject_action(bb_file, subject, action):
    # Fill bounding boxes TLBR
    bboxes = np.load(bb_file, allow_pickle=True).item()
    bboxes_subject_action = bboxes[subject][action]
    for camera, bbox_array in bboxes_subject_action.items():
        for frame_idx, bbox in enumerate(bbox_array):
            bbox[:] = square_the_bbox(bbox)
    return bboxes_subject_action

def get_bbox_subject_action(bboxes, idx):
    bbox = {}
    for (camera_idx, camera) in enumerate(retval['camera_names']):
        bbox[camera] = bboxes[camera][idx]
    return bbox

def fill_cameras(h5_cameras):
    info = np.empty(
        (len(retval['subject_names']), len(retval['camera_names'])),
        dtype=[
            ('R', np.float32, (3,3)),
            ('t', np.float32, (3,1)),
            ('K', np.float32, (3,3)),
            ('dist', np.float32, 5)
        ]
    )

    cameras_params = h5py.File(h5_cameras, 'r')
    # Fill retval['cameras']
    for subject_idx, subject in enumerate(retval['subject_names']):
        for camera_idx, camera in enumerate(retval['camera_names']):
            assert len(cameras_params[subject.replace('S', 'subject')]) == 4
            camera_params = cameras_params[subject.replace('S', 'subject')]['camera%d' % (camera_idx+1)]
            camera_retval = info[subject_idx][camera_idx]

            def camera_array_to_name(array):
                return ''.join(chr(int(x[0])) for x in array)
            assert camera_array_to_name(camera_params['Name']) == camera

            camera_retval['R'] = np.array(camera_params['R']).T
            camera_retval['t'] = -camera_retval['R'] @ camera_params['T']

            camera_retval['K'] = 0
            camera_retval['K'][:2, 2] = camera_params['c'][:, 0]
            camera_retval['K'][0, 0] = camera_params['f'][0]
            camera_retval['K'][1, 1] = camera_params['f'][1]
            camera_retval['K'][2, 2] = 1.0

            camera_retval['dist'][:2] = camera_params['k'][:2, 0]
            camera_retval['dist'][2:4] = camera_params['p'][:, 0]
            camera_retval['dist'][4] = camera_params['k'][2, 0]
    return info

def fill_cameras_subject(h5_cameras,subject):
    info = np.empty(
        len(retval['camera_names']),
        dtype=[
            ('R', np.float32, (3,3)),
            ('t', np.float32, (3,1)),
            ('K', np.float32, (3,3)),
            ('dist', np.float32, 5)
        ]
    )

    cameras = {}
    subject_idx = retval['subject_names'].index(subject)
    cameras_params = h5py.File(h5_cameras, 'r')
    # Fill retval['cameras']
    for camera_idx, camera in enumerate(retval['camera_names']):
        assert len(cameras_params[subject.replace('S', 'subject')]) == 4
        camera_params = cameras_params[subject.replace('S', 'subject')]['camera%d' % (camera_idx+1)]
        camera_retval = info[camera_idx]

        def camera_array_to_name(array):
            return ''.join(chr(int(x[0])) for x in array)
        assert camera_array_to_name(camera_params['Name']) == camera

        camera_retval['R'] = np.array(camera_params['R']).T
        camera_retval['t'] = -camera_retval['R'] @ camera_params['T']

        camera_retval['K'] = 0
        camera_retval['K'][:2, 2] = camera_params['c'][:, 0]
        camera_retval['K'][0, 0] = camera_params['f'][0]
        camera_retval['K'][1, 1] = camera_params['f'][1]
        camera_retval['K'][2, 2] = 1.0

        camera_retval['dist'][:2] = camera_params['k'][:2, 0]
        camera_retval['dist'][2:4] = camera_params['p'][:, 0]
        camera_retval['dist'][4] = camera_params['k'][2, 0]
        cameras[camera] = camera_retval
    return cameras

#retval['bboxes'] = fill_bbox(bbox_file)
#retval['cameras'] = fill_cameras(h5_file)

class Detector:
    def __init__(self, config, device = "cuda:0"):
        super().__init__()

        self.model = {
        "ransac": RANSACTriangulationNet,
        "alg": AlgebraicTriangulationNet,
        "vol": VolumetricTriangulationNet
        }[config.model.name](config, device=device).to(device)

        if config.model.init_weights:
            state_dict = torch.load(config.model.checkpoint)
            for key in list(state_dict.keys()):
                new_key = key.replace("module.", "")
                state_dict[new_key] = state_dict.pop(key)

            state_dict = torch.load(config.model.checkpoint)
            self.model.load_state_dict(state_dict, strict=True)
            print("Successfully loaded pretrained weights for whole model")

    def infer(self, batch, model_type, device, config):
        """
        For a single image inference
        """
        outputBatch = {}
        inputBatch = {}
        images_batch = []
        image_batch = image_batch_to_torch(batch['images'])
        image_batch = image_batch.to(device)
        images_batch.append(image_batch)

        images_batch = torch.stack(images_batch, dim=0)

        #proj_matricies_batch = [torch.from_numpy(camera.projection).float().to(device) for camera in batch['cameras']]
        proj_matricies_batch = torch.stack([torch.from_numpy(camera.projection) for camera in batch['cameras']], dim=0)
        proj_matricies_batch = proj_matricies_batch.float().to(device)
        proj_matricies_batchs = [] # shape (batch_size, n_views, 3, 4)
        proj_matricies_batchs.append(proj_matricies_batch)
        proj_matricies_batchs = torch.stack(proj_matricies_batchs,dim=0)

        #print(proj_matricies_batchs,proj_matricies_batchs.shape,len(batch),images_batch.shape)

        keypoints_2d_pred, cuboids_pred, base_points_pred, volumes_pred, coord_volumes_pred = None, None, None, None, None
        if model_type == "alg" or model_type == "ransac":
            keypoints_3d_pred, keypoints_2d_pred, heatmaps_pred, confidences_pred = self.model(images_batch, proj_matricies_batchs, batch)
        elif model_type == "vol":
            keypoints_3d_pred, heatmaps_pred, volumes_pred, confidences_pred, cuboids_pred, coord_volumes_pred, base_points_pred = self.model(images_batch, proj_matricies_batchs, batch)

        outputBatch["keypoints_3d_pred"] = keypoints_3d_pred
        outputBatch["heatmaps_pred"] = heatmaps_pred
        outputBatch["volumes_pred"] = volumes_pred
        outputBatch["confidences_pred"] = confidences_pred
        outputBatch["cuboids_pred"] = confidences_pred
        outputBatch["coord_volumes_pred"] = coord_volumes_pred
        outputBatch["base_points_pred"] = base_points_pred

        inputBatch["images_batch"] = images_batch
        return outputBatch, inputBatch

    def inferHuman36Data(self, batch, model_type, device, config, randomize_n_views,
                                        min_n_views,
                                        max_n_views):
        """
        For batch inferences 
        """
        outputBatch = {}
        inputBatch = {}
        collatFunction = dataset_utils.make_collate_fn(randomize_n_views,
                                        min_n_views,
                                        max_n_views)
        batch = collatFunction(batch)
        images_batch, keypoints_3d_gt, keypoints_3d_validity_gt, proj_matricies_batch  = dataset_utils.prepare_batch(batch, device, config)
        #print(proj_matricies_batch,proj_matricies_batch.shape,len(batch),images_batch.shape)

        keypoints_2d_pred, cuboids_pred, base_points_pred, volumes_pred, coord_volumes_pred = None, None, None, None, None
        if model_type == "alg" or model_type == "ransac":
            keypoints_3d_pred, keypoints_2d_pred, heatmaps_pred, confidences_pred = self.model(images_batch, proj_matricies_batch, batch)
        elif model_type == "vol":
            keypoints_3d_pred, heatmaps_pred, volumes_pred, confidences_pred, cuboids_pred, coord_volumes_pred, base_points_pred = self.model(images_batch, proj_matricies_batch, batch)

        outputBatch["keypoints_3d_pred"] = keypoints_3d_pred
        outputBatch["heatmaps_pred"] = heatmaps_pred
        outputBatch["volumes_pred"] = volumes_pred
        outputBatch["confidences_pred"] = confidences_pred
        outputBatch["cuboids_pred"] = confidences_pred
        outputBatch["coord_volumes_pred"] = coord_volumes_pred
        outputBatch["base_points_pred"] = base_points_pred

        inputBatch["images_batch"] = images_batch
        inputBatch["proj_matricies_batch"] = proj_matricies_batch
        return outputBatch, inputBatch

def viewSample(sample,idx=0):
    camera_idx = 0
    image = sample['images'][camera_idx]
    camera = sample['cameras'][camera_idx]
    subject = sample['subject'][camera_idx]
    action = sample['action'][camera_idx]

    display = image.copy()
    keypoints_2d = project(camera.projection, sample['keypoints_3d'][:, :3])
    for i,(x,y) in enumerate(keypoints_2d):
        cv2.circle(display, (int(x), int(y)), 3, (0,0,255), -1)
    file = f"./result/{subject}-{action}-{camera.name}-{idx}.png"
    cv2.imwrite(file, display)

def viewHeatmaps(sample,idx,prediction,config):
    # TODO get the visualization done
    images_batch = []
    for image_batch in sample['images']:
        images_batch.append(image_batch)

    heatmaps_vis = vis.visualize_heatmaps(
                                inputBatch["images_batch"], prediction["heatmaps_pred"],
                                kind=config.kind,
                                batch_index=0, size=5,
                                max_n_rows=10, max_n_cols=10)
    heatmaps_vis = heatmaps_vis.transpose(2, 0, 1)
    for i in range(0,4):
        cv2.imwrite(f"./result/heatmaps_test_{idx}_{i}.png", heatmaps_vis[i,:,:])

def viewVideo(sample):
    displays = []
    # Project and draw keypoints on images
    for camera_idx in range(len(sample['cameras'])): #camera_indexes_to_show:
        # import ipdb; ipdb.set_trace()
        display = sample['images'][camera_idx]

        cv2.putText(display, f"Cam {camera_idx:02}", (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0))
        displays.append(display)

    # Fancy stacked images
    for j, display in enumerate(displays):
        if j == 0:
            combined = display
        else:
            combined = np.concatenate((combined, display), axis=1)

    return combined

def viewVideoResult(sample,idx, prediction,config,size=(384,384)):
    displays = []

    keypoints3d_pred = prediction['keypoints_3d_pred'].cpu()
    keypoints_3d_pred = keypoints3d_pred[0,:, :3].detach().numpy()

    # Project and draw keypoints on images
    for camera_idx in range(len(sample['cameras'])): #camera_indexes_to_show:
        camera = sample['cameras'][camera_idx]

        keypoints_2d_pred = project(camera.projection, keypoints_3d_pred)

        # import ipdb; ipdb.set_trace()
        img = sample['images'][camera_idx]

        pred_kind = config.pred_kind if hasattr(config, "pred_kind") else config.kind
        display = vis.draw_2d_pose_cv2(keypoints_2d_pred, img, kind=pred_kind)
        cv2.putText(display, f"Cam {camera_idx:02}", (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0))
        displays.append(display)

    display3 = vis.draw_3d_pose_image(keypoints_3d_pred,kind=pred_kind,radius=450)
    display3 = cv2.cvtColor(display3,cv2.COLOR_RGBA2RGB)
    display3 = cv2.resize(display3, size, interpolation=cv2.INTER_AREA)
    cv2.putText(display3, f"3D prediction", (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0))
    displays.append(display3)

    # Fancy stacked images
    for j, display in enumerate(displays):
        if j == 0:
            combined = display
        else:
            combined = np.concatenate((combined, display), axis=1)

    return combined

def viewResult(sample,idx,prediction,config,save_images_instead=1,size=(384,384)):
    displays = []

    camera_idx = 0
    camera = sample['cameras'][camera_idx]
    subject = sample['subject'][camera_idx]
    action = sample['action'][camera_idx]

    keypoints3d_pred = prediction['keypoints_3d_pred'].cpu()
    keypoints_3d_pred = keypoints3d_pred[0,:, :3].detach().numpy()
    keypoints_3d_gt = sample['keypoints_3d'][:, :3]

    # Project and draw keypoints on images
    for camera_idx in range(len(sample['cameras'])): #camera_indexes_to_show:
        camera = sample['cameras'][camera_idx]

        keypoints_2d_pred = project(camera.projection, keypoints_3d_pred)
        keypoints_2d_gt = project(camera.projection, keypoints_3d_gt)

        # import ipdb; ipdb.set_trace()
        img = sample['images'][camera_idx]

        pred_kind = config.pred_kind if hasattr(config, "pred_kind") else config.kind
        display = vis.draw_2d_pose_cv2(keypoints_2d_pred, img, kind=pred_kind)
        #display = vis.draw_2d_pose_cv2(keypoints_2d_gt, img, kind=config.kind)
        cv2.putText(display, f"Cam {camera_idx:02}", (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0))

        displays.append(display)

    display3 = vis.draw_3d_pose_image(keypoints_3d_pred,kind=pred_kind,radius=450)
    display3 = cv2.cvtColor(display3,cv2.COLOR_RGBA2RGB)
    display3 = cv2.resize(display3, size, interpolation=cv2.INTER_AREA)
    cv2.putText(display3, f"3D prediction", (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0))
    displays.append(display3)

    display3_gt = vis.draw_3d_pose_image(sample['keypoints_3d'][:, :3],kind=pred_kind,radius=450)
    display3_gt = cv2.cvtColor(display3_gt,cv2.COLOR_RGBA2RGB)
    display3_gt = cv2.resize(display3_gt, size, interpolation=cv2.INTER_AREA)
    cv2.putText(display3_gt, f"3D GT", (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0))
    displays.append(display3_gt)

    # Fancy stacked images
    for j, display in enumerate(displays):
        if j == 0:
            combined = display
        else:
            combined = np.concatenate((combined, display), axis=1)

    # Load
    if save_images_instead:
        file = f"./result/result-{subject}-{action}-{camera.name}-{idx}.png"
        cv2.imwrite(file, combined)   
    else:
        cv2.imshow('w', combined)
        cv2.setWindowTitle('w', f"Index {idx}")
        c = cv2.waitKey(0) % 256

        if c == ord('q') or c == 27:
            print('Quitting...')
            cv2.destroyAllWindows()

def prepareSample(idx, labels, human36mRoot, keyPoint3d = None , imageShape = None, scaleBox = 1.0, crop = True, normImage = False):
    sample = defaultdict(list) # return value
    shot = labels['table'][idx]
    subject = labels['subject_names'][shot['subject_idx']]
    action = labels['action_names'][shot['action_idx']]
    frame_idx = shot['frame_idx']

    for camera_idx, camera_name in enumerate(labels['camera_names']):
        bbox = shot['bbox_by_camera_tlbr'][camera_idx][[1,0,3,2]] # TLBR to LTRB
        bbox_height = bbox[2] - bbox[0]

        if bbox_height == 0:
            # convention: if the bbox is empty, then this view is missing
            continue

        # scale the bounding box
        bbox = scale_bbox(bbox, scaleBox)

        # load image
        image_path = os.path.join(human36mRoot, subject, action, 'imageSequence', camera_name, 'img_%06d.jpg' % (frame_idx+1))
        assert os.path.isfile(image_path), '%s doesn\'t exist' % image_path
        image = cv2.imread(image_path)
        
        # load camera
        shot_camera = labels['cameras'][shot['subject_idx'], camera_idx]
        #print(shot_camera)
        retval_camera = Camera(shot_camera['R'], shot_camera['t'], shot_camera['K'], shot_camera['dist'], camera_name)

        if crop:
                # crop image
                image = crop_image(image, bbox)
                retval_camera.update_after_crop(bbox)
                

        if imageShape is not None:
            # resize
            image_shape_before_resize = image.shape[:2]
            image = resize_image(image, imageShape)
            retval_camera.update_after_resize(image_shape_before_resize, imageShape)

            sample['image_shapes_before_resize'].append(image_shape_before_resize)

        if normImage:
            image = normalize_image(image)

        sample['images'].append(image)
        sample['detections'].append(bbox + (1.0,)) # TODO add real confidences
        sample['cameras'].append(retval_camera)
        sample['proj_matrices'].append(retval_camera.projection)
        sample["action"].append(action)
        sample["subject"].append(subject)
        sample["frameId"].append(frame_idx)
        # 3D keypoints
        # add dummy confidences
        sample['keypoints_3d'] = np.pad(
            shot['keypoints'][:17],
            ((0,0), (0,1)), 'constant', constant_values=1.0)

    # build cuboid
    # base_point = sample['keypoints_3d'][6, :3]
    # sides = np.array([self.cuboid_side, self.cuboid_side, self.cuboid_side])
    # position = base_point - sides / 2
    # sample['cuboids'] = volumetric.Cuboid3D(position, sides)

    # save sample's index
    sample['indexes'] = idx

    if keyPoint3d is not None:
        sample['pred_keypoints_3d'] = keyPoint3d[idx]

    sample.default_factory = None
    return sample

def prepareVideoSample(info, images, cameras, bboxes, subject = 'S1', imageShape = [384, 384], scaleBox = 1.0, crop = True, normImage = False):
    sample = defaultdict(list) # return value
    subject_idx = info['subject_names'].index(subject) 
    for camera_idx, camera_name in enumerate(info['camera_names']):
        bbox = bboxes[camera_name][[1,0,3,2]] # TLBR to LTRB
        bbox_height = bbox[2] - bbox[0]

        if bbox_height == 0:
            # convention: if the bbox is empty, then this view is missing
            continue

        # scale the bounding box
        bbox = scale_bbox(bbox, scaleBox)
        # load camera
        shot_camera = cameras[camera_name]
        image = images[camera_name]
        #print(shot_camera)
        retval_camera = Camera(shot_camera['R'], shot_camera['t'], shot_camera['K'], shot_camera['dist'], camera_name)

        if crop:
                # crop image
                image = crop_image(image, bbox)
                retval_camera.update_after_crop(bbox)

        if imageShape is not None:
            # resize
            image_shape_before_resize = image.shape[:2]
            image = resize_image(image, imageShape)
            retval_camera.update_after_resize(image_shape_before_resize, imageShape)

        sample['images'].append(image)
        sample['cameras'].append(retval_camera)
        sample['proj_matrices'].append(retval_camera.projection)
        # projection matricies
    #print(sample['proj_matrices'])
    sample.default_factory = None
    return sample

def loadHuman36mLabel(path,train = True, withDamageAction=True, retain_every_n_frames_in_test=1):
    """
    this load the label, including bouding box, camera matrices
    """
    test = not train
    labels = np.load(path, allow_pickle=True).item()
    train_subjects = ['S1', 'S5', 'S6', 'S7', 'S8']
    test_subjects = ['S9', 'S11']

    train_subjects = list(labels['subject_names'].index(x) for x in train_subjects)
    test_subjects  = list(labels['subject_names'].index(x) for x in test_subjects)

    indices = []
    if train:
        mask = np.isin(labels['table']['subject_idx'], train_subjects, assume_unique=True)
        indices.append(np.nonzero(mask)[0])
    if test:
        mask = np.isin(labels['table']['subject_idx'], test_subjects, assume_unique=True)

        if not withDamageAction:
            mask_S9 = labels['table']['subject_idx'] == labels['subject_names'].index('S9')

            damaged_actions = 'Greeting-2', 'SittingDown-2', 'Waiting-1'
            damaged_actions = [labels['action_names'].index(x) for x in damaged_actions]
            mask_damaged_actions = np.isin(labels['table']['action_idx'], damaged_actions)

            mask &= ~(mask_S9 & mask_damaged_actions)
        
            
        indices.append(np.nonzero(mask)[0][::retain_every_n_frames_in_test])
    labels['table'] = labels['table'][np.concatenate(indices)]
    return labels

def loadPrePelvis(path):
    pred_results = np.load(path, allow_pickle=True)
    keypoints_3d_pred = pred_results['keypoints_3d'][np.argsort(pred_results['indexes'])]
    return keypoints_3d_pred

def infer(model_type="alg",max_num=5, save_images_instead=1, crop=True):

    if model_type == "alg":
        config = cfg.load_config("./experiments/human36m/train/human36m_alg.yaml")
    elif model_type == "vol":
        config = cfg.load_config("./experiments/human36m/train/human36m_vol_softmax.yaml")
        pelvis3d = loadPrePelvis(config.dataset.train.pred_results_path)

    device = torch.device(0)
    labels = loadHuman36mLabel(config.dataset.train.labels_path)
    detector = Detector(config, device=device)
    for idx in range(max_num):
        sample = [prepareSample(100+idx, labels, config.dataset.train.h36m_root, keyPoint3d=None, crop=crop, imageShape=config.image_shape)]
        viewSample(sample[0],idx)
        prediction, inputBatch = detector.inferHuman36Data(sample, model_type, device, config,
                                                                randomize_n_views=config.dataset.val.randomize_n_views,
                                                                min_n_views=config.dataset.val.min_n_views,
                                                                max_n_views=config.dataset.val.max_n_views)
        viewResult(sample[0],idx,prediction,config,save_images_instead=save_images_instead)

def infer_videos(model_type="alg",subject="S1", action="Sitting-1", max_num=5, save_images_instead=True, crop=True):

    if model_type == "alg":
        config = cfg.load_config("./experiments/human36m/train/human36m_alg.yaml")
    elif model_type == "vol":
        config = cfg.load_config("./experiments/human36m/train/human36m_vol_softmax.yaml")
        pelvis3d = loadPrePelvis(config.dataset.train.pred_results_path)

    device = torch.device(0)
    detector = Detector(config, device=device)

    bboxes = fill_bbox_subject_action(bbox_file, subject, action)
    cameras = fill_cameras_subject(h5_file,subject)
    cap = {}
    wri = None
    human36mRoot = "/dataset/experiment-dataset/extracted/"
    video_path = os.path.join(human36mRoot, subject, 'Videos')

    for (camera_idx, camera) in enumerate(retval['camera_names']):
        video_name = video_path+'/'+action.replace("-"," ")+'.'+camera+'.mp4'
        assert os.path.isfile(video_name), '%s doesn\'t exist' % video_name
        cap[camera] = cv2.VideoCapture(video_name) 
        size = (int(cap[camera].get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap[camera].get(cv2.CAP_PROP_FRAME_HEIGHT)))

    if save_images_instead:
        wri = cv2.VideoWriter(
               f'./result/result-{subject}-{action}.mp4',cv2.VideoWriter_fourcc('m','p','4','v'),
               30,(1920,384))
    idx = 0
    #while True:
    while True:
        frames = {}
        for (camera_idx, camera) in enumerate(retval['camera_names']):
            success,frames[camera] = cap[camera].read()
            if success != True:
                break

        bbox = get_bbox_subject_action(bboxes,idx)
        sample = prepareVideoSample(info=retval, images=frames, cameras=cameras, bboxes=bbox, subject = subject, imageShape = [384, 384], scaleBox = 1.0, crop = True, normImage = False)
        prediction, inputBatch = detector.infer(sample, model_type, device, config)

        combined = viewVideoResult(sample,idx, prediction,config)
        #combined = viewVideo(sample)
        idx = idx + 1
        if save_images_instead:
            if idx < max_num:
                #file = f"./result/result-video-{subject}-{action}-{camera}-{idx}.png"
                #cv2.imwrite(file, combined)
                wri.write(combined)
            else:
                break
        else:
            cv2.imshow('w', combined)
            cv2.setWindowTitle('w', f"Index {idx}")

            c = cv2.waitKey(0) % 256
            if c == ord('q') or c == 27:
                print('Quitting...')
                break;

    cv2.destroyAllWindows()
    for (camera_idx, camera) in enumerate(retval['camera_names']):
        cap[camera].release()
    if save_images_instead: wri.release()

if __name__ == "__main__":
    #infer("alg",max_num=2, crop=True)
    infer_videos("alg",max_num=1000, save_images_instead=False, crop=True)
