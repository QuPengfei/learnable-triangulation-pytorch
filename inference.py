import torch
import numpy as np
import cv2
import os
from collections import defaultdict 
from mvn.models.triangulation import RANSACTriangulationNet, AlgebraicTriangulationNet, VolumetricTriangulationNet
from mvn.models.loss import KeypointsMSELoss, KeypointsMSESmoothLoss, KeypointsMAELoss, KeypointsL2Loss, VolumetricCELoss
from mvn.utils import img, multiview, op, vis, misc, cfg
from mvn.utils.img import get_square_bbox, resize_image, crop_image, normalize_image, scale_bbox
from mvn.utils.multiview import Camera
from mvn.utils.multiview import project_3d_points_to_image_plane_without_distortion as project
from mvn.datasets import utils as dataset_utils


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
        
    def inference(self, image, projMatrix, model_type, device, config, isSaveOutput=False, savePath=""):
        """
        For a single image inference
        """
        outputBatch = {}
        inputBatch = {}
        for image_batch in batch['images']:
            image_batch = image_batch_to_torch(image_batch)
            image_batch = image_batch.to(device)
            images_batch.append(image_batch)

        images_batch = torch.stack(images_batch, dim=0)

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
        print(proj_matricies_batch,proj_matricies_batch.shape)

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

def viewResult(sample,idx,prediction,config,save_images_instead=1):
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

    size = (384,384)
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

def infer(model_type="alg",max_num=5):

    if model_type == "alg":
        config = cfg.load_config("./experiments/human36m/train/human36m_alg.yaml")
    elif model_type == "vol":
        config = cfg.load_config("./experiments/human36m/train/human36m_vol_softmax.yaml")
        pelvis3d = loadPrePelvis(config.dataset.train.pred_results_path)

    device = torch.device(0)
    labels = loadHuman36mLabel(config.dataset.train.labels_path)
    detector = Detector(config, device)
    for idx in range(max_num):
        sample = [prepareSample(100+idx, labels, config.dataset.train.h36m_root, keyPoint3d=None, imageShape=config.image_shape)]
        viewSample(sample[0],idx)
        prediction, inputBatch = detector.inferHuman36Data(sample, model_type, device, config,
                                                                randomize_n_views=config.dataset.val.randomize_n_views,
                                                                min_n_views=config.dataset.val.min_n_views,
                                                                max_n_views=config.dataset.val.max_n_views)
        viewResult(sample[0],idx,prediction,config,save_images_instead=0)

if __name__ == "__main__":
    infer("alg",max_num=100)
