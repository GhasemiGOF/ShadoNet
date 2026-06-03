"""Evaluate a trained FCN nucleus classifier."""

from __future__ import annotations

import copy
import os
import time
import warnings

import click
import numpy as np
import scipy.io as sio
import torch
from skimage.feature import peak_local_max
from torch.autograd import Variable
from tqdm import tqdm

from nureg.tools.analysis_util import get_seed_name
from nureg.tools.evaluationclass import eval_folder

warnings.filterwarnings('ignore', category=RuntimeWarning)

BCD_DATASETS = {
    'BCD', 'BCDSam', 'BCDSam_fold_1', 'BCDSam_fold_2', 'BCDSam_fold_3',
    'BCDSam_fold_4', 'BCDSam_fold_5', 'BCD_fold_1', 'BCD_fold_2',
    'BCD_fold_3', 'BCD_fold_4', 'BCD_fold_5',
}
NET_DATASETS = {
    'NETnewClass', 'NETnewClassSam', 'NETnewClass_mix_p20', 'NETnewClass_mix_p40',
    'NETnewClass_mix_p60', 'NETnewClass_mix_p80', 'NETnewClass_sam_cell_p20',
    'NETnewClass_sam_cell_p40', 'NETnewClass_sam_cell_p60', 'NETnewClass_sam_cell_p80',
    'NETnewClass_no_sam', 'NETnewClass_raw_sam', 'NETnewClass_sam_area',
    'NETnewClass_sam_geom', 'NETnewClass_sam_full', 'NETnewClass_sam_unfiltered',
    'NETnewClass_sam_overlap', 'NETnewClass256', 'NETnewClassSam256',
    'NETnewClassSam_fold_1', 'NETnewClassSam_fold_2', 'NETnewClassSam_fold_3',
    'NETnewClassSam_fold_4', 'NETnewClassSam_fold_5', 'NETnewClassSam_fold_6',
    'NETnewClass_fold_1', 'NETnewClass_fold_2', 'NETnewClass_fold_3',
    'NETnewClass_fold_4', 'NETnewClass_fold_5', 'NETnewClass_fold_6',
}
PNET_DATASETS = {
    'PNET', 'PNET11', 'PNETSam', 'PNETSam_fold_1', 'PNETSam_fold_2',
    'PNETSam_fold_3', 'PNETSam_fold_4', 'PNETSam_fold_5', 'PNET_fold_1',
    'PNET_fold_2', 'PNET_fold_3', 'PNET_fold_4', 'PNET_fold_5',
}
SHIDC_BARE_DATASETS = {'SHIDC_bare', 'SHIDC_bare_SAM'}
SHIDC_DATASETS = {
    'SHIDC', 'SHIDCSam', 'SHIDC500', 'SHIDC500Sam', 'SHIDC256', 'SHIDC256Sam',
    'SHIDCSam_fold_1', 'SHIDCSam_fold_2', 'SHIDCSam_fold_3',
    'SHIDCSam_fold_4', 'SHIDCSam_fold_5', 'SHIDC_fold_1', 'SHIDC_fold_2',
    'SHIDC_fold_3', 'SHIDC_fold_4', 'SHIDC_fold_5',
}
PANNUKE_DATASETS = {
    'PanNuke', 'PanNukeSam', 'PanNukeBreast', 'PanNukeBreastSam',
    'PanNukeBreast256', 'PanNukeBreast256Sam',
}


def _load_project_components(dataset_name: str, model_name: str):
    try:
        from nureg.data.data_loader import dataset_obj, get_fcn_dataset
        from nureg.models.models import get_model, models
    except ImportError as exc:
        raise click.ClickException(
            "Missing nureg.data or nureg.models modules. Add the project data/model "
            "package to this repository before running evaluation."
        ) from exc

    if dataset_name not in dataset_obj:
        raise click.ClickException(
            f"Unknown dataset {dataset_name!r}. Available datasets: {sorted(dataset_obj.keys())}"
        )
    if model_name not in models:
        raise click.ClickException(
            f"Unknown model {model_name!r}. Available models: {sorted(models.keys())}"
        )
    return get_fcn_dataset, get_model


def parse_pred_label_map(spec):
    """Parse e.g. '1:1,2:2,3:2' or '1:1,2:2,3:drop' into {int: int|None}."""
    if not spec:
        return None
    mapping = {}
    for part in spec.split(','):
        key, val = part.split(':')
        mapping[int(key)] = None if val == 'drop' else int(val)
    return mapping


def remap_pred_labels_coords(coordinates, labels, mapping):
    if mapping is None or labels.size == 0:
        return coordinates, labels
    labels = np.squeeze(labels).astype(int)
    keep, new_labels = [], []
    for i, lab in enumerate(labels):
        if lab not in mapping:
            keep.append(i)
            new_labels.append(lab)
            continue
        target = mapping[lab]
        if target is None:
            continue
        keep.append(i)
        new_labels.append(target)
    if not keep:
        return np.empty((0, coordinates.shape[1] if coordinates.ndim == 2 else 2), dtype=coordinates.dtype), np.asarray([], dtype=labels.dtype)
    return coordinates[keep], np.asarray(new_labels, dtype=labels.dtype)


@click.command()
@click.argument('path', type=click.Path(exists=True))
@click.option('--dataset', default='NET', type=str)
@click.option('--datadir', default='',
              type=click.Path(exists=True))
@click.option('--eval_result_folder', default='experiments',
              type=click.Path())
@click.option('--model', default='frcn', type=str)
@click.option('--gpu', default='0')
@click.option('--num_cls', default=1, type=int)
@click.option(
    '--pred_label_map',
    default='',
    help='Remap predicted class IDs before scoring, e.g. 1:1,2:2,3:2 or 1:2,2:1,3:drop',
)
@click.option('--discrim_feat/--discrim_score', default=False)
def main(path, dataset, datadir, eval_result_folder, model, gpu, num_cls, pred_label_map, discrim_feat):
    # Set CUDA device explicitly
    if torch.cuda.is_available():
        try:
            torch.cuda.set_device(int(gpu))
        except (TypeError, ValueError):
            print(f'Could not set CUDA device from --gpu={gpu!r}; using current device')
        print(f'Using GPU: {gpu}')
    else:
        print('CUDA not available, using CPU')

    os.makedirs(eval_result_folder, exist_ok=True)

    get_fcn_dataset, get_model = _load_project_components(dataset, model)

    if os.path.isfile(path):
        print('Evaluate model: ', path)
    else:
        raise click.ClickException(f'No trained model found at: {path}')
    net = get_model(model, num_cls=num_cls)
    weights_dict = torch.load(path, map_location=lambda storage, loc: storage)
    net.load_state_dict(weights_dict)

    # Move model to appropriate device
    if torch.cuda.is_available():
        net = net.cuda()
        print(f'Model moved to GPU {gpu}')
    else:
        print('Model using CPU')

    net.eval()
    label_map = parse_pred_label_map(pred_label_map)
    if label_map:
        print('Prediction label remap:', label_map)

    data_split_pool = ['test']
    for data_split in data_split_pool:
        print('Evaluate ' + data_split)
        ds = get_fcn_dataset(dataset, os.path.join(
            datadir, dataset), split=data_split)
        loader = torch.utils.data.DataLoader(ds, num_workers=8)

        if len(loader) == 0:
            print('Empty data loader')
            return
        else:
            model_parent = os.path.basename(os.path.dirname(path)) or 'model'
            model_file = os.path.splitext(os.path.basename(path))[0]
            train_model_path = f'{model_parent}_{model_file}'
            savefolder = os.path.join(
                eval_result_folder, data_split, dataset, train_model_path)
            if not os.path.exists(savefolder):
                os.makedirs(savefolder)

        resultsDict = {}
        votingmap_name = 'votingmap'
        voting_time_name = 'prediction_time'
        detectmap_name = 'detectmap'
        classmap_name = 'classmap'
        threshold_pool = np.arange(0.0, 1.01, 0.05)
        local_min_pool = [2, 4, 6, 8, 10, 12, 14, 16]
        gd_radius = 16

        iterations = tqdm(enumerate(loader))
        for im_i, (im, _, im_name) in iterations:
            resultDictPath_mat = os.path.join(savefolder, im_name[0] + '.mat')

            if torch.cuda.is_available():
                im = Variable(im.cuda())
            else:
                im = Variable(im)
            votingStarting_time = time.time()
            pred_map = net.predict(im)
            if isinstance(pred_map, torch.Tensor):
                pred_map = pred_map.detach().cpu().numpy()
            VotingMap = np.squeeze(pred_map)
            votingEnding_time = time.time()
            print("prediction time: ", votingEnding_time - votingStarting_time)

            # Check if VotingMap is num_cls x H x W
            assert len(
                VotingMap.shape) == 3, "The shape of VotingMap is not correct"
            DetectMap = np.amax(VotingMap, axis=0)
            ClassMap = np.argmax(VotingMap, axis=0) + \
                1  # class labels begin with 1
            # ClassMap = np.argmax(VotingMap, axis=0) + 1  # class labels begin with 1

            resultsDict[votingmap_name] = np.copy(VotingMap)
            resultsDict[detectmap_name] = np.copy(DetectMap)
            resultsDict[classmap_name] = np.copy(ClassMap)
            resultsDict[voting_time_name] = votingEnding_time - \
                votingStarting_time

            for threshhold in threshold_pool:
                DetectMap_copy = copy.deepcopy(DetectMap)
                DetectMap_copy[DetectMap_copy < threshhold *
                               np.max(DetectMap_copy[:])] = 0
                for min_len in local_min_pool:
                    localseedname = get_seed_name(threshhold, min_len)
                    localseedlabel = get_seed_name(
                        threshhold, min_len) + '_label'
                    localseedtime = get_seed_name(
                        threshhold, min_len) + '_time'

                    thisStart = time.time()
                    # GOF_Fix
                    # coordinates = peak_local_max(DetectMap_copy, min_distance= min_len, indices = True) # N x 2
                    coordinates = peak_local_max(
                        DetectMap_copy, min_distance=min_len)  # N x 2
                    thisEnd = time.time()

                    if coordinates.size == 0:
                        coordinates = np.asarray([])
                        labels = np.asarray([])
                        print("Detect: Empty coordinates for img:{s} for parameter t_{thd:3.2f}_r_{rad:3.2f}".format(
                            s=im_name[0], thd=threshhold, rad=min_len))
                    else:
                        labels = ClassMap[coordinates[:, 0].astype(
                            int), coordinates[:, 1].astype(int)]
                        if label_map:
                            coordinates, labels = remap_pred_labels_coords(
                                coordinates, labels, label_map)

                    resultsDict[localseedname] = coordinates
                    resultsDict[localseedlabel] = labels
                    resultsDict[localseedtime] = thisEnd - \
                        thisStart + resultsDict[voting_time_name]

            sio.savemat(resultDictPath_mat, resultsDict)

        # overlay predictions on images
        imgfolder = os.path.join(datadir, dataset, 'images', data_split)
        # printCoordsClass(savefolder, imgfolder, ['.png', '.jpg', '.bmp'], threshhold=threshold_pool[8], min_len=local_min_pool[2], num_cls=num_cls)

        # quantitative analysis

        if dataset in NET_DATASETS:
            if data_split == 'test':
                labels = [1, 2, 3]
                labelweights = {1: 0.037, 2: 0.749, 3: 0.214}
            elif data_split == 'val':
                labels = [1, 2, 3]
                labelweights = {1: 0.079, 2: 0.730, 3: 0.191}
        elif dataset in BCD_DATASETS:
            labels = [1, 2]
            labelweights = {1: 0.334, 2: 0.666}
        elif dataset in SHIDC_BARE_DATASETS:
            # Class weights = GT nucleus counts per class / total (from mats/<split>/*_withcontour.mat, field Labels).
            # SHIDC_bare: test 700 mats, 49778 nuclei; val 166 mats, 11045 nuclei (counts as of dataset in repo).
            labels = [1, 2, 3]
            if data_split == 'val':
                labelweights = {1: 0.297239, 2: 0.658941, 3: 0.043820}
            elif data_split == 'test':
                labelweights = {1: 0.316505, 2: 0.655772, 3: 0.027723}
            else:
                labelweights = {1: 0.311431, 2: 0.662850, 3: 0.025719}
        elif dataset in SHIDC_DATASETS:
            if data_split == 'val':
                labels = [1, 2, 3]
                labelweights = {1: 0.079, 2: 0.730, 3: 0.191}
            elif data_split == 'test':
                labels = [1, 2, 3]
                labelweights = {1: 0.037, 2: 0.749, 3: 0.214}

        elif dataset in {'PanNuke', 'PanNukeSam'}:
            labels = [1, 2, 3, 4, 5]
            labelweights = {1: 0.408, 2: 0.17, 3: 0.267, 4: 0.015, 5: 0.14}

        elif dataset in {'PanNukeBreast', 'PanNukeBreast256', 'PanNukeBreastSam', 'PanNukeBreast256Sam'}:
            labels = [1, 2, 3, 4, 5]
            labelweights = {1: 0.407, 2: 0.116, 3: 0.220, 4: 0.0002, 5: 0.255}
        elif dataset in PNET_DATASETS:
            labels = [1, 2, 3]
            labelweights = {1: 0.037, 2: 0.749, 3: 0.214}

        else:
            raise ValueError(f"Unsupported dataset '{dataset}'")

        resfolder = savefolder
        # modelfolder = path.split('/',2)[1] + '_' + path.rsplit('/',1)[1].split('.')[0]
        modelfolder = os.path.basename(os.path.dirname(path)) or 'model'
        eval_savefolder = os.path.join(eval_result_folder, data_split, dataset)

        if not os.path.exists(eval_savefolder):
            os.makedirs(eval_savefolder)
        eval_folder(imgfolder=imgfolder, resfolder=resfolder, savefolder=eval_savefolder,
                    radius=gd_radius, resultmask=modelfolder, thresh_pool=threshold_pool,
                    len_pool=local_min_pool, imgExt=['.bmp', '.jpg', '.png'], contourname='Contours', contourlabel='Labels',
                    matExt=['_withcontour'], eval_class=True, labels=labels,
                    labelweights=labelweights)


if __name__ == '__main__':
    main()
