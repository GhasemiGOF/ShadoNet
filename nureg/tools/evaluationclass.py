from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import scipy.io as sio
from scipy.io import loadmat
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.pairwise import pairwise_distances

from .analysis_util import get_seed_name, getfilelist


def _find_mat_file(imgfolder, imgname, mat_exts):
    """Find the ground-truth .mat file that corresponds to an image."""
    imgfolder = Path(imgfolder)
    dataset_root = imgfolder.parent.parent
    mats_dir = dataset_root / "mats" / imgfolder.name
    for mat_ext in mat_exts:
        candidate = mats_dir / f"{imgname}{mat_ext}.mat"
        if candidate.exists():
            return candidate
    return None


def _centers_to_row_col(centers):
    """Convert stored centers from (x, y) into evaluation order (row, col)."""
    centers = np.asarray(centers)
    if centers.size == 0:
        return np.empty((0, 2), dtype=float)
    if centers.ndim != 2:
        centers = np.reshape(centers, (-1, 2))
    if centers.shape[0] == 2:
        xy = centers.T
    else:
        xy = centers
    gt = np.zeros_like(xy, dtype=float)
    gt[:, 0] = xy[:, 1]
    gt[:, 1] = xy[:, 0]
    return gt


def _safe_concat(arrays):
    arrays = [np.asarray(a).reshape(-1) for a in arrays if np.asarray(a).size > 0]
    if not arrays:
        return np.asarray([], dtype=float)
    return np.concatenate(arrays)


def _safe_mean(values):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return 0.0
    return float(np.mean(values))


def _safe_std(values):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return 0.0
    return float(np.std(values))


def eval_folder(imgfolder='.', resfolder='.', savefolder='.', radius=16, resultmask='',
                thresh_pool=[0.55], len_pool=[9], imgExt=['.tif'], contourname='Contours', contourlabel='Labels',
                matExt=["", '_withcontour', '_gt'], eval_class=True, labels=[1, 2, 3],
                labelweights=None):

    if eval_class == True and labelweights is None:
        labelweights = {}
        for il in labels:
            labelweights[il] = 1.0/len(labels)

    print('Currently evaluating: ', imgfolder)
    img_list, img_pool = getfilelist(imgfolder, imgExt)

    num_img, num_thresh, num_len = len(
        img_pool), len(thresh_pool), len(len_pool)
    print('Number of images: ', num_img)

    TP_res = np.zeros((num_img, num_thresh, num_len))
    FP_res = np.zeros((num_img, num_thresh, num_len))
    FN_res = np.zeros((num_img, num_thresh, num_len))
    TN_res = np.zeros((num_img, num_thresh, num_len))
    pre_res = np.zeros((num_img, num_thresh, num_len))
    rec_res = np.zeros((num_img, num_thresh, num_len))
    f1_res = np.zeros((num_img, num_thresh, num_len))
    spe_res = np.zeros((num_img, num_thresh, num_len))
    diff_res = np.zeros((num_img, num_thresh, num_len))
    dict_res = {}

    # distance_res = [[None]*num_len]*num_thresh

    res_json = {}
    save_path = os.path.join(savefolder, resultmask +
                             '_radius' + str(radius) + '_res3.json')

    for th_idx, thresh in enumerate(thresh_pool):
        for len_idx, maxlen in enumerate(len_pool):
            key_name = get_seed_name(thresh, maxlen)
            dict_res[key_name + 'TPidx_detaction'] = {}
            dict_res[key_name + 'FPidx_detaction'] = {}
            dict_res[key_name + 'TNidx_detaction'] = {}
            dict_res[key_name + 'FNidx_detaction'] = {}
            dict_res[key_name + 'TPidx_classify'] = {}
            dict_res[key_name + 'FPidx_classify'] = {}
            dict_res[key_name + 'TNidx_classify'] = {}
            dict_res[key_name + 'FNidx_classify'] = {}
            conf_mat_res = np.zeros((2, 2))  # For detection
            classify_conf_mat_res = np.zeros((len(labels)+1, len(labels)+1))
            distance_tmp = []
            distance_avgimage_tmp = np.zeros(num_img)

            if eval_class == True:
                classify_pre_tmp, classify_rec_tmp, classify_f1_tmp = np.zeros(
                    num_img), np.zeros(num_img), np.zeros(num_img)
                classify_spe_tmp, classify_sen_tmp, classify_acc_tmp = np.zeros(
                    num_img), np.zeros(num_img), np.zeros(num_img)
                classify_diff_tmp = np.zeros(num_img)
                # It is the same as: classify_distance_avgimage_tmp = np.zeros(num_img)
                classify_distance_tmp = np.zeros(num_img)
                classify_tmp = {}
                for il in labels:
                    classify_tmp['class'+str(il)] = {}
                    classify_tmp['class'+str(il)]['TP'] = np.zeros(num_img)
                    classify_tmp['class'+str(il)]['FP'] = np.zeros(num_img)
                    classify_tmp['class'+str(il)]['FN'] = np.zeros(num_img)
                    classify_tmp['class'+str(il)]['TN'] = np.zeros(num_img)
                    classify_tmp['class' +
                                 str(il)]['precision'] = np.zeros(num_img)
                    classify_tmp['class'+str(il)]['recall'] = np.zeros(num_img)
                    classify_tmp['class'+str(il)]['f1'] = np.zeros(num_img)
                    classify_tmp['class' +
                                 str(il)]['specificity'] = np.zeros(num_img)
                    classify_tmp['class' +
                                 str(il)]['sensitivity'] = np.zeros(num_img)
                    classify_tmp['class' +
                                 str(il)]['accuracy'] = np.zeros(num_img)
                    classify_tmp['class'+str(il)]['diff'] = np.zeros(num_img)
                    classify_tmp['class'+str(il)]['distance'] = []

            print('now done the eval')

            for idx, imgname in enumerate(img_pool):

                resultDictPath = os.path.join(resfolder, imgname + '.mat')
                gtPath = _find_mat_file(imgfolder, imgname, matExt)
                if gtPath is None:
                    print(f"Missing ground-truth mat for image {imgname}; skipping.")
                    continue

                loaded_mt = loadmat(gtPath)
                gt = _centers_to_row_col(loaded_mt['Centers'])
                gt_label = np.squeeze(loaded_mt[contourlabel]).astype(int)

                try:
                    resultDict = loadmat(resultDictPath)
                    # key_name = get_seed_name(thresh, maxlen)
                    key_label = key_name + '_label'
                    # pdb.set_trace()
                    thisresult = resultDict[key_name]
                    thisresult_label = np.squeeze(resultDict[key_label]).astype(int)
                    detectmap = resultDict['detectmap']
                    num_pixels = detectmap.shape[0] * detectmap.shape[1]
                except (OSError, FileNotFoundError, ValueError) as e:
                    print(f"⚠️ Skipping corrupted file: {resultDictPath}")
                    print(f"   Error: {e}")
                    continue

                imgpath = img_list[idx]
                if thisresult.shape[0] == 0:
                    print("Zero detected seeds for img:{s} for parameter t_{thd:3.2f}_r_{rad:3.2f}".format(
                        s=imgname, thd=thresh, rad=maxlen))
                    thisresult = np.array([[-1000, -1000]])
                    thisresult_label = np.array([-1])
                if gt.shape[0] == 0:
                    print("Zero ground truth seeds for img:{s} for parameter t_{thd:3.2f}_r_{rad:3.2f}".format(
                        s=imgname, thd=thresh, rad=maxlen))
                    gt = np.array([[-1000, -1000]])
                    gt_label = np.array([-1])

                if eval_class == True:
                    print('now detect classify eval')

                    performdict, (valid_res, valid_gt, idx_FP, idx_FN), (idx_TP_classify, idx_TN_classify, idx_FP_classify, idx_FN_classify) = detectclassify_eval(
                        thisresult, thisresult_label, gt, gt_label, radius, labels, labelweights, num_pixels)
                    print('now done detect classify eval')
                    # weighted performance for one image
                    classify_pre_tmp[idx] = performdict['classify']['precision']
                    classify_rec_tmp[idx] = performdict['classify']['recall']
                    classify_f1_tmp[idx] = performdict['classify']['f1']
                    classify_spe_tmp[idx] = performdict['classify']['specificity']
                    classify_sen_tmp[idx] = performdict['classify']['sensitivity']
                    classify_acc_tmp[idx] = performdict['classify']['accuracy']
                    classify_diff_tmp[idx] = performdict['classify']['diff']
                    classify_distance_tmp[idx] = performdict['classify']['distance']
                    classify_conf_mat_res += performdict['classify']['conf_mat']
                    dict_res[key_name +
                             'TPidx_classify'][imgname] = idx_TP_classify
                    dict_res[key_name +
                             'FPidx_classify'][imgname] = idx_FP_classify
                    dict_res[key_name +
                             'TNidx_classify'][imgname] = idx_TN_classify
                    dict_res[key_name +
                             'FNidx_classify'][imgname] = idx_FN_classify

                    for il in labels:
                        classify_tmp['class' +
                                     str(il)]['TP'][idx] = performdict['class'+str(il)]['TP']
                        classify_tmp['class' +
                                     str(il)]['FP'][idx] = performdict['class'+str(il)]['FP']
                        classify_tmp['class' +
                                     str(il)]['FN'][idx] = performdict['class'+str(il)]['FN']
                        classify_tmp['class' +
                                     str(il)]['TN'][idx] = performdict['class'+str(il)]['TN']
                        classify_tmp['class'+str(
                            il)]['precision'][idx] = performdict['class'+str(il)]['precision']
                        classify_tmp['class' +
                                     str(il)]['recall'][idx] = performdict['class'+str(il)]['recall']
                        classify_tmp['class' +
                                     str(il)]['f1'][idx] = performdict['class'+str(il)]['f1']
                        classify_tmp['class'+str(
                            il)]['specificity'][idx] = performdict['class'+str(il)]['specificity']
                        classify_tmp['class'+str(
                            il)]['sensitivity'][idx] = performdict['class'+str(il)]['sensitivity']
                        classify_tmp['class'+str(
                            il)]['accuracy'][idx] = performdict['class'+str(il)]['accuracy']
                        classify_tmp['class' +
                                     str(il)]['diff'][idx] = performdict['class'+str(il)]['diff']
                        classify_tmp['class'+str(il)]['distance'].append(
                            performdict['class'+str(il)]['distance'])
                else:
                    performdict, (valid_res, valid_gt, idx_FP, idx_FN) = detect_eval(
                        thisresult, gt, radius, num_pixels)

                TP_res[idx, th_idx, len_idx] = performdict['detect']['TP']
                FP_res[idx, th_idx, len_idx] = performdict['detect']['FP']
                FN_res[idx, th_idx, len_idx] = performdict['detect']['FN']
                TN_res[idx, th_idx, len_idx] = performdict['detect']['TN']
                pre_res[idx, th_idx, len_idx] = performdict['detect']['precision']
                rec_res[idx, th_idx, len_idx] = performdict['detect']['recall']
                f1_res[idx, th_idx, len_idx] = performdict['detect']['f1']
                spe_res[idx, th_idx,
                        len_idx] = performdict['detect']['specificity']
                diff_res[idx, th_idx, len_idx] = performdict['detect']['diff']
                distance_tmp.append(performdict['detect']['distance'])
                conf_mat_res += performdict['detect']['conf_mat']
                dict_res[key_name + 'TPidx_detaction'][imgname] = valid_res
                dict_res[key_name + 'FPidx_detaction'][imgname] = idx_FP
                dict_res[key_name + 'TNidx_detaction'][imgname] = valid_gt
                dict_res[key_name + 'FNidx_detaction'][imgname] = idx_FN

            print('done the do iteration')

            # detection performance
            # classify_dict, classify_avgimage_dict = None, None
            detect_dict, detect_avgimage_dict = {}, {}
            detect_dict['TP'] = np.sum(TP_res[:, th_idx, len_idx])
            detect_dict['FP'] = np.sum(FP_res[:, th_idx, len_idx])
            detect_dict['FN'] = np.sum(FN_res[:, th_idx, len_idx])
            detect_dict['TN'] = np.sum(TN_res[:, th_idx, len_idx])
            detect_dict['conf_mat'] = conf_mat_res.tolist()
            detect_dict['precision'] = float(np.sum(TP_res[:, th_idx, len_idx]))/(
                np.sum(TP_res[:, th_idx, len_idx]) + np.sum(FP_res[:, th_idx, len_idx]) + 1e-10)
            detect_dict['recall'] = float(np.sum(TP_res[:, th_idx, len_idx]))/(
                np.sum(TP_res[:, th_idx, len_idx]) + np.sum(FN_res[:, th_idx, len_idx]) + 1e-10)
            detect_dict['f1'] = (2.0*detect_dict['precision']*detect_dict['recall'])/(
                detect_dict['precision']+detect_dict['recall'] + 1e-10)
            detect_dict['specificity'] = float(np.sum(TN_res[:, th_idx, len_idx]))/(
                np.sum(TN_res[:, th_idx, len_idx]) + np.sum(FP_res[:, th_idx, len_idx]) + 1e-10)
            detect_dict['sensitivity'] = detect_dict['recall']
            detect_dict['accuracy'] = float(detect_dict['TP'] + detect_dict['TN']) / (
                detect_dict['TP'] + detect_dict['TN'] + detect_dict['FP'] + detect_dict['FN'] + 1e-10)
            detect_dict['diff'] = np.mean(diff_res[:, th_idx, len_idx])
            detect_dict['diff_std'] = np.std(diff_res[:, th_idx, len_idx])
            distance_cat = _safe_concat(distance_tmp)
            # distance_res[th_idx][len_idx] = distance_cat
            detect_dict['distance'] = _safe_mean(distance_cat)  # distance average
            detect_dict['distance_std'] = _safe_std(distance_cat)

            classify_dict = {}

            if eval_class == True:
                # classification performance
                classify_dict['precision'] = 0.0
                classify_dict['recall'] = 0.0
                classify_dict['f1'] = 0.0
                classify_dict['specificity'] = 0.0
                classify_dict['sensitivity'] = 0.0
                classify_dict['accuracy'] = 0.0
                classify_dict['diff'] = 0.0
                classify_dict['diff_std'] = 0.0
                classify_dict['distance'] = 0.0
                classify_dict['distance_std'] = 0.0
                classify_dict['conf_mat'] = classify_conf_mat_res.tolist()
                for il in labels:
                    TP_tmp = np.sum(classify_tmp['class'+str(il)]['TP'])
                    FP_tmp = np.sum(classify_tmp['class'+str(il)]['FP'])
                    FN_tmp = np.sum(classify_tmp['class'+str(il)]['FN'])
                    TN_tmp = np.sum(classify_tmp['class'+str(il)]['TN'])
                    pre_tmp = float(TP_tmp)/(TP_tmp + FP_tmp + 1e-10)
                    rec_tmp = float(TP_tmp)/(TP_tmp + FN_tmp + 1e-10)
                    f1_tmp = (2.0*pre_tmp*rec_tmp)/(pre_tmp+rec_tmp + 1e-10)
                    sen_tmp = rec_tmp  # sensitiviy equals to recall
                    # specificity (maybe this does not make sense for multi-class)
                    spe_tmp = float(TN_tmp)/(FP_tmp + TN_tmp + 1e-10)
                    acc_tmp = float(TP_tmp + TN_tmp) / \
                        (TP_tmp + FP_tmp + TN_tmp + FN_tmp + 1e-10)

                    classify_dict['class'+str(il)] = {}
                    classify_dict['class'+str(il)]['TP'] = TP_tmp
                    classify_dict['class'+str(il)]['FP'] = FP_tmp
                    classify_dict['class'+str(il)]['FN'] = FN_tmp
                    classify_dict['class'+str(il)]['TN'] = TN_tmp
                    classify_dict['class'+str(il)]['precision'] = pre_tmp
                    classify_dict['class'+str(il)]['recall'] = rec_tmp
                    classify_dict['class'+str(il)]['f1'] = f1_tmp
                    classify_dict['class'+str(il)]['specificity'] = spe_tmp
                    classify_dict['class'+str(il)]['sensitivity'] = sen_tmp
                    classify_dict['class'+str(il)]['accuracy'] = acc_tmp
                    classify_dict['class'+str(il)]['diff'] = np.mean(
                        classify_tmp['class'+str(il)]['diff'])
                    classify_dict['class'+str(il)]['diff_std'] = np.std(
                        classify_tmp['class'+str(il)]['diff'])
                    classify_distance_cat = _safe_concat(
                        classify_tmp['class'+str(il)]['distance'])
                    classify_dict['class' +
                                  str(il)]['distance'] = _safe_mean(classify_distance_cat)
                    classify_dict['class' +
                                  str(il)]['distance_std'] = _safe_std(classify_distance_cat)

                    classify_dict['precision'] += labelweights[il] * pre_tmp
                    classify_dict['recall'] += labelweights[il] * rec_tmp
                    classify_dict['f1'] += labelweights[il] * f1_tmp
                    classify_dict['specificity'] += labelweights[il] * spe_tmp
                    classify_dict['sensitivity'] += labelweights[il] * sen_tmp
                    classify_dict['accuracy'] += labelweights[il] * acc_tmp
                    classify_dict['diff'] += labelweights[il] * \
                        classify_dict['class'+str(il)]['diff']
                    classify_dict['diff_std'] += labelweights[il] * \
                        classify_dict['class'+str(il)]['diff_std']
                    classify_dict['distance'] += labelweights[il] * \
                        classify_dict['class'+str(il)]['distance']
                    classify_dict['distance_std'] += labelweights[il] * \
                        classify_dict['class'+str(il)]['distance_std']

            # t=threshold, r=radius, gtr=ground-truth radius (used to define ground-truth region)
            marker = 't_' + '{:01.02f}'.format(thresh) + '_r_' + '{:02d}'.format(
                maxlen) + '_gtr_' + '{:02.02f}'.format(radius)
            res_json[marker] = {
                "detect_dict": detect_dict,
                "classify_dict": classify_dict
            }

    # Save TP, FP, TN, and FN indices for resulted image printing

    for idx, imgname in enumerate(img_pool):
        resultDictPath = os.path.join(resfolder, imgname + '.mat')
        resultDict = loadmat(resultDictPath)
        gtPath = _find_mat_file(imgfolder, imgname, matExt)
        if gtPath is None:
            print(f"Missing ground-truth mat for image {imgname}; skipping annotation copy.")
            continue
        gtDict = loadmat(gtPath)

        resultDict['gt_Centers'] = gtDict['Centers']
        resultDict['gt_Labels'] = gtDict[contourlabel]

        for th_idx, thresh in enumerate(thresh_pool):
            for len_idx, maxlen in enumerate(len_pool):
                key_name = get_seed_name(thresh, maxlen)
                resultDict[key_name + 'TPidx_detaction'] = dict_res[key_name +
                                                                    'TPidx_detaction'][imgname]
                resultDict[key_name + 'FPidx_detaction'] = dict_res[key_name +
                                                                    'FPidx_detaction'][imgname]
                resultDict[key_name + 'TNidx_detaction'] = dict_res[key_name +
                                                                    'TNidx_detaction'][imgname]
                resultDict[key_name + 'FNidx_detaction'] = dict_res[key_name +
                                                                    'FNidx_detaction'][imgname]
                if eval_class == True:
                    resultDict[key_name + 'TPidx_classify'] = dict_res[key_name +
                                                                       'TPidx_classify'][imgname]
                    resultDict[key_name + 'FPidx_classify'] = dict_res[key_name +
                                                                       'FPidx_classify'][imgname]
                    resultDict[key_name + 'TNidx_classify'] = dict_res[key_name +
                                                                       'TNidx_classify'][imgname]
                    resultDict[key_name + 'FNidx_classify'] = dict_res[key_name +
                                                                       'FNidx_classify'][imgname]
        sio.savemat(resultDictPath, resultDict)

    with open(save_path, 'w') as outfile:
        json.dump(res_json, outfile)


def detect_eval(res, gt, radius, num_pixels):
    '''
    res: N*2 tensor for (row, col) seeds.
    gt:  N*2 tensor for (row, col) gt seeds.
    '''

    num_det = res.shape[0]
    num_gt = gt.shape[0]
    valid_row, valid_col, idx_FP, idx_FN = graph_match(res, gt, radius)

    TP = len(valid_row)
    FP = num_det - len(valid_row)
    FN = num_gt - len(valid_row)
    # For object detection, every pixel that does not correspond to a nucleus is considered negative, and true negative means this pixel is predicted as negative
    TN = num_pixels - TP - FN - FP

    pre = float(TP)/(TP + FP + 1e-10)
    rec = float(TP)/(TP + FN + 1e-10)  # sensitivity = recall
    f1 = (2.0*pre*rec)/(pre+rec + 1e-10)
    spe = float(TN)/(FP + TN + 1e-10)
    sen = rec
    acc = float(TP + TN)/(TP + FP + TN + FN + 1e-10)

    matched_res = res[valid_row]
    matched_gt = gt[valid_col]

    difference = abs(num_det - num_gt)
    distance = np.sqrt(np.sum((matched_gt - matched_res)**2, axis=1))

    performdict = {}
    # overall performance for one image
    performdict['detect'] = {}
    performdict['detect']['TP'] = TP
    performdict['detect']['FP'] = FP
    performdict['detect']['FN'] = FN
    performdict['detect']['TN'] = TN
    performdict['detect']['conf_mat'] = np.array([[TP, FP], [FN, TN]])
    performdict['detect']['precision'] = pre
    performdict['detect']['recall'] = rec
    performdict['detect']['f1'] = f1
    performdict['detect']['specificity'] = spe
    performdict['detect']['sensitiviy'] = sen
    performdict['detect']['accuracy'] = acc
    performdict['detect']['diff'] = difference
    performdict['detect']['distance'] = distance  # N x 1
    return performdict, (valid_row, valid_col, idx_FP, idx_FN)


def detectclassify_eval(res, res_label, gt, gt_label, radius, labels=[1, 2, 3], labelweights=None, num_pixels=None):
    if labelweights is None:
        labelweights = {}
        for il in labels:
            labelweights[il] = 1.0/len(labels)

    assert res.shape[0] == res_label.shape[0], "labels and automatic results do not match."
    assert gt.shape[0] == gt_label.shape[0], "labels and gold standard do not match."
    num_det = res.shape[0]
    num_gt = gt.shape[0]
    valid_row, valid_col, idx_FP, idx_FN = graph_match(res, gt, radius)
    matched_res = res[valid_row]
    matched_res_label = res_label[valid_row]
    matched_gt = gt[valid_col]
    matched_gt_label = gt_label[valid_col]

    performdict = {}
    # overall performance for one image
    performdict['detect'] = {}
    TP = len(valid_row)
    FP = num_det - len(valid_row)
    FN = num_gt - len(valid_row)
    TN = num_pixels - TP - FN - FP
    pre = float(TP)/(TP + FP + 1e-10)
    rec = float(TP)/(TP + FN + 1e-10)
    f1 = (2.0*pre*rec)/(pre+rec + 1e-10)
    spe = float(TN)/(TN + FP + 1e-10)
    sen = rec
    acc = float(TP + TN)/(TP + FP + TN + FN + 1e-10)
    difference = abs(num_det - num_gt)
    distance = np.sqrt(np.sum((matched_gt - matched_res)**2, axis=1))
    performdict['detect']['TP'] = TP
    performdict['detect']['FP'] = FP
    performdict['detect']['FN'] = FN
    performdict['detect']['TN'] = TN
    performdict['detect']['conf_mat'] = np.array([[TP, FP], [FN, TN]])
    performdict['detect']['precision'] = pre
    performdict['detect']['recall'] = rec
    performdict['detect']['f1'] = f1
    performdict['detect']['specificity'] = spe
    performdict['detect']['sensitivity'] = sen  # equal to recall
    performdict['detect']['accuracy'] = acc  # equal to recall
    performdict['detect']['diff'] = difference
    performdict['detect']['distance'] = distance

    # each class for one image
    performdict['classify'] = {}
    performdict['classify']['conf_mat'] = np.zeros(
        (len(labels)+1, len(labels)+1))
    performdict['classify']['precision'] = 0.0
    performdict['classify']['recall'] = 0.0
    performdict['classify']['f1'] = 0.0
    performdict['classify']['specificity'] = 0.0
    performdict['classify']['sensitivity'] = 0.0
    performdict['classify']['accuracy'] = 0.0
    performdict['classify']['diff'] = 0.0
    performdict['classify']['distance'] = 0.0
    performdict['classify']['distance_std'] = 0.0
    idx_TP_classify, idx_TN_classify, idx_FP_classify, idx_FN_classify = np.array(
        [], dtype=int), np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=int)
    for il in labels:
        inum_res = res_label[res_label == il].size
        inum_gt = gt_label[gt_label == il].size
        TP = np.sum(matched_gt_label[matched_res_label == il] == il)
        FP = inum_res - TP
        FN = inum_gt - TP
        TN = num_pixels - TP - FN - FP
        pre = float(TP)/(TP + FP + 1e-10)
        rec = float(TP)/(TP + FN + 1e-10)
        f1 = (2.0*pre*rec)/(pre+rec + 1e-10)
        sen = rec  # sensitiviy equals to recall
        spe = float(TN)/(FP + TN + 1e-10)
        acc = float(TP + TN)/(TP + FP + TN + FN + 1e-10)
        performdict['class'+str(il)] = {}
        performdict['class'+str(il)]['TP'] = TP
        performdict['class'+str(il)]['FP'] = FP
        performdict['class'+str(il)]['FN'] = FN
        performdict['class'+str(il)]['TN'] = TN
        performdict['class'+str(il)]['precision'] = pre
        performdict['class'+str(il)]['recall'] = rec
        performdict['class'+str(il)]['f1'] = f1
        performdict['class'+str(il)]['specificity'] = spe
        performdict['class'+str(il)]['sensitivity'] = sen
        performdict['class'+str(il)]['accuracy'] = acc
        performdict['class'+str(il)]['diff'] = abs(inum_res - inum_gt)
        imatched_gt = matched_gt[matched_res_label == il, :]
        imatched_gt = imatched_gt[matched_gt_label[matched_res_label == il] == il, :]
        imatched_res = matched_res[matched_gt_label == il, :]
        imatched_res = imatched_res[matched_res_label[matched_gt_label == il] == il, :]
        performdict['class'+str(il)]['distance'] = np.sqrt(
            np.sum((imatched_gt - imatched_res)**2, axis=1))  # N x 1

        performdict['classify']['precision'] += labelweights[il] * \
            pre  # overall precision for one image
        performdict['classify']['recall'] += labelweights[il] * rec
        performdict['classify']['f1'] += labelweights[il] * f1
        performdict['classify']['specificity'] += labelweights[il] * spe
        performdict['classify']['sensitivity'] += labelweights[il] * sen
        performdict['classify']['accuracy'] += labelweights[il] * acc
        performdict['classify']['diff'] += labelweights[il] * \
            performdict['class'+str(il)]['diff']
        performdict['classify']['distance'] += labelweights[il] * \
            _safe_mean(performdict['class'+str(il)]['distance'])
        performdict['classify']['distance_std'] += labelweights[il] * \
            _safe_std(performdict['class'+str(il)]['distance'])

        # length of valid_row = length of matched_res_label
        ivalid_row = valid_row[matched_gt_label == il]
        idx_TP_classify = np.append(
            idx_TP_classify, ivalid_row[matched_res_label[matched_gt_label == il] == il])
        idx_FP_classify = np.append(
            idx_FP_classify, ivalid_row[matched_res_label[matched_gt_label == il] != il])

        ivalid_col = valid_col[matched_res_label == il]
        idx_TN_classify = np.append(
            idx_TN_classify, ivalid_col[matched_gt_label[matched_res_label == il] == il])
        idx_FN_classify = np.append(
            idx_FN_classify, ivalid_col[matched_gt_label[matched_res_label == il] != il])
        for il_gt in labels:
            performdict['classify']['conf_mat'][il-1, il_gt -
                                                1] = np.sum(matched_gt_label[matched_res_label == il] == il_gt)
        performdict['classify']['conf_mat'][il-1, -1] = inum_res - \
            np.sum(performdict['classify']['conf_mat'][il-1, 0:-1])

    # add those detections/classificatons that do not associated ground truths
    idx_FP_classify = np.append(idx_FP_classify, idx_FP)
    # add those ground truths that do not associated detections/classificatons
    idx_FN_classify = np.append(idx_FN_classify, idx_FN)
    for il_gt in labels:
        performdict['classify']['conf_mat'][-1, il_gt-1] = gt_label[gt_label ==
                                                                    il_gt].size - np.sum(performdict['classify']['conf_mat'][0:-1, il_gt-1])
    performdict['classify']['conf_mat'][-1, -1] = 0
    performdict['classify']['conf_mat'][-1, -1] = num_pixels - \
        np.sum(performdict['classify']['conf_mat'])
    print(performdict['classify']['conf_mat'])
    return performdict, (valid_row, valid_col, idx_FP, idx_FN), (idx_TP_classify, idx_TN_classify, idx_FP_classify, idx_FN_classify)


def graph_match(res, gt, radius):
    '''
    Parameters
    ----------
    res: N*2
    gt:  N*2
    '''
    res = np.asarray(res).reshape(-1, 2)
    gt = np.asarray(gt).reshape(-1, 2)
    num_det = res.shape[0]
    num_gt = gt.shape[0]

    if num_det == 0 or num_gt == 0:
        return (
            np.asarray([], dtype=int),
            np.asarray([], dtype=int),
            np.arange(num_det, dtype=int),
            np.arange(num_gt, dtype=int),
        )

    distmatrix = pairwise_distances(res, gt, metric='euclidean')
    row_ind, col_ind = linear_sum_assignment(distmatrix)

    valid_mask = distmatrix[row_ind, col_ind] <= radius
    valid_col = col_ind[valid_mask]
    valid_row = row_ind[valid_mask]
    idx_res = np.array(range(num_det))
    idx_gt = np.array(range(num_gt))
    idx_FP = np.delete(idx_res, valid_row)
    idx_FN = np.delete(idx_gt, valid_col)
    return valid_row, valid_col, idx_FP, idx_FN
