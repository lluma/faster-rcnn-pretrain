from __future__ import print_function
from __future__ import absolute_import

import os
import sys
import pandas as pd
import numpy as np
import scipy.sparse
import pickle
### DEBUG
# this_dir = os.path.dirname(__file__)
# path = os.path.split(this_dir)[0]
# print ('== Add \'' + this_dir + '\' to path. ==')
# print ('== Add \'' + path + '\' to path. ==')
# sys.path.insert(0, this_dir)
# sys.path.insert(0, path)
###

from .imdb import imdb
# from datasets.imdb import imdb ### DEBUG

# >>>> obsolete, because it depends on sth outside of this project
from model.utils.config import cfg
# from model.utils.config import cfg ### DEBUG

from .ocid_eval import ocid_eval

try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3

class ocid(imdb):
    def __init__(self, image_set, data_path=None):
        imdb.__init__(self, 'ocid_' + image_set)
        self._image_set = image_set
        self._data_path = self._get_default_path() if data_path is None \
            else data_path
        self._classes = ('__background__',  # always index 0
                         'cereal_box', 'kleenex', 'food_box', 'sponge', 'bulb_box',
                         'food_can', 'toothpaste', 'marker', 'soda_can', 'apple', 'shampoo',
                         'tomato', 'pear', 'coffee_mug', 'peach', 'ball', 'orange',
                         'glue_stick', 'flashlight', 'lemon', 'banana', 'potato', 'lime',
                         'bell_pepper', 'binder', 'keyboard', 'instant_noodles', 'food_bag',
                         'hand_towel', 'stapler', 'bowl', 'wood_block', 'cracker_box',
                         'sugar_box', 'pudding_box', 'gelatin_box', 'nine-hole_peg_test',
                         'timer', 'foam_brick', 'rubik’s_cube', 'lego', 'mini_soccer_ball',
                         'chips_can', 'tomato_soup_can', 'master_chef_can', 'tennis_ball',
                         'baseball', 'racquetball', 'softball', 'golf_ball', 'pitcher_base',
                         'tuna_fish_can', 'bleach_cleanser', 'power_drill', 'mug',
                         'mustard_bottle', 'large_clamp', 'medium_clamp')
        
#         self._classes = ('__background__',  # always index 0
#                          'box', 'sponge', 'can', 'tube', 'pen', 'fruit', 'bottle', 'cup',
#                          'ball', 'glue stick', 'flashlight', 'vegetable', 'binder',
#                          'keyboard', 'bag', 'towel', 'stapler', 'bowl', 'cube', 'timer',
#                          'lego', 'power drill', 'clamp')

        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
        self._image_index = self._load_image_set_index()
        self._roidb_handler = self.gt_roidb
        self._scene_dict = self._load_scene_dict()
        
        assert os.path.exists(self._data_path), \
            'Path does not exist: {}'.format(self._data_path)
        
    def _get_default_path(self):
        """
        Return the default path where PASCAL VOC is expected to be installed.
        """
        return os.path.join(cfg.DATA_DIR, 'OCID-dataset')
    
    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        # Example path to image set file:
        # self._data_path + /train.txt
        image_set_file = os.path.join(self._data_path, self._image_set + '.txt')
        assert os.path.exists(image_set_file), \
            'Path does not exist: {}'.format(image_set_file)
        with open(image_set_file) as f:
            image_index = [x.strip() for x in f.readlines()]
        return image_index
        
    def _load_ocid_annotation(self, index):
        """
        Load image and bounding boxes info from CSV file in the OCID
        format.
        """
        filename = os.path.join(self._data_path, 'annotation.csv')
        annotations = pd.read_csv(filename)
        objs = annotations[annotations.scene_idx == int(index)].reset_index()
        num_objs = len(objs)
        
        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        seg_areas = np.zeros((num_objs), dtype=np.float32)
        
        # Load object bounding boxes into a data frame.
        for ix, obj in objs.iterrows():
            bbox = eval(obj['bbox'])
            
            x1 = float(bbox[0])
            y1 = float(bbox[1])
            x2 = float(bbox[2])
            y2 = float(bbox[3])

            cls = self._class_to_ind[obj['class'].lower().strip()]
            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0
            seg_areas[ix] = (x2 - x1 + 1) * (y2 - y1 + 1)

        overlaps = scipy.sparse.csr_matrix(overlaps)

        return {'boxes': boxes,
                'gt_classes': gt_classes,
                'gt_overlaps': overlaps,
                'flipped': False,
                'seg_areas': seg_areas}
    
    def _load_scene_dict(self):
        filename = os.path.join(self._data_path, 'scene_list.txt')
        with open(filename, 'r') as f:
            rows = f.readlines()
            f.close()
        
        scene_dict = {}
        for row in rows:
            scene_idx, scene_path = row.replace('\n', '').split(' ')
            scene_dict[scene_idx] = scene_path
        
        return scene_dict
    
    def _get_ocid_results_file_template(self):
        # OCID-dataset/results/Main/ocid_det_test_timer.txt
        filename = 'ocid_det_' + self._image_set + '_{:s}.txt'
        filedir = os.path.join(self._data_path, 'results', 'Main')
        if not os.path.exists(filedir):
            os.makedirs(filedir)
        path = os.path.join(filedir, filename)
        return path

    def _write_ocid_results_file(self, all_boxes):
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            print('Writing {} OCID results file'.format(cls))
            filename = self._get_ocid_results_file_template().format(cls)
            with open(filename, 'wt') as f:
                for im_ind, index in enumerate(self.image_index):
                    dets = all_boxes[cls_ind][im_ind]
                    if dets == []:
                        continue
                    # the OCID expects 0-based indices
                    for k in xrange(dets.shape[0]):
                        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(index, dets[k, -1],
                                       dets[k, 0], dets[k, 1],
                                       dets[k, 2], dets[k, 3]))
                        
    def _do_python_eval(self, output_dir='output'):
#         annopath = os.path.join(
#             self._devkit_path,
#             'VOC' + self._year,
#             'Annotations',
#             '{:s}.xml')
        annopath = os.path.join(self._data_path, 'annotation.csv')
        imagesetfile = os.path.join(self._data_path, self._image_set + '.txt')
        cachedir = os.path.join(self._data_path, 'annotations_cache')
        
        aps = []
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        for i, cls in enumerate(self._classes):
            if cls == '__background__':
                continue
            filename = self._get_ocid_results_file_template().format(cls)
            rec, prec, ap = ocid_eval(
                filename, annopath, imagesetfile, cls, cachedir, self._scene_dict, ovthresh=0.5)
            aps += [ap]
            print('AP for {} = {:.4f}'.format(cls, ap))
            with open(os.path.join(output_dir, cls + '_pr.pkl'), 'wb') as f:
                pickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
        print('Mean AP = {:.4f}'.format(np.mean(aps)))
        print('~~~~~~~~')
        print('Results:')
        for ap in aps:
            print('{:.3f}'.format(ap))
        print('{:.3f}'.format(np.mean(aps)))
        print('~~~~~~~~')
        print('')
        print('--------------------------------------------------------------')
        print('Results computed with the **unofficial** Python eval code.')
        print('Results should be very close to the official MATLAB eval code.')
        print('Recompute with `./tools/reval.py --matlab ...` for your paper.')
        print('-- Thanks, The Management')
        print('--------------------------------------------------------------')
        
    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])
    
    def image_id_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return i
    
    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        image_path = os.path.join(self._data_path, self._scene_dict[index])
        assert os.path.exists(image_path), \
            'Path does not exist: {}'.format(image_path)
        return image_path
    
    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = pickle.load(fid)
            print('{} gt roidb loaded from {}'.format(self.name, cache_file))
            return roidb

        gt_roidb = [self._load_ocid_annotation(index)
                    for index in self.image_index]
        with open(cache_file, 'wb') as fid:
            pickle.dump(gt_roidb, fid, pickle.HIGHEST_PROTOCOL)
        print('wrote gt roidb to {}'.format(cache_file))

        return gt_roidb
    
    def evaluate_detections(self, all_boxes, output_dir):
        self._write_ocid_results_file(all_boxes)
        self._do_python_eval(output_dir)
    
    
if __name__ == "__main__":
#     print ('Testing OCID dataset load trainset...')
#     d = ocid(image_set='train')
#     d._roidb_handler()
#     img_path = d.image_path_at(0)
    
    print ('Testing OCID dataset load testset...')
    d = ocid(image_set='test')
    d._roidb_handler()
    img_path = d.image_path_at(0)
    