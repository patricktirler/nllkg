from argparse import ArgumentParser
from typing import Dict, List, Optional, Sequence, Union
import os.path as osp
from rich.progress import track
import numpy as np
from mmengine.logging import print_log
import mmcv
import mmengine
from mmdet.structures import DetDataSample
from mmdet.apis import DetInferencer

InputType = Union[str, np.ndarray]
InputsType = Union[InputType, Sequence[InputType]]
PredType = List[DetDataSample]
ImgType = Union[np.ndarray, Sequence[np.ndarray]]


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        'inputs', type=str, help='Input image file or folder path.')
    parser.add_argument(
        'model',
        type=str,
        help='Config or checkpoint .pth file or the model name '
        'and alias defined in metafile. The model configuration '
        'file will try to read from .pth if the parameter is '
        'a .pth weights file.')
    parser.add_argument('--weights', default=None, help='Checkpoint file')
    parser.add_argument(
        '--out-dir',
        type=str,
        default='outputs',
        help='Output directory of images or prediction results.')
    parser.add_argument('--texts', help='text prompt')
    parser.add_argument('--relation-texts', help='relation text prompt')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--pred-score-thr',
        type=float,
        default=0.3,
        help='bbox score threshold')
    parser.add_argument(
        '--batch-size', type=int, default=1, help='Inference batch size.')
    parser.add_argument(
        '--show',
        action='store_true',
        help='Display the image in a popup window.')
    parser.add_argument(
        '--no-save-vis',
        action='store_true',
        help='Do not save detection vis results')
    parser.add_argument(
        '--no-save-pred',
        action='store_true',
        help='Do not save detection json results')
    parser.add_argument(
        '--print-result',
        action='store_true',
        help='Whether to print the results.')
    parser.add_argument(
        '--palette',
        default='none',
        choices=['coco', 'voc', 'citys', 'random', 'none'],
        help='Color palette used for visualization')
    parser.add_argument(
        '--custom-entities',
        '-c',
        action='store_true',
        help='Whether to customize entity names? '
        'If so, the input text should be '
        '"cls_name1 . cls_name2 . cls_name3 ." format')

    call_args = vars(parser.parse_args())

    if call_args['no_save_vis'] and call_args['no_save_pred']:
        call_args['out_dir'] = ''

    if call_args['model'].endswith('.pth'):
        print_log('The model is a weight file, automatically '
                  'assign the model to --weights')
        call_args['weights'] = call_args['model']
        call_args['model'] = None

    init_kws = ['model', 'weights', 'device', 'palette']
    init_args = {}
    for init_kw in init_kws:
        init_args[init_kw] = call_args.pop(init_kw)

    return init_args, call_args

def predinstances2dict(pred_instances: PredType) -> dict:
    """Convert prediction instances to a dictionary format.

    Args:
        pred_instances (PredType): The prediction instances.

    Returns:
        dict: The converted dictionary.
    """
    pred_instances = pred_instances.numpy()
    return {
        'keypoint_label_names': [l.strip() for l in pred_instances.label_names],
        'keypoint_labels': pred_instances.labels.tolist(),
        'keypoint_scores': pred_instances.scores.tolist(),
        'keypoint_coords': pred_instances.keypoints.tolist(),
        'keypoint_relation_scores': pred_instances.relation_scores.tolist(),
    }


class OpenVocPoseInferencer(DetInferencer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(
            self,
            inputs: InputsType,
            batch_size: int = 1,
            return_vis: bool = False,
            show: bool = False,
            wait_time: int = 0,
            no_save_vis: bool = False,
            draw_pred: bool = True,
            pred_score_thr: float = 0.3,
            return_datasamples: bool = False,
            print_result: bool = False,
            no_save_pred: bool = True,
            out_dir: str = '',
            texts: Optional[Union[str, list]] = None,
            relation_texts: Optional[Union[str, list]] = None,
            stuff_texts: Optional[Union[str, list]] = None,
            custom_entities: bool = False,
            **kwargs) -> dict:
        """Call the inferencer.

        Args:
            inputs (InputsType): Inputs for the inferencer.
            batch_size (int): Inference batch size. Defaults to 1.
            show (bool): Whether to display the visualization results in a
                popup window. Defaults to False.
            wait_time (float): The interval of show (s). Defaults to 0.
            no_save_vis (bool): Whether to force not to save prediction
                vis results. Defaults to False.
            draw_pred (bool): Whether to draw predicted bounding boxes.
                Defaults to True.
            pred_score_thr (float): Minimum score of bboxes to draw.
                Defaults to 0.3.
            return_datasamples (bool): Whether to return results as
                :obj:`DetDataSample`. Defaults to False.
            print_result (bool): Whether to print the inference result w/o
                visualization to the console. Defaults to False.
            no_save_pred (bool): Whether to force not to save prediction
                results. Defaults to True.
            out_dir: Dir to save the inference results or
                visualization. If left as empty, no file will be saved.
                Defaults to ''.
            texts (str | list[str]): Text prompts. Defaults to None.
            relation_texts (str | list[str]): Relation text prompts. Defaults to None.
            stuff_texts (str | list[str]): Stuff text prompts of open
                panoptic task. Defaults to None.
            custom_entities (bool): Whether to use custom entities.
                Defaults to False. Only used in GLIP.
            **kwargs: Other keyword arguments passed to :meth:`preprocess`,
                :meth:`forward`, :meth:`visualize` and :meth:`postprocess`.
                Each key in kwargs should be in the corresponding set of
                ``preprocess_kwargs``, ``forward_kwargs``, ``visualize_kwargs``
                and ``postprocess_kwargs``.

        Returns:
            dict: Inference and visualization results.
        """
        (
            preprocess_kwargs,
            forward_kwargs,
            visualize_kwargs,
            postprocess_kwargs,
        ) = self._dispatch_kwargs(**kwargs)

        ori_inputs = self._inputs_to_list(inputs)

        if texts is not None and isinstance(texts, str):
            texts = [texts] * len(ori_inputs)
        if relation_texts is not None and isinstance(relation_texts, str):
            relation_texts = [relation_texts] * len(ori_inputs)
        if stuff_texts is not None and isinstance(stuff_texts, str):
            stuff_texts = [stuff_texts] * len(ori_inputs)
        if texts is not None:
            assert len(texts) == len(ori_inputs)
            for i in range(len(texts)):
                if isinstance(ori_inputs[i], str):
                    ori_inputs[i] = {
                        'text': texts[i],
                        'img_path': ori_inputs[i],
                        'custom_entities': custom_entities
                    }
                else:
                    ori_inputs[i] = {
                        'text': texts[i],
                        'img': ori_inputs[i],
                        'custom_entities': custom_entities
                    }
        if stuff_texts is not None:
            assert len(stuff_texts) == len(ori_inputs)
            for i in range(len(stuff_texts)):
                ori_inputs[i]['stuff_text'] = stuff_texts[i]
        if relation_texts is not None:
            assert len(relation_texts) == len(ori_inputs)
            for i in range(len(relation_texts)):
                ori_inputs[i]['relation_text'] = relation_texts[i]

        inputs = self.preprocess(
            ori_inputs, batch_size=batch_size, **preprocess_kwargs)

        results_dict = {'predictions': [], 'visualization': []}
        for ori_imgs, data in (track(inputs, description='Inference')
                               if self.show_progress else inputs):
            preds = self.forward(data, **forward_kwargs)
            visualization = self.visualize(
                ori_imgs,
                preds,
                return_vis=return_vis,
                show=show,
                wait_time=wait_time,
                draw_pred=draw_pred,
                pred_score_thr=pred_score_thr,
                no_save_vis=no_save_vis,
                img_out_dir=out_dir,
                **visualize_kwargs)
            results = self.postprocess(
                preds,
                visualization,
                return_datasamples=return_datasamples,
                print_result=print_result,
                no_save_pred=no_save_pred,
                pred_out_dir=out_dir,
                **postprocess_kwargs)
            results_dict['predictions'].extend(results['predictions'])
            if results['visualization'] is not None:
                results_dict['visualization'].extend(results['visualization'])
        return results_dict
    
    def visualize(self,
                  inputs: InputsType,
                  preds: PredType,
                  return_vis: bool = False,
                  show: bool = False,
                  wait_time: int = 0,
                  draw_pred: bool = True,
                  pred_score_thr: float = 0.3,
                  pred_relation_score_thr: float = 0.3,
                  no_save_vis: bool = False,
                  img_out_dir: str = '',
                  **kwargs) -> Union[List[np.ndarray], None]:
        """Visualize predictions.

        Args:
            inputs (List[Union[str, np.ndarray]]): Inputs for the inferencer.
            preds (List[:obj:`DetDataSample`]): Predictions of the model.
            return_vis (bool): Whether to return the visualization result.
                Defaults to False.
            show (bool): Whether to display the image in a popup window.
                Defaults to False.
            wait_time (float): The interval of show (s). Defaults to 0.
            draw_pred (bool): Whether to draw predicted bounding boxes.
                Defaults to True.
            pred_score_thr (float): Minimum score of keypoints to draw.
                Defaults to 0.3.
            pred_relation_score_thr (float): Minimum score of relations to draw.
                Defaults to 0.3.
            no_save_vis (bool): Whether to force not to save prediction
                vis results. Defaults to False.
            img_out_dir (str): Output directory of visualization results.
                If left as empty, no file will be saved. Defaults to ''.

        Returns:
            List[np.ndarray] or None: Returns visualization results only if
            applicable.
        """
        if no_save_vis is True:
            img_out_dir = ''

        if not show and img_out_dir == '' and not return_vis:
            return None

        if self.visualizer is None:
            raise ValueError('Visualization needs the "visualizer" term'
                             'defined in the config, but got None.')

        results = []

        for single_input, pred in zip(inputs, preds):
            if isinstance(single_input, str):
                img_bytes = mmengine.fileio.get(single_input)
                img = mmcv.imfrombytes(img_bytes)
                img = img[:, :, ::-1]
                img_name = osp.basename(single_input)
            elif isinstance(single_input, np.ndarray):
                img = single_input.copy()
                img_num = str(self.num_visualized_imgs).zfill(8)
                img_name = f'{img_num}.jpg'
            else:
                raise ValueError('Unsupported input type: '
                                 f'{type(single_input)}')

            out_file = osp.join(img_out_dir, 'vis',
                                img_name) if img_out_dir != '' else None

            self.visualizer.add_datasample(
                img_name,
                img,
                pred,
                show=show,
                wait_time=wait_time,
                draw_gt=False,
                draw_pred=draw_pred,
                pred_score_thr=pred_score_thr,
                pred_relation_score_thr=pred_relation_score_thr,
                out_file=out_file,
            )
            results.append(self.visualizer.get_image())
            self.num_visualized_imgs += 1

        return results
        
    def pred2dict(self,
                  data_sample: DetDataSample,
                  pred_out_dir: str = '') -> Dict:
        
        is_save_pred = pred_out_dir != ''

        if is_save_pred and 'img_path' in data_sample:
            img_path = osp.basename(data_sample.img_path)
            img_path = osp.splitext(img_path)[0]
            out_json_path = osp.join(pred_out_dir, 'preds', img_path + '.json')
        elif is_save_pred:
            out_json_path = osp.join(pred_out_dir, 'preds', f'{self.num_predicted_imgs}.json')
            self.num_predicted_imgs += 1

        result = {}
        if 'pred_instances' in data_sample:
            result = predinstances2dict(data_sample.pred_instances)

        if is_save_pred:
            mmengine.dump(result, out_json_path)

        return result


def main():
    init_args, call_args = parse_args()
    # TODO: may consume too much memory if your input folder has a lot of images.
    inferencer = OpenVocPoseInferencer(**init_args)
    inferencer(**call_args)

    if call_args['out_dir'] != '' and not (call_args['no_save_vis'] and call_args['no_save_pred']):
        print_log(f'results have been saved at {call_args["out_dir"]}')


if __name__ == '__main__':
    main()