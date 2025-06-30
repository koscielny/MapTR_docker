
# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import mmcv
import os
import sys
import torch
import warnings
import json
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, collate, scatter
from mmcv.runner import load_checkpoint, wrap_fp16_model

from mmdet3d.apis import init_model
from mmdet3d.datasets import build_dataset
from mmdet3d.datasets.pipelines import Compose

# Import GPU monitoring utilities
sys.path.append('/app')  # Add runpod_docker path
try:
    from gpu_utils import setup_gpu_monitoring, monitor_memory_usage, cleanup_and_monitor
    GPU_MONITORING_AVAILABLE = True
except ImportError:
    print("Warning: GPU monitoring utilities not available")
    GPU_MONITORING_AVAILABLE = False
    def setup_gpu_monitoring(): pass
    def monitor_memory_usage(stage): pass
    def cleanup_and_monitor(): pass

def parse_args():
    parser = argparse.ArgumentParser(description='MMDet3D demo for MapTR')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--sample-token',
        required=True,
        help='The nuScenes sample token for inference.')
    parser.add_argument(
        '--dataroot',
        default='data/nuscenes',
        help='Root path of the nuScenes dataset.')
    parser.add_argument(
        '--out-file', 
        required=True, 
        help='Path to save the inference results in JSON format.')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    # Setup GPU monitoring
    setup_gpu_monitoring()

    # --- 1. 初始化模型和配置 ---
    print("Initializing MapTR model...")
    monitor_memory_usage("before_model_initialization")
    cfg = Config.fromfile(args.config)
    
    # 导入插件模块
    if hasattr(cfg, 'plugin'):
        if cfg.plugin:
            import importlib
            if hasattr(cfg, 'plugin_dir'):
                plugin_dir = cfg.plugin_dir
                _module_dir = os.path.dirname(plugin_dir)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]
                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(f"Importing plugin: {_module_path}")
                importlib.import_module(_module_path)
            else:
                raise ImportError("`plugin_dir` is not specified in config.")

    # 构建并加载模型
    model = init_model(cfg, args.checkpoint, device=args.device)
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)
    model.eval()
    print("MapTR model loaded successfully.")

    # --- 2. 准备输入数据 ---
    print(f"Preparing input data for sample_token: {args.sample_token}...")
    
    cfg.data.test.test_mode = True
    cfg.data.test.data_root = args.dataroot
    if not os.path.isabs(cfg.data.test.ann_file):
        cfg.data.test.ann_file = os.path.join(args.dataroot, cfg.data.test.ann_file)
    
    if 'dataset' in cfg.data.test and cfg.data.test.dataset.type == 'CBGSDataset':
        cfg.data.test = cfg.data.test.dataset

    dataset = build_dataset(cfg.data.test)
    
    sample_idx = -1
    for i, info in enumerate(dataset.data_infos):
        if info['token'] == args.sample_token:
            sample_idx = i
            break
    
    if sample_idx == -1:
        raise ValueError(f"Error: sample_token not found in dataset: {args.sample_token}")

    data = dataset[sample_idx]
    data_batch = collate([data], samples_per_gpu=1)
    
    if args.device != 'cpu':
        data_batch = scatter(data_batch, [args.device])[0]
    else:
        for key, val in data_batch.items():
            if isinstance(val[0], mmcv.parallel.DataContainer):
                data_batch[key] = val[0].data

    # --- 3. 执行推理 ---
    print(f"Running inference on sample_token: {args.sample_token}...")
    monitor_memory_usage("before_inference")
    
    with torch.no_grad():
        result = model(return_loss=False, rescale=True, **data_batch)
    
    monitor_memory_usage("after_inference")
    print("Inference completed.")

    # --- 4. 处理并保存输出 ---
    # MapTR 的输出格式：result[0]['pts_bbox'] 包含检测结果
    map_classes = ['divider', 'ped_crossing', 'boundary']
    score_threshold = 0.3  # 置信度阈值
    
    try:
        if 'pts_bbox' in result[0]:
            result_dict = result[0]['pts_bbox']
            
            # 提取模型输出
            boxes_3d = result_dict['boxes_3d']  # [N, 4] 边界框
            scores_3d = result_dict['scores_3d']  # [N] 置信度分数
            labels_3d = result_dict['labels_3d']  # [N] 类别标签
            pts_3d = result_dict['pts_3d']        # [N, num_pts, 2] 矢量点
            
            # 转换为numpy数组以便处理
            import torch
            if isinstance(scores_3d, torch.Tensor):
                scores_3d = scores_3d.cpu().numpy()
                labels_3d = labels_3d.cpu().numpy()
                boxes_3d = boxes_3d.cpu().numpy()
                pts_3d = pts_3d.cpu().numpy()
            
            # 根据置信度阈值过滤结果
            keep = scores_3d > score_threshold
            
            output_data = []
            for i, (score, label, bbox, pts) in enumerate(zip(
                scores_3d[keep], labels_3d[keep], boxes_3d[keep], pts_3d[keep]
            )):
                # 获取类别名称
                class_name = map_classes[int(label)] if int(label) < len(map_classes) else f'class_{int(label)}'
                
                output_item = {
                    'id': int(i),
                    'class_name': class_name,
                    'class_id': int(label),
                    'confidence': float(score),
                    'bbox': bbox.tolist(),  # [xmin, ymin, xmax, ymax]
                    'pts': pts.tolist(),    # [[x1, y1], [x2, y2], ...] 矢量点序列
                    'num_pts': len(pts)
                }
                output_data.append(output_item)
            
            print(f"Found {len(output_data)} map elements with confidence > {score_threshold}")
            
        else:
            print("Warning: 'pts_bbox' key not found in model output.")
            # 尝试保存原始结果的键值，以便调试
            available_keys = list(result[0].keys()) if result else []
            output_data = {
                "error": "Unexpected output format",
                "available_keys": available_keys
            }
    
    except Exception as e:
        print(f"Error processing MapTR output: {str(e)}")
        output_data = {
            "error": f"Output processing failed: {str(e)}",
            "raw_result_keys": list(result[0].keys()) if result and len(result) > 0 else []
        }


    os.makedirs(os.path.dirname(args.out_file), exist_ok=True)
    with open(args.out_file, "w") as f:
        json.dump(output_data, f, indent=4)
    print(f"Inference results saved to: {args.out_file}")
    
    # Cleanup GPU memory and show final stats
    cleanup_and_monitor()

if __name__ == '__main__':
    main()
