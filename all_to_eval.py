# /root/autodl-tmp/batch_process.py
import cv2
import torch
import numpy as np
from ultralytics import YOLO
import sys
import os
import glob
from tqdm import tqdm
from mmcv import Config
from types import SimpleNamespace
# We still need STrack to reset the ID, but not for format conversion
from custom_byte_tracker import ByteTracker, STrack

# ==============================================================================
# 1. 导入 Metric3D 模块
# ==============================================================================
print(">>> [DEBUG] 步骤 1: 导入 Metric3D 模块...")
METRIC3D_PATH = r'C:\Users\Administrator\Desktop\autodl-tmp (1)\Metric3D'
if METRIC3D_PATH not in sys.path:
    sys.path.insert(0, METRIC3D_PATH)
try:
    from mono.model.monodepth_model import DepthModel as MonoDepthModel

    print(">>> [INFO] Metric3D 模块导入成功。")
except ImportError as e:
    print(f"!!! [ERROR] 从 Metric3D 导入模块失败: {e}")
    raise

# ==============================================================================
# 2. 配置与路径定义
# ==============================================================================
print("\n>>> [DEBUG] 步骤 2: 配置模型和文件路径...")
YOLO_MODEL_PATH = r'C:\Users\Administrator\Desktop\autodl-tmp (1)\weights\epoch30.pt'
METRIC3D_MODEL_PATH = r'C:\Users\Administrator\Desktop\autodl-tmp (1)\weights\metric_depth_vit_large_800k.pth'
METRIC3D_CONFIG_PATH = r'C:\Users\Administrator\Desktop\autodl-tmp (1)\Metric3D\mono\configs\HourglassDecoder\vit.raft5.large.py'
INPUT_VIDEOS_DIR = r'C:\Users\Administrator\Desktop\autodl-tmp (1)\kitti_videos'  # <-- MAKE SURE THIS PATH IS CORRECT
OUTPUT_EVAL_DIR = r'C:\Users\Administrator\Desktop\autodl-tmp (1)\eval_outputs3'

os.makedirs(OUTPUT_EVAL_DIR, exist_ok=True)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f">>> [INFO] 将要使用的设备: {DEVICE}")

# ==============================================================================
# 3. 模型加载 (全局加载一次)
# ==============================================================================
print("\n>>> [DEBUG] 步骤 3: 开始加载深度学习模型...")
try:
    yolo_model = YOLO(YOLO_MODEL_PATH)
    TARGET_CLASS_NAME = 'Car'
    TARGET_CLASS_ID = [k for k, v in yolo_model.names.items() if v == TARGET_CLASS_NAME][0]
    print(f">>> [INFO] 目标类别 '{TARGET_CLASS_NAME}' ID为: {TARGET_CLASS_ID}")
except Exception as e:
    print(f"!!! [ERROR] 加载 YOLOv8 模型失败: {e}")
    raise

try:
    cfg = Config.fromfile(METRIC3D_CONFIG_PATH)
    cfg.model.backbone.use_mask_token = False
    metric3d_model = MonoDepthModel(cfg).to(DEVICE)
    checkpoint = torch.load(METRIC3D_MODEL_PATH, map_location=DEVICE)
    state_dict = checkpoint.get('model_state_dict', checkpoint.get('model', checkpoint))
    metric3d_model.load_state_dict(state_dict, strict=False)
    metric3d_model.eval()
    print(">>> [SUCCESS] Metric3Dv2 模型加载成功！")
except Exception as e:
    print(f"!!! [FATAL ERROR] 加载 Metric3Dv2 模型时出错: {e}")
    raise


# ==============================================================================
# 4. 视频处理主函数
# ==============================================================================
def process_video_for_eval(input_path, output_txt_path):
    print(f"\n--- 开始处理视频: {os.path.basename(input_path)} ---")
    cap = cv2.VideoCapture(input_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    metric3d_input_size = (cfg.data_basic['vit_size'][1], cfg.data_basic['vit_size'][0])

    tracker_args = SimpleNamespace(track_high_thresh=0.5, track_low_thresh=0.1, new_track_thresh=0.6,
                                   track_buffer=30, match_thresh=0.8, mot20=False)
    tracker = ByteTracker(args=tracker_args, frame_rate=fps)
    STrack.release_id()

    # MODIFIED: Frame count now starts at 0 for KITTI format
    frame_count = 0
    with open(output_txt_path, 'w') as f_out:
        with tqdm(total=total_frames, desc=f"处理 {os.path.basename(input_path)}") as pbar:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # a. 目标检测
                det_results = yolo_model(frame, classes=[TARGET_CLASS_ID], verbose=False)[0]

                # b. 深度估计
                with torch.no_grad():
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    rgb_frame_resized = cv2.resize(rgb_frame, metric3d_input_size)
                    rgb_torch = torch.from_numpy(rgb_frame_resized).permute(2, 0, 1).unsqueeze(0).float().to(
                        DEVICE) / 255.0
                    pred_output = metric3d_model(data={'input': rgb_torch})
                    pred_depth_np = pred_output[0].squeeze().cpu().numpy()
                    pred_depth_filtered = cv2.resize(pred_depth_np, (width, height))

                # c. 准备带深度的检测结果
                detections_with_depth = []
                if det_results.boxes.shape[0] > 0:
                    for box in det_results.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        score = box.conf[0].item()
                        cls_id = box.cls[0].item()

                        roi_w, roi_h = int((x2 - x1) * 0.25), int((y2 - y1) * 0.25)
                        roi_x1, roi_y1 = x1 + ((x2 - x1) - roi_w) // 2, y1 + ((y2 - y1) - roi_h) // 2
                        depth_roi = pred_depth_filtered[roi_y1:roi_y1 + roi_h, roi_x1:roi_x1 + roi_w]
                        initial_depth = np.median(depth_roi) if depth_roi.size > 0 else 0.0
                        detections_with_depth.append([x1, y1, x2, y2, score, cls_id, initial_depth])

                # d. 更新跟踪器
                # The output format is [x1, y1, x2, y2, track_id, score, class_id, depth]
                tracks = tracker.update(np.array(detections_with_depth)) if len(
                    detections_with_depth) > 0 else np.empty((0, 8))

                # ========================================================================
                # MODIFIED: Write results in the requested KITTI tracking format
                # ========================================================================
                if tracks.shape[0] > 0:
                    for track in tracks:
                        bb_left, bb_top, bb_right, bb_bottom = track[0], track[1], track[2], track[3]
                        track_id = int(track[4])
                        score = track[5]

                        # Write the 17-column KITTI format string
                        f_out.write(
                            f"{frame_count} {track_id} {TARGET_CLASS_NAME} -1 -1 -10 "
                            f"{bb_left:.2f} {bb_top:.2f} {bb_right:.2f} {bb_bottom:.2f} "
                            f"-1 -1 -1 -1000 -1000 -1000 -10 {score:.4f}\n"
                        )

                # MODIFIED: Increment frame count at the end of the loop
                frame_count += 1
                pbar.update(1)

    cap.release()
    print(f"--- 处理完成！输出已保存至: {output_txt_path} ---")


# ==============================================================================
# 5. 批量处理主程序
# ==============================================================================
if __name__ == '__main__':
    print("\n>>> [DEBUG] 步骤 5: 开始执行批量处理主程序...")

    video_files = glob.glob(os.path.join(INPUT_VIDEOS_DIR, '*.mp4'))
    if not video_files:
        # Note: The error log showed kitti_videos, but doc specified input_videos. Double-check your path.
        print(f"!!! [WARNING] 在目录 {INPUT_VIDEOS_DIR} 中未找到任何 .mp4 视频文件。")
    else:
        print(f">>> [INFO] 找到 {len(video_files)} 个视频文件进行处理。")

    for video_path in sorted(video_files):
        try:
            video_name = os.path.basename(video_path)
            output_name = os.path.splitext(video_name)[0] + '.txt'
            output_path = os.path.join(OUTPUT_EVAL_DIR, output_name)

            process_video_for_eval(video_path, output_path)

        except Exception as e:
            print(f"!!! [FATAL ERROR] 处理视频 {video_path} 时发生严重错误: {e}")
            import traceback

            traceback.print_exc()
            continue

    print("\n>>> [DEBUG] 所有视频处理完毕。\n" + "=" * 60)