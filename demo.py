# %%
# ==============================================================================
# 0. 关键依赖库检查 (用于调试)
# ==============================================================================
import sys

sys.path.append("C:/Users/Administrator/Desktop/autodl-tmp (1)/Metric3D")
print(">>> [DEBUG] 步骤 0: 检查关键库版本...")
try:
    import mmcv
    import timm
    import ultralytics
    from filterpy.kalman import KalmanFilter
    from sklearn.mixture import GaussianMixture

    print(f">>> [INFO] mmcv version: {mmcv.__version__}")
    print(f">>> [INFO] timm version: {timm.__version__}")
    print(f">>> [INFO] ultralytics version: {ultralytics.__version__}")
    print(">>> [INFO] filterpy 和 scikit-learn (GMM) 库已成功导入。")
except ImportError as e:
    print(f"!!! [ERROR] 缺少核心库: {e}")
    raise
print(">>> [DEBUG] 步骤 0: 检查完成。\n" + "=" * 60 + "\n")

# ==============================================================================
# 1. 导入必要的库
# ==============================================================================
print(">>> [DEBUG] 步骤 1: 开始导入核心库...")
try:
    import cv2
    import torch
    import numpy as np
    from ultralytics import YOLO
    import sys
    import os
    from tqdm import tqdm
    from mmcv import Config
    from types import SimpleNamespace
    from filterpy.kalman import KalmanFilter as FilterPyKalmanFilter
    from sklearn.mixture import GaussianMixture
    # 导入自定义跟踪器
    from custom_byte_tracker import ByteTracker

    print(">>> [DEBUG] 核心库导入成功。")
except ImportError as e:
    print(f"!!! [ERROR] 导入核心库失败: {e}")
    raise

# --- 导入 Metric3D 相关的模块 ---
METRIC3D_PATH = r'C:\Users\Administrator\Desktop\autodl-tmp (1)\Metric3D'
if METRIC3D_PATH not in sys.path:
    sys.path.insert(0, METRIC3D_PATH)
try:
    from mono.model.monodepth_model import DepthModel as MonoDepthModel

    print(">>> [DEBUG] Metric3D 模块导入成功。")
except ImportError as e:
    print(f"!!! [ERROR] 从 Metric3D 导入模块失败: {e}")
    raise
print(">>> [DEBUG] 步骤 1: 所有库导入完成。\n" + "=" * 60 + "\n")

# ==============================================================================
# 2. 配置区域与路径检查
# ==============================================================================
print(">>> [DEBUG] 步骤 2: 配置模型和文件路径...")
YOLO_MODEL_PATH = 'weights/epoch30.pt'
METRIC3D_MODEL_PATH = 'weights/metric_depth_vit_large_800k.pth'
METRIC3D_CONFIG_PATH = 'Metric3D/mono/configs/HourglassDecoder/vit.raft5.large.py'
INPUT_VIDEO_PATH = 'VIDEOS/2.mp4'
OUTPUT_VIDEO_PATH = 'output1.mp4'

paths_to_check = {
    "YOLOv8 权重": YOLO_MODEL_PATH,
    "Metric3D 权重": METRIC3D_MODEL_PATH,
    "Metric3D 配置": METRIC3D_CONFIG_PATH,
    "输入视频": INPUT_VIDEO_PATH,
}
if not all(os.path.exists(p) for p in paths_to_check.values()):
    raise FileNotFoundError("一个或多个关键文件路径无效。")

print(">>> [DEBUG] 所有文件路径检查通过。")
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f">>> [DEBUG] 将要使用的设备: {DEVICE}")
print(">>> [DEBUG] 步骤 2: 配置完成。\n" + "=" * 60 + "\n")

# ==============================================================================
# 3. 模型加载
# ==============================================================================
print(">>> [DEBUG] 步骤 3: 开始加载深度学习模型...")
try:
    yolo_model = YOLO(YOLO_MODEL_PATH)
    TARGET_CLASS_NAME = 'Car'
    if hasattr(yolo_model, 'names') and isinstance(yolo_model.names, dict):
        TARGET_CLASS_ID = [k for k, v in yolo_model.names.items() if v == TARGET_CLASS_NAME][0]
        print(f">>> [INFO] 目标类别 '{TARGET_CLASS_NAME}' 已找到, ID为: {TARGET_CLASS_ID}")
    else:
        raise ValueError("YOLO 模型没有有效的 'names' 属性或格式不正确")
except Exception as e:
    print(f"!!! [ERROR] 加载 YOLOv8 模型失败: {e}")
    raise

try:
    cfg = Config.fromfile(METRIC3D_CONFIG_PATH)
    cfg.model.backbone.use_mask_token = False
    metric3d_model = MonoDepthModel(cfg).to(DEVICE)
    checkpoint = torch.load(METRIC3D_MODEL_PATH, map_location=DEVICE, weights_only=True)
    state_dict = checkpoint.get('model_state_dict', checkpoint.get('model', checkpoint))
    metric3d_model.load_state_dict(state_dict, strict=False)
    metric3d_model.eval()
    print(">>> [SUCCESS] Metric3Dv2 模型加载成功！")
except Exception as e:
    print(f"!!! [FATAL ERROR] 加载 Metric3Dv2 模型时出错: {e}")
    raise

print(">>> [DEBUG] 步骤 3: 所有模型加载完成。\n" + "=" * 60 + "\n")

# ==============================================================================
# 4. 视频处理主函数 (完整修复版)
# ==============================================================================
print(">>> [DEBUG] 步骤 4: 定义视频处理函数...")


def process_video_with_robust_depth_fusion(input_path, output_path):
    print("\n--- 开始视频处理  ---")
    cap = cv2.VideoCapture(input_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    metric3d_input_size = (cfg.data_basic['vit_size'][1], cfg.data_basic['vit_size'][0])
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    print(f">>> [INFO] 输入视频信息: {width}x{height} @ {fps:.2f} FPS, 共 {total_frames} 帧。")

    # 初始化跟踪器
    tracker_args = SimpleNamespace(
        track_high_thresh=0.5,
        track_low_thresh=0.1,
        new_track_thresh=0.6,
        track_buffer=30,
        match_thresh=0.8,
        mot20=False
    )
    tracker = ByteTracker(args=tracker_args, frame_rate=fps)

    # 深度滤波器字典
    robust_depth_filters = {}
    # 帧计数器
    frame_count = 0

    with tqdm(total=total_frames, desc="视频处理进度") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            annotated_frame = frame.copy()

            # 步骤 1: 目标检测
            det_results = yolo_model(frame, classes=[TARGET_CLASS_ID], verbose=False)[0]

            # 步骤 2: 全局深度图预测
            with torch.no_grad():
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb_frame_resized = cv2.resize(rgb_frame, metric3d_input_size)
                rgb_torch = torch.from_numpy(rgb_frame_resized).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE) / 255.0
                pred_output = metric3d_model(data={'input': rgb_torch})
                pred_depth_np = pred_output[0].squeeze().cpu().numpy()
                pred_depth_resized = cv2.resize(pred_depth_np, (width, height)).astype(np.float32)
                pred_depth_filtered = cv2.bilateralFilter(pred_depth_resized, d=5, sigmaColor=0.2, sigmaSpace=15)

            # 步骤 3: 跟踪前 - 为每个检测框计算鲁棒的初始深度
            detections_with_depth = []
            if det_results.boxes.shape[0] > 0:
                for box in det_results.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    score = box.conf[0].item()
                    cls_id = box.cls[0].item()

                    box_w, box_h = x2 - x1, y2 - y1
                    if box_w <= 0 or box_h <= 0: continue

                    # 提取中心区域用于计算初始深度
                    roi_w, roi_h = int(box_w * 0.25), int(box_h * 0.25)
                    roi_x1, roi_y1 = x1 + (box_w - roi_w) // 2, y1 + (box_h - roi_h) // 2
                    roi_x2, roi_y2 = roi_x1 + roi_w, roi_y1 + roi_h

                    depth_roi = pred_depth_filtered[roi_y1:roi_y2, roi_x1:roi_x2]
                    initial_depth = np.median(depth_roi) if depth_roi.size > 0 else 0.0

                    detections_with_depth.append([x1, y1, x2, y2, score, cls_id, initial_depth])

            # 步骤 4: 跟踪中 - 调用自定义跟踪器进行数据关联
            tracks = tracker.update(np.array(detections_with_depth)) if len(detections_with_depth) > 0 else np.empty(
                (0, 8))

            # 步骤 5: 跟踪后 - 深度计算+可视化
            active_track_ids = set()
            if tracks.shape[0] > 0:
                for track in tracks:
                    x1, y1, x2, y2 = map(int, track[:4])
                    track_id = int(track[4])
                    active_track_ids.add(track_id)
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    box_w, box_h = x2 - x1, y2 - y1
                    if box_w <= 0 or box_h <= 0: continue

                    # ---------------------- 深度计算部分 ----------------------
                    # 提取更宽的中心区域用于GMM聚类
                    roi_w, roi_h = int(box_w * 0.5), int(box_h * 0.5)
                    roi_x1, roi_y1 = max(x1 + (box_w - roi_w) // 2, 0), max(y1 + (box_h - roi_h) // 2, 0)
                    roi_x2, roi_y2 = min(roi_x1 + roi_w, width), min(roi_y1 + roi_h, height)
                    depth_roi = pred_depth_filtered[roi_y1:roi_y2, roi_x1:roi_x2]

                    observed_depth = 0.0
                    if depth_roi.size > 10:
                        try:
                            pixels = depth_roi.flatten().reshape(-1, 1)
                            n_components_range = range(1, 4)
                            lowest_bic = np.infty
                            best_gmm = None
                            for n_components in n_components_range:
                                gmm = GaussianMixture(n_components=n_components, random_state=0)
                                gmm.fit(pixels)
                                bic_score = gmm.bic(pixels)
                                if bic_score < lowest_bic:
                                    lowest_bic, best_gmm = bic_score, gmm

                            cluster_means = best_gmm.means_.flatten()

                            if track_id in robust_depth_filters:
                                kf = robust_depth_filters[track_id]
                                kf.predict()
                                predicted_depth = kf.x[0]
                                observed_depth = min(cluster_means, key=lambda x: abs(x - predicted_depth))
                            else:
                                observed_depth = min(cluster_means)
                        except Exception:
                            observed_depth = np.median(depth_roi) if depth_roi.size > 0 else 0
                    elif depth_roi.size > 0:
                        observed_depth = np.median(depth_roi)

                    if observed_depth <= 0: continue

                    # 卡尔曼滤波平滑深度
                    if track_id not in robust_depth_filters:
                        kf = FilterPyKalmanFilter(dim_x=2, dim_z=1)
                        kf.x = np.array([observed_depth, 0.])
                        kf.F = np.array([[1., 1.], [0., 1.]])
                        kf.H = np.array([[1., 0.]])
                        kf.P *= 100.
                        kf.R = 5
                        kf.Q = 0.1
                        robust_depth_filters[track_id] = kf
                    else:
                        kf = robust_depth_filters[track_id]
                        kf.update(observed_depth)

                    smoothed_depth = kf.x[0]

                    # ---------------------- 可视化部分 ----------------------
                    depth_text = f"ID:{track_id} D:{smoothed_depth:.2f}m"
                    (text_w, text_h), _ = cv2.getTextSize(depth_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(annotated_frame, (x1, y1 - 25), (x1 + text_w + 5, y1 - 5), (0, 100, 0), -1)
                    cv2.putText(annotated_frame, depth_text, (x1 + 2, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # 清理不活跃目标的缓存
            inactive_ids = set(robust_depth_filters.keys()) - active_track_ids
            for inactive_id in inactive_ids:
                del robust_depth_filters[inactive_id]

            out.write(annotated_frame)
            pbar.update(1)

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"\n--- 视频处理完成！输出保存在: {output_path} ---")


print(">>> [DEBUG] 步骤 4: 视频处理函数定义完成。\n" + "=" * 60 + "\n")

# ==============================================================================
# 5. 运行主程序
# ==============================================================================
print(">>> [DEBUG] 步骤 5: 开始执行主程序...")
try:
    process_video_with_robust_depth_fusion(INPUT_VIDEO_PATH, OUTPUT_VIDEO_PATH)
except Exception as e:
    print(f"!!! [FATAL ERROR] 在视频处理过程中发生严重错误: {e}")
    import traceback
    traceback.print_exc()
print(">>> [DEBUG] 步骤 5: 主程序执行完毕。\n" + "=" * 60)