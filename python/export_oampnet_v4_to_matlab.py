"""\
导出 OAMPNetV4 训练参数到 MATLAB .mat 文件

目的
- 将 PyTorch 训练得到的 OAMPNetV4 权重中的可学习参数（gamma / damping / temperature）
  导出为 MATLAB 可直接 load 的 .mat，供 oampnet_detector.m 使用。

默认约定（与 train_oampnet_v4.py 对齐）
- 模型权重文件:  data/best_oampnet_v4{_version}_model.pth
- 导出参数文件:  data/oampnet_v4{_version}_params.mat

用法示例
  # 导出 tsv1 版本
  python export_oampnet_v4_to_matlab.py --version tsv1

  # 指定 data 目录（如果你的工程结构不同）
  python export_oampnet_v4_to_matlab.py --version tsv1 --data_dir ../../data

说明
- 该导出器与 train_oampnet_v4.py 的权重命名规则一致。
"""

import argparse
import os
import sys
from typing import Tuple, Optional

import numpy as np
import scipy.io as sio
import torch


def _infer_num_layers_from_state_dict(state_dict) -> int:
    """从 state_dict 推断 num_layers（gamma 向量长度）。"""
    if "gamma" not in state_dict:
        raise KeyError("state_dict 中找不到 'gamma'，无法推断 num_layers。")
    g = state_dict["gamma"]
    try:
        return int(g.numel())
    except Exception:
        return int(np.prod(np.array(g).shape))


def _resolve_data_dir(data_dir: Optional[str]) -> str:
    """默认 data_dir = ../../data（与 python/training/ 下脚本一致）。"""
    if data_dir:
        return os.path.abspath(data_dir)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    cand_project = os.path.abspath(os.path.join(script_dir, "..", "data"))
    if os.path.isdir(cand_project):
        return cand_project

    cand_repo = os.path.abspath(os.path.join(script_dir, "..", "..", "data"))
    return cand_repo


def _import_oampnetv4() -> Tuple[object, str]:
    """在不同目录结构下，尽量稳健地导入 models.oampnet_v4.OAMPNetV4。"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)  # python/
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)

    try:
        from models.oampnet_v4 import OAMPNetV4  # type: ignore
        return OAMPNetV4, "models.oampnet_v4"
    except Exception as e1:
        if script_dir not in sys.path:
            sys.path.insert(0, script_dir)
        try:
            from oampnet_v4 import OAMPNetV4  # type: ignore
            return OAMPNetV4, "oampnet_v4"
        except Exception as e2:
            raise ImportError(
                "无法导入 OAMPNetV4。请确认工程中存在 models/oampnet_v4.py，"
                "或将该文件路径加入 PYTHONPATH。\n"
                f"models 导入错误: {e1}\n平铺导入错误: {e2}"
            )


def main() -> None:
    ap = argparse.ArgumentParser(description="Export OAMPNetV4 params to MATLAB")
    ap.add_argument("--version", type=str, default="", help="数据/模型版本后缀，例如 tsv1（会拼成 _tsv1）")
    ap.add_argument("--data_dir", type=str, default=None, help="data 目录（默认按工程结构自动推断）")
    ap.add_argument("--N", type=int, default=256, help="符号维度 N（默认 256）")
    ap.add_argument("--model_name", type=str, default=None,
                    help="可选：显式指定模型权重文件名（否则使用 best_oampnet_v4{_version}_model.pth）")
    args = ap.parse_args()

    data_dir = _resolve_data_dir(args.data_dir)
    version_str = f"_{args.version}" if args.version else ""

    if args.model_name is None:
        model_name = f"best_oampnet_v4{version_str}_model.pth"
    else:
        model_name = args.model_name

    model_path = os.path.join(data_dir, model_name)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"找不到模型权重: {model_path}")

    state_dict = torch.load(model_path, map_location="cpu")
    num_layers = _infer_num_layers_from_state_dict(state_dict)

    OAMPNetV4, import_from = _import_oampnetv4()
    model = OAMPNetV4(N=args.N, num_layers=num_layers)
    model.load_state_dict(state_dict, strict=True)

    print("========================================")
    print("OAMPNetV4 参数导出")
    print("========================================")
    print(f"导入来源: {import_from}")
    print(f"模型权重: {model_path}")
    print(f"N={args.N}, num_layers={num_layers}")

    params = model.get_learned_params()

    matlab_params = {
        "gamma": params["gamma"].astype(np.float64),
        "damping": params["damping"].astype(np.float64),
        "temperature": params["temperature"].astype(np.float64),
        "num_layers": np.array([params["num_layers"]], dtype=np.float64),
        "N": np.array([params["N"]], dtype=np.float64),
        "version": np.array([args.version], dtype=object),
    }

    save_name = f"oampnet_v4{version_str}_params.mat"
    save_path = os.path.join(data_dir, save_name)
    sio.savemat(save_path, matlab_params)

    print(f"参数已保存: {save_path}")
    print("\n学习到的参数（前3个元素预览）:")
    print(f"  gamma[:3]        = {params['gamma'][:3]}")
    print(f"  damping[:3]      = {params['damping'][:3]}")
    print(f"  temperature[:3]  = {params['temperature'][:3]}")


if __name__ == "__main__":
    main()
