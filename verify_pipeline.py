"""Quick verification script to test all pipeline modules."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

def main():
    print("=" * 60)
    print("  FULL PIPELINE VERIFICATION")
    print("=" * 60)

    # 1. Models
    from src.models import build_model, SwinUNETRModel, UNet3DModel, MODEL_REGISTRY
    print(f"\n[1] Model Registry: {list(MODEL_REGISTRY.keys())}")

    s = SwinUNETRModel(feature_size=48)
    u = UNet3DModel()
    print(f"    SwinUNETR: {s.num_parameters():,} params")
    print(f"    UNet3D:   {u.num_parameters():,} params")

    # 2. Optimizer + Scheduler
    from src.models.swin_unetr import build_swin_optimizer_scheduler
    opt, sched = build_swin_optimizer_scheduler(s, lr=1e-4, max_epochs=100)
    print(f"\n[2] Optimizer: {type(opt).__name__}, Scheduler: {type(sched).__name__}")

    # Differential LR
    groups = s.get_parameter_groups()
    print(f"    Differential LR groups: {len(groups)}")

    # 3. Training modules
    from src.training import SegmentationTrainer, get_loss_function, run_model_comparison
    loss = get_loss_function("dice_ce")
    print(f"\n[3] Loss: {type(loss).__name__}")
    print(f"    Trainer: {SegmentationTrainer.__name__}")
    print(f"    Comparison: {run_model_comparison.__name__}")

    # 4. Data modules
    from src.data import DataSplitter, create_dataloaders
    print(f"\n[4] DataSplitter: OK")
    print(f"    create_dataloaders: OK")

    # 5. Inference
    from src.inference import InferenceEngine
    print(f"\n[5] InferenceEngine: OK")

    # 6. Config
    from src.config import get_config
    cfg = get_config()
    print(f"\n[6] Config: {cfg.project_name}")
    print(f"    Version: {cfg.get('project.version', 'N/A')}")

    # 7. Model info
    swin_info = s.get_model_info()
    unet_info = u.get_model_info()
    print(f"\n[7] SwinUNETR info:")
    print(f"    Architecture: {swin_info['architecture']}")
    print(f"    Encoder params: {swin_info['encoder_params']:,}")
    print(f"    Decoder params: {swin_info['decoder_params']:,}")
    print(f"    Deep supervision: {swin_info['deep_supervision']}")

    print(f"\n    UNet3D info:")
    print(f"    Architecture: {unet_info['architecture']}")
    print(f"    Channels: {unet_info['channels']}")

    # 8. build_model factory
    m1 = build_model("unet3d")
    m2 = build_model("swin_unetr", feature_size=48)
    print(f"\n[8] build_model('unet3d'): {type(m1).__name__} ({m1.num_parameters():,} params)")
    print(f"    build_model('swin_unetr'): {type(m2).__name__} ({m2.num_parameters():,} params)")

    print("\n" + "=" * 60)
    print("  ALL MODULES VERIFIED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    main()
