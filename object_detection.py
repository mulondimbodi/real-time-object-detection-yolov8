import argparse
from pathlib import Path

from ultralytics import YOLO

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_source(source):
    return int(source) if source.isdigit() else source


def infer_mode(source):
    if source.isdigit():
        return "track"

    suffix = Path(source).suffix.lower()
    return "detect" if suffix in IMAGE_EXTENSIONS else "track"


def run_detection(model_path, source, mode, confidence, show):
    model = YOLO(model_path)
    input_source = parse_source(source)
    selected_mode = mode if mode != "auto" else infer_mode(source)

    if selected_mode == "detect":
        model.predict(source=input_source, conf=confidence, show=show, save=True)
    else:
        model.track(
            source=input_source,
            tracker="bytetrack.yaml",
            conf=confidence,
            show=show,
            save=True,
        )

    print("Processing complete. Check the runs/ folder for saved results.")


def main():
    parser = argparse.ArgumentParser(
        description="Run YOLOv8 object detection or tracking on images, videos, or webcam streams."
    )
    parser.add_argument(
        "--source",
        default="0",
        help="Input source path or webcam index. Examples: 0, assets/input/asian-market.jpg",
    )
    parser.add_argument(
        "--mode",
        choices=["auto", "detect", "track"],
        default="auto",
        help="Processing mode. Auto detects images and tracks video/webcam sources.",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.35,
        help="Minimum confidence threshold for predictions.",
    )
    parser.add_argument(
        "--model",
        default="yolov8n.pt",
        help="YOLOv8 model weights to use.",
    )
    parser.add_argument(
        "--hide",
        action="store_true",
        help="Disable the live preview window while saving results.",
    )

    args = parser.parse_args()

    run_detection(
        model_path=args.model,
        source=args.source,
        mode=args.mode,
        confidence=args.confidence,
        show=not args.hide,
    )


if __name__ == "__main__":
    main()