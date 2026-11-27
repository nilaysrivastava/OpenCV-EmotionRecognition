"""CLI entry point using the same model and preprocessing as the web studio."""

import argparse
import json
from pathlib import Path

from app.inference import InputError, ModelService


def main():
    parser = argparse.ArgumentParser(
        description="Recognize facial expressions with this project’s trained CNN."
    )
    parser.add_argument(
        "--image", type=Path, help="Analyze one portrait and print JSON."
    )
    parser.add_argument(
        "--model-dir", type=Path, default=Path(__file__).parent / "models"
    )
    parser.add_argument("--output", type=Path, help="Save the JSON observation.")
    parser.add_argument(
        "--camera",
        type=int,
        help="Open a local webcam (requires desktop opencv-python).",
    )
    args = parser.parse_args()
    service = ModelService(args.model_dir)
    if not service.status()["ready"]:
        parser.exit(
            2,
            "Your trained model is not ready. Install its weights and model_config.json in models/. No replacement model is used.\n",
        )
    if args.image:
        if not args.image.is_file():
            parser.error("Image file not found.")
        try:
            result = service.analyze(args.image.read_bytes())
        except (InputError, OSError) as exc:
            parser.exit(2, str(exc) + "\n")
        text = json.dumps(result, indent=2)
        if args.output:
            args.output.write_text(text + "\n")
        print(text)
        return
    if args.camera is None:
        parser.error(
            "Choose --image IMAGE or --camera 0. For the browser interface, run python run.py."
        )
    import cv2

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        parser.exit(
            2,
            "Camera could not be opened. Check permission or use the browser interface.\n",
        )
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            encoded = cv2.imencode(".jpg", frame)[1].tobytes()
            result = service.analyze(encoded)
            sx = frame.shape[1] / result["width"]
            sy = frame.shape[0] / result["height"]
            for face in result["faces"]:
                b = face["box"]
                x, y, w, h = (
                    int(b["x"] * sx),
                    int(b["y"] * sy),
                    int(b["width"] * sx),
                    int(b["height"] * sy),
                )
                cv2.rectangle(frame, (x, y), (x + w, y + h), (155, 210, 180), 2)
                cv2.putText(
                    frame,
                    face["label"],
                    (x, max(20, y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (155, 210, 180),
                    2,
                )
            cv2.imshow("Expression Atlas", frame)
            if cv2.waitKey(1) & 255 == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
