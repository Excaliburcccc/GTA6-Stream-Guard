"""Export a silent copy with local masks and one output frame per input frame."""

from pathlib import Path
import tempfile

import cv2

from app import fps_of, open_source
from blood_masking import BloodMosaicDetector
from local_masking import ColorFilter, LocalMaskDetector, MaskTracker, bounded_boxes, mosaic
from police_vehicles import PoliceVehicleDetector
from shot_boundaries import ShotBoundaries


def export_local_video(args):
    if args.export.exists():
        raise ValueError("Output already exists. Choose a new MP4 filename.")
    region = LocalMaskDetector()
    police = PoliceVehicleDetector()
    if not police.available:
        raise RuntimeError(f"Police detector unavailable: {police.error}")
    region_track = MaskTracker(retain_unmatched=False)
    police_track = MaskTracker(retain_unmatched=False)
    blood = BloodMosaicDetector() if args.blood_mode != "off" else None
    blood_track = MaskTracker(retain_unmatched=False)
    advance_police = getattr(police, "advance", None)
    color = ColorFilter(args.blood_mode)
    source = open_source(args)
    writer = None
    temporary = None
    frames = 0
    roi_frames = 0
    vehicle_frames = 0
    blood_frames = 0
    color_frames = 0
    boundaries = ShotBoundaries()
    next_police_sample = 0.0
    next_region_sample = 0.0
    next_blood_sample = 0.0

    try:
        fps = fps_of(source)
        reported = source.get(cv2.CAP_PROP_FRAME_COUNT)
        while True:
            ok, original = source.read()
            if not ok:
                break
            height, width = original.shape[:2]
            if writer is None:
                size = ((width + 1) // 2 * 2, (height + 1) // 2 * 2)
                with tempfile.NamedTemporaryFile(
                    dir=args.export.parent, suffix=".mp4", delete=False
                ) as file:
                    temporary = Path(file.name)
                writer = cv2.VideoWriter(str(temporary), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
                if not writer.isOpened():
                    raise RuntimeError("Could not open the MP4 encoder.")
            elif (width + width % 2, height + height % 2) != size:
                raise RuntimeError("Input changes frame dimensions during export.")

            timestamp = frames / fps
            cut = boundaries.advance(original)
            if cut:
                police.reset()
                reset_region = getattr(region, "reset", None)
                if reset_region is not None:
                    reset_region()
                region_track.clear()
                police_track.clear()
                if blood is not None:
                    blood.reset()
                    blood_track.clear()
            police_due = cut or timestamp + 0.0005 >= next_police_sample
            region_due = cut or timestamp + 0.0005 >= next_region_sample
            blood_due = blood is not None and (cut or timestamp + 0.0005 >= next_blood_sample)
            if police_due or region_due or blood_due or advance_police is not None:
                scale = min(1.0, 1280 / width, 720 / height)
                sample = (cv2.resize(
                    original, (max(1, round(width * scale)), max(1, round(height * scale))),
                    interpolation=cv2.INTER_AREA
                ) if scale < 1 else original)
            if region_due:
                region_track.update(region.detect(sample), timestamp)
                next_region_sample = timestamp + 1 / 18
            if blood_due:
                blood_track.update(blood.detect(sample), timestamp)
                next_blood_sample = timestamp + 1 / 12
            if police_due:
                police_track.update(police.detect(sample), timestamp)
                next_police_sample = timestamp + 1 / 12
            elif advance_police is not None:
                police_track.update(advance_police(sample), timestamp)

            def scale_back(boxes):
                sw, sh = sample.shape[1], sample.shape[0]
                return [(round(x1 * width / sw), round(y1 * height / sh),
                         round(x2 * width / sw), round(y2 * height / sh))
                        for x1, y1, x2, y2 in boxes]

            regions = scale_back(region_track.active(timestamp))
            vehicles = scale_back(police_track.active(timestamp))
            blood_regions = scale_back(blood_track.active(timestamp))
            regions, blood_regions, vehicles = bounded_boxes(original.shape, regions, blood_regions, vehicles)
            output, changed = color.apply(original)
            mosaic(output, vehicles, blocks=6)
            mosaic(output, blood_regions, solid=args.cover_style == "solid", blocks=4)
            mosaic(output, regions, solid=args.cover_style == "solid")
            if width % 2 or height % 2:
                output = cv2.copyMakeBorder(output, 0, height % 2, 0, width % 2, cv2.BORDER_CONSTANT)
            writer.write(output)
            frames += 1
            roi_frames += bool(regions)
            vehicle_frames += bool(vehicles)
            blood_frames += bool(blood_regions)
            color_frames += changed

        # OpenCV estimates WebM frame counts from duration and can round up once.
        if not frames or (reported > 0 and frames + 1 < reported):
            raise RuntimeError(f"Incomplete input. Decoded {frames} frames; container reports {reported:g}.")
        writer.release()
        writer = None
        if not temporary.stat().st_size:
            raise RuntimeError("The MP4 encoder did not write any data.")
        temporary.rename(args.export)
        print(
            f"Exported {frames} silent frames at {fps:.6g} fps to {args.export}. "
            f"Police mosaics: {vehicle_frames}; exposure mosaics: {roi_frames}; "
            f"blood mosaics: {blood_frames}; color filter: {color_frames}; "
            "automatic full-screen covers: 0.", flush=True,
        )
    finally:
        source.release()
        if writer is not None:
            writer.release()
        if temporary is not None:
            temporary.unlink(missing_ok=True)
