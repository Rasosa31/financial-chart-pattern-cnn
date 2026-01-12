import os
import shutil
import pandas as pd


def temporal_split(
    images_dir="data/train/images",
    output_dir="data/walk_forward",
    splits=None
):
    """
    Crea splits temporales train/test para walk-forward validation
    """

    if splits is None:
        splits = [
            ("2015-01-01", "2017-12-31", "2018-01-01", "2018-12-31"),
            ("2015-01-01", "2018-12-31", "2019-01-01", "2019-12-31"),
            ("2015-01-01", "2019-12-31", "2020-01-01", "2020-12-31"),
            ("2015-01-01", "2020-12-31", "2021-01-01", "2021-12-31"),
            ("2015-01-01", "2021-12-31", "2022-01-01", "2022-12-31"),
            ("2015-01-01", "2022-12-31", "2023-01-01", "2023-12-31"),
        ]

    for i, (train_start, train_end, test_start, test_end) in enumerate(splits):
        split_dir = os.path.join(output_dir, f"split_{i+1}")

        for subset in ["train", "test"]:
            for cls in ["up", "down"]:
                os.makedirs(
                    os.path.join(split_dir, subset, "images", cls),
                    exist_ok=True
                )

        for cls in ["up", "down"]:
            cls_dir = os.path.join(images_dir, cls)

            if not os.path.exists(cls_dir):
                print(f"⚠️ Carpeta no encontrada: {cls_dir}")
                continue

            for fname in os.listdir(cls_dir):
                try:
                    # chart_YYYY-MM-DD_xxx.png
                    date_str = fname.split("_")[1]
                    date = pd.to_datetime(date_str)
                except Exception:
                    continue

                src = os.path.join(cls_dir, fname)

                if train_start <= date.strftime("%Y-%m-%d") <= train_end:
                    dst = os.path.join(split_dir, "train", "images", cls, fname)
                elif test_start <= date.strftime("%Y-%m-%d") <= test_end:
                    dst = os.path.join(split_dir, "test", "images", cls, fname)
                else:
                    continue

                shutil.copy(src, dst)

        print(f"✅ Split {i+1} creado")


if __name__ == "__main__":
    temporal_split()
