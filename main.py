import random
import sys
from pathlib import Path

import fasttext

DATA_PATH = Path("datasets/ads.txt")
MODEL_PATH = Path("output_models/is_ad.bin")
TRAIN_PATH = Path("train.txt")
TEST_PATH = Path("test.txt")


def main():
    # Windows consoles default to cp1252, which can't encode Vietnamese.
    sys.stdout.reconfigure(encoding="utf-8")

    with open(DATA_PATH, encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    random.seed(42)
    random.shuffle(lines)

    ads     = [line for line in lines if line.startswith("__label__ad ")]
    not_ads = [line for line in lines if line.startswith("__label__not_ad")]

    split_a = int(len(ads) * 0.8)
    split_n = int(len(not_ads) * 0.8)

    train = ads[:split_a] + not_ads[:split_n]
    test  = ads[split_a:] + not_ads[split_n:]

    random.shuffle(train)
    random.shuffle(test)

    with open(TRAIN_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(train))
    with open(TEST_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(test))

    print(f"Train: {len(train)} | Test: {len(test)}")

    model = fasttext.train_supervised(
        str(TRAIN_PATH),
        epoch=25,
        lr=0.5,
        wordNgrams=2,
        minCount=1,
    )

    n, precision, recall = model.test(str(TEST_PATH))
    f1 = 2 * (precision * recall) / (precision + recall)
    print(f"Precision: {precision:.2f} | Recall: {recall:.2f} | F1: {f1:.2f}")

    # Live test
    samples = [
        "shop mình vừa về hàng mới inbox nha 150k",
        "hôm nay mưa to quá chán ghê",
        "tuyển nhân viên lương 8tr liên hệ zalo",
        "giá vàng hôm nay tăng mạnh",
    ]
    for t in samples:
        label, score = model.predict(t)
        print(f"{label[0].replace('__label__','')} {score[0]:.2f} | {t}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(MODEL_PATH))


if __name__ == "__main__":
    main()
