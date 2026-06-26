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
        "chắc nhiều bác chủ phòng sẽ cần, em có bán nha, thiết kế theo yêu cầu, freeship",
        "Ông ngoại em đang cần bán mảnh Đất  giá chỉ có 2 tỷ  Ai muốn mua qua nhà em chỗ Vạn Phúc Hà Đông em dẫn đi xem đất ạ. Đất của ông bà em ông em giờ muốn về quê ở để an dưỡng tuổi già cũng là để gần tổ tiên. Nên ông có nhờ e bán ạ. Liên hệ em qua sdt: 00000000000 Để đi xem trực tiếp",
        "🌺 Chủ cần bán nhà Phú Thượng Dt/65m2 6tầng thang máy Ô tô đỗ cổng,vào nhà Giá bán nhỉnh 15tỷ. 🔥 Khu vực thoáng mát, đẹp, gần Sông Hồng, hưởng trọn tiện ích KDT Ciputra, vài bước là ra đường đôi Sunshine. +Sổ đỏ chính chủ vuông đẹp cất két. ☎️ Zalo E Đặng Phương 000.000.0000 để được xem nhà.",
        "cần bán 4 chỉ pnj mua 18.7, nay bán 14.6 ngay bây giờ kv q5 hcm",
        "Em Cần Bán 100k ROS , 100k FLC , 100k BAV zá tốt. Có bán số lượng nhỏ 5k , 10k.. Anh chị cần mua xin Alo Hảo : 00000000000"
    ]
    for t in samples:
        label, score = model.predict(t)
        print(f"{label[0].replace('__label__','')} {score[0]:.2f} | {t}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(MODEL_PATH))


if __name__ == "__main__":
    main()
