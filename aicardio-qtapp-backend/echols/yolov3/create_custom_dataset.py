import os, glob, shutil
import cv2
import argparse

def get_bbox(msk_path):
    mask = cv2.imread(msk_path, cv2.IMREAD_GRAYSCALE)
    ret, thresh = cv2.threshold(mask, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    assert len(contours) == 1

    contour = contours[0]
    xmin, xmax = contour[...,0].min(), contour[...,0].max()
    ymin, ymax = contour[...,1].min(), contour[...,1].max()
    h, w = mask.shape[:2]
    xmin /= w
    xmax /= w
    ymin /= h
    ymax /= h
    xc = (xmin+xmax) / 2
    yc = (ymin+ymax) / 2
    width = xmax-xmin
    height = ymax-ymin

    return xc, yc, width, height


def generate_dataset(src_aicardio_dataset_dir, target_custom_dir, split):

    img_dir = os.path.join(src_aicardio_dataset_dir, 'images')
    msk_dir = os.path.join(src_aicardio_dataset_dir, 'masks')

    def create_sample(img_path, msk_path):
        bname = os.path.basename(img_path)
        os.makedirs(os.path.join(target_custom_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(target_custom_dir, "labels"), exist_ok=True)
        tgt_img_path = os.path.join(target_custom_dir, "images", bname)
        tgt_lbl_path = os.path.join(target_custom_dir, "labels", f"{bname[:-4]}.txt")
        if not os.path.exists(tgt_img_path): 
            shutil.copy(img_path, tgt_img_path)

        xc, yc, width, height = get_bbox(msk_path)
        with open(tgt_lbl_path, 'w') as f:
            print(f"0 {xc:.4f} {yc:.4f} {width:.4f} {height:.4f}", file=f)

        return tgt_img_path

    with open(os.path.join(target_custom_dir, f"{split}.txt"), 'w') as split_file:
        for msk_path in glob.glob(os.path.join(msk_dir, '*.png')):
            bname = os.path.basename(msk_path)[:-4] # remove .png
            img_path = os.path.join(img_dir, f"{bname}.jpg")
            if not os.path.isfile(img_path): continue
            
            tgt_img_path = create_sample(img_path, msk_path)
            print(tgt_img_path, file=split_file)
            print(tgt_img_path)


def draw_bbox(tgt_img_path):
    img = cv2.imread(tgt_img_path)
    h, w = img.shape[:2]
    tgt_lbl_path = os.path.join(os.path.dirname(tgt_img_path), "..", "labels", f"{os.path.basename(tgt_img_path)[:-4]}.txt")
    with open(tgt_lbl_path) as f:
        line = f.readline()
        tokens = line.strip().split(' ')
        xc, yc, width, height = [float(tk) for tk in tokens[1:]]
        xmin, xmax = int((xc - width / 2)*w), int((xc + width / 2)*w)
        ymin, ymax = int((yc - height / 2)*h), int((yc + height / 2)*h)
        print(xmin, ymin, xmax, ymax)
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255, 255, 0), thickness=2)
        cv2.imwrite(os.path.join(os.path.dirname(tgt_img_path), "..", "tmp", os.path.basename(tgt_img_path)), img)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_dir", type=str, required=True, help="/data.local/tuannm/data/2020-02-04_4C/train_dev")
    parser.add_argument("--tgt_dir", type=str, required=True, help="data/custom")
    parser.add_argument("--split", type=str, required=True, help="train | valid")
    opt = parser.parse_args()
    print(opt)

    src_dir = opt.src_dir # "/data.local/tuannm/data/2020-02-04_4C/train_dev"
    tgt_dir = opt.tgt_dir # "data/custom"
    split = opt.split # "train"

    generate_dataset(src_dir, tgt_dir, split)

    # draw_bbox("data/custom/images/From40Frs__MIET 50T restrictive heart EF R__4C__IMG-0099-00001.dcm_24.jpg")
