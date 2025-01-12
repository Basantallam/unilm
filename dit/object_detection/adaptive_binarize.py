import argparse
import os
import cv2
import tqdm


def convert(fn):
    # given a file name, convert it into binary and store at the same position
    img = cv2.imread(fn)
    gim = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gim = cv2.adaptiveThreshold(gim, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 45, 11)
    g3im = cv2.cvtColor(gim, cv2.COLOR_GRAY2BGR)
    cv2.imwrite(fn, g3im)



if __name__ == '__main__':
    """
    Now only feasible for trackA_XX
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', default="/content/apex/PATH-to-ICDAR/at_trackA_archival/train")
    args = parser.parse_args()
    print("print ",os.listdir(args.root_dir),"\n")
    for fdname in os.listdir(args.root_dir):
        if fdname.endswith(".json") or fdname.endswith("trackA_archival"):
            continue
        ffdname = os.path.join(args.root_dir, fdname)
        print("print2 ",os.listdir(ffdname),"\n")
        print("curr = ", fdname)
        for file in tqdm.tqdm(os.listdir(ffdname)):
            if file.endswith(".xml"):
                continue
            ffile = os.path.join(ffdname, file)
            print("print3", ffile+"\n")
            convert(ffile)
