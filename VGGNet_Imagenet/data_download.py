import os
import wget


def bar_custom(current, total, width=80):
    progress = "Downloading: %d%% [%d / %d] bytes" % (current / total * 100, current, total)
    return progress


def download_imagenet(root='D:'):
    """
    download_imagenet train, validation set
    :param img_dir: root for download imagenet
    :return:
    """

    # make url
    train_url = 'https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar'
    val_url = 'http://www.image-net.org/challenges/LSVRC/ILSVRC2012_img_val.tar'
    devkit_url = 'http://www.image-net.org/challenges/LSVRC/2012/ILSVRC2012_devkit_t12.tar.gz'

    
    root = root + "/data/"
    os.makedirs(root, exist_ok=True)
    os.chdir(root)
    print("Download at", root, "...")
    wget.download(url=train_url, out=root, bar=bar_custom)
    # print('')
    # wget.download(url=val_url, out=root, bar=bar_custom)
    # print('')
    # wget.download(url=devkit_url, out=root, bar=bar_custom)
    print('')
    print('done!')


if __name__ == '__main__':
    download_imagenet(root='D:')