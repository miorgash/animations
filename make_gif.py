import pathlib as pl
from PIL import Image

def make_gif(file_path):

    image_path = pl.Path('fig')
    append_images = [Image.open(f) for f in image_path.glob('*.png')]
    im = append_images[0]
    im.save(file_path,
            save_all=True,
            duration=600,
            append_images=append_images[1:])


def delete_png(dir_path):
    p = pl.Path(dir_path)
    print(p)
    for f in p.glob('*.png'):
        print(f)
        f.unlink()