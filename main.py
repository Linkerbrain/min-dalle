import argparse
import os
from PIL import Image
from pathlib import Path

from dalle_e_mini import generate_image

from min_dalle.dalle_mini import DalleMini
from min_dalle.cool_traceback import cool_traceback

parser = argparse.ArgumentParser()

parser.add_argument('--mega', action='store_true')
parser.add_argument('--no-mega', dest='mega', action='store_false')
parser.set_defaults(mega=False)

parser.add_argument('--seed', type=int, default=7)

"""
Main helpers
"""

def ascii_from_image(image: Image.Image, size: int) -> str:
    rgb_pixels = image.resize((size, int(0.55 * size))).convert('L').getdata()
    chars = list('.,;/IOX')
    chars = [chars[i * len(chars) // 256] for i in rgb_pixels]
    chars = [chars[i * size: (i + 1) * size] for i in range(size // 2)]
    return '\n'.join(''.join(row) for row in chars)

def save_image(image, prompt, seed):
    dest = os.path.join('./results/', f'{prompt}_{seed}.png')
    print("saving image to", dest)
    image.save(dest)

    return image

"""
Main
"""

def main(args):
    print("Using args:", args)

    # initiate model
    dalle = DalleMini(is_mega=args.mega)

    while True:
        print("\n\n")
        prompt = input("Choose Your Prompt: ")

        # inference
        image = dalle.make_promptart(prompt = prompt, seed=args.seed)

        if image != None:
            save_image(image, prompt=prompt, seed=args.seed)
            print(ascii_from_image(image, size=128))


if __name__ == '__main__':
    args = parser.parse_args()
    cool_traceback(main, args)