from PIL import Image
import glob

frame_folder = 'tmp/'


frames = [Image.open(image) for image in glob.glob(f"{frame_folder}/*reg_0.005_*.png")]
frame_one = frames[0]
frame_one.save("MNIST_reg0005.gif", format="GIF", append_images=frames,
        save_all=True, duration=2*len(frames), loop=1)