import argparse
from datetime import datetime

import torch
import torch.nn.functional as F

import os
from PIL import Image
from torchvision import utils

from simple_diffusion.scheduler import DDIMScheduler
from simple_diffusion.model import UNet

n_timesteps = 1000
n_inference_timesteps = 250
seedvalue =  500 # This value decides how many samples are gomna be created, for example for seed value 500 it produces 500 (each seed) * 32 (batch of different) images 

def main(args,seedpass):
    print("Generating 32 samples for seed: ",seedpass)
    model = UNet(3, image_size=(args.resolution,2*args.resolution), hidden_dims=[64, 128, 256, 512],
                 use_flash_attn=args.use_flash_attn)
    noise_scheduler = DDIMScheduler(num_train_timesteps=n_timesteps,
                                    beta_schedule="cosine")

    pretrained = torch.load(args.pretrained_model_path,map_location="cuda:7")["ema_model_state"]
    model.load_state_dict(pretrained, strict=False)

    device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    model.eval()
    with torch.no_grad():
        # has to be instantiated every time, because of reproducibility
        generator = torch.manual_seed(seedpass)
        generated_images = noise_scheduler.generate(
            model,
            num_inference_steps=n_inference_timesteps,
            generator=generator,
            eta=1.0,
            batch_size=args.eval_batch_size)

        images = generated_images["sample"]
        images_processed = (images * 255).round().astype("uint8")

        current_date = datetime.today().strftime('%Y%m%d_%H%M%S')
        out_dir = f"./{args.samples_dir}/{current_date}_seed{seedpass}/"
        os.makedirs(out_dir)
        for idx, image in enumerate(images_processed):
            image = Image.fromarray(image)
            image.save(f"{out_dir}/{idx}.jpeg")

        utils.save_image(generated_images["sample_pt"],
                         f"{out_dir}/grid.jpeg",
                         nrow=args.eval_batch_size // 4)

for each_seed in range(seedvalue):
    if __name__ == "__main__":
        parser = argparse.ArgumentParser(
            description="Simple script for image generation.")
        parser.add_argument("--samples_dir", type=str, default="generated_samples/")
        parser.add_argument("--resolution", type=int, default=128)
        parser.add_argument("--pretrained_model_path",
                            type=str,
                            default="trained_models/ddpm-model-ep574.pth",
                            help="Path to pretrained model")
        parser.add_argument("--eval_batch_size", type=int, default=32)
        parser.add_argument('--use_flash_attn', action='store_true')

        args = parser.parse_args()

        main(args,each_seed)
