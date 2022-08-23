import math
import os
import random
from copy import deepcopy

import torch
import torch.nn.functional as F
from torchvision.utils import save_image

import mylib
import mylib.misc as misc
from mylib.torch_utils import unnormalize, update_average, warmup_learning_rate
from .criterion import (ReconstructionLoss, SwappedPredictionLoss,
                        compute_grad_gp, get_adversarial_loss, mse_loss)
from .networks import Generator, MultiPrototypes, StyleDiscriminator


class StyleAwareDiscriminator(mylib.BaseModel):

    @staticmethod
    def add_commandline_args(parser):
        # Model parameters.
        parser.add_argument(
            "--mod-type", choices={"adain", "stylegan2"}, default="stylegan2",
        )
        parser.add_argument("--latent-dim", type=int, default=256)
        parser.add_argument("--style-dim", type=int, default=512)
        parser.add_argument("--nb-proto", type=int, default=[32], nargs="+")
        parser.add_argument("--f-depth", type=int, default=2)
        parser.add_argument("--g-ch-mul", type=float, default=1.0)
        parser.add_argument("--d-ch-mul", type=float, default=1.0)
        parser.add_argument(
            "--gen-arch", choices={"skip", "resnet"}, default="skip",
        )
        parser.add_argument("--cnt-res", type=int, default=16)
        parser.add_argument("--big-disc", type=misc.str2bool, default=False)
        # Criterion.
        parser.add_argument(
            "--gan", choices={"nonsat", "hinge"}, default="nonsat",
        )
        parser.add_argument(
            "--recon", nargs="+",
            choices={"l1", "mse", "lpips", "L1", "MSE", "LPIPS"},
            default=["lpips"],
        )
        # Criterion parameters.
        parser.add_argument("--temperature", type=float, default=0.1)
        parser.add_argument("--lambda-swap", type=float, default=1.0)
        parser.add_argument("--lambda-r1", type=float, default=1.0)
        parser.add_argument("--lambda-rec", type=float, default=1.0)
        parser.add_argument("--lambda-sty", type=float, default=1.0)
        parser.add_argument("--use-queue-after", type=int, default=20000)
        # Optimizer parameters.
        parser.add_argument("--lr", type=float, default=1e-4)
        parser.add_argument("--beta1", type=float, default=0.0)
        parser.add_argument("--beta2", type=float, default=0.99)
        # Misc.
        parser.add_argument("--lazy-r1-freq", type=int, default=16)
        parser.add_argument("--cnt-preserv-freq", type=int, default=16)
        parser.add_argument("--warmup-step", type=int, default=3000)
        parser.add_argument("--latent-ratio", type=float, default=0.2)
        return parser

    def _create_networks(self):
        opt = self.opt
        image_size = opt.image_size

        self.G = Generator(
            image_size=image_size,
            latent_dim=opt.latent_dim,
            style_dim=opt.style_dim,
            f_depth=opt.f_depth,
            content_resolution=opt.cnt_res,
            architecture=opt.gen_arch,
            mod_type=opt.mod_type,
            channel_multiplier=opt.g_ch_mul,
        )
        self.D = StyleDiscriminator(
            image_size=image_size,
            latent_dim=opt.latent_dim,
            big_disc=opt.big_disc,
            channel_multiplier=opt.d_ch_mul,
        )

        queue_sizes = []
        for n_proto in opt.nb_proto:
            queue_size = 0
            if n_proto > opt.batch_size:
                queue_size = (n_proto // opt.batch_size) * opt.batch_size
            queue_sizes.append(queue_size)
        self.prototypes = MultiPrototypes(
            in_features=opt.latent_dim,
            num_prototypes=opt.nb_proto,
            num_queue=2,
            queue_sizes=queue_sizes,
        )

        # Non-trainable networks.
        if self.rank == 0:
            self.G_ema = deepcopy(self.G).requires_grad_(False)
            self.D_ema = deepcopy(self.D).requires_grad_(False)
            self.prototypes_ema = deepcopy(self.prototypes)
            self.prototypes_ema.requires_grad_(False)

    def _create_criterions(self):
        opt = self.opt
        self.gan_loss = get_adversarial_loss(opt.gan)
        self.swapped_prediction = SwappedPredictionLoss(opt.temperature)
        self.recon_loss = ReconstructionLoss(opt.recon)

    def _create_optimizers(self):
        opt = self.opt
        self.optimizers = {}

        c = 1.0
        if opt.cnt_preserv_freq > 0:
            c = opt.cnt_preserv_freq / (1. + opt.cnt_preserv_freq)
        self.optimizers["G"] = torch.optim.Adam(
            self.G.parameters(), lr=opt.lr * c,
            betas=(opt.beta1 ** c, opt.beta2 ** c)
        )
        self.optimizers["G"].zero_grad(set_to_none=True)

        c = 1.0
        if opt.lambda_r1 > 0.0 and opt.lazy_r1_freq > 0:
            c = opt.lazy_r1_freq / (1. + opt.lazy_r1_freq)
        self.optimizers["D"] = torch.optim.Adam(
            [
                {"params": self.D.parameters()},
                {
                    "params": self.prototypes.parameters(),
                    "lr": opt.lr * 0.01,
                },
            ], lr=opt.lr * c, betas=(opt.beta1 ** c, opt.beta2 ** c)
        )
        self.optimizers["D"].zero_grad(set_to_none=True)

    def forward(self, input, target, heatmap=None, return_codes=False):
        assert isinstance(target, torch.Tensor) and target.dim() in (2, 4)
        if target.dim() == 2:  # If target is the latent vector.
            sty = target
        else:  # If target is the image.
            with torch.no_grad():
                sty = self.D(target, command="encode")
        cnt = self.G(input, command="encode")
        x = self.G(cnt, sty, command="decode", heatmap=heatmap)
        return (x, cnt, sty) if return_codes else x

    def set_input(self, step, xs):
        self.step = step
        self.x_real, self.x_aug = xs
        self.batch_size = self.x_real.size(0)
        self.x_ref = self.x_real[torch.randperm(self.batch_size)]
        if hasattr(self, "fan"):
            self.hm = self.fan.get_heatmap(self.x_real)
        else:
            self.hm = None

        # Cancel gradients of last layer during first epoch.
        self.freeze_prototypes = step < 500
        self.use_queue = step >= self.opt.use_queue_after
        self.do_lazy_r1 = self.opt.lambda_r1 > 0.0 \
            and step % self.opt.lazy_r1_freq == 0
        self.do_content_preserving = self.opt.cnt_preserv_freq > 0 \
            and step % self.opt.cnt_preserv_freq == 0

        # adjust_learning_rate
        for optim in self.optimizers.values():
            warmup_learning_rate(
                optim, self.opt.lr, step, self.opt.warmup_step,
            )

    def training_step(self):
        if torch.rand(1) < self.opt.latent_ratio:
            target = self.prototypes(self.batch_size, command="sample")
        else:
            target = self.x_ref
        out = self(self.x_real, target, self.hm, return_codes=True)
        x_fake, cnt_real, sty_ref = out

        self.loss.clear()
        sty_org = self._update_discriminator(x_fake)
        self._update_generator(x_fake, cnt_real, sty_org, sty_ref)
        self._update_average()
        return self.loss

    def _update_discriminator(self, x_fake):
        self.D.requires_grad_(True)
        self.prototypes.requires_grad_(True)

        sty_org, logit_r = self.D(self.x_real)
        sty_aug = self.D(self.x_aug, command="encode")
        _, logit_f = self.D(x_fake.detach())

        # Compute the adversarial loss.
        loss_adv = self.gan_loss("D", logit_r, logit_f)
        self.loss["Loss/adv_D"] = loss_adv.detach()

        # Compute the swapped prediction loss.
        styles = sty_org, sty_aug
        queue_ids = (0, 1) if self.use_queue else (None, None)
        multi_scores = self.prototypes(styles, queue_ids)
        loss_swap = 0.0
        for scores in multi_scores:
            loss_swap += self.swapped_prediction(scores, [0,1], [1,0], self.batch_size)
        loss_swap /= len(scores)
        self.loss["Loss/swap"] = loss_swap.detach()

        # Update the discriminator and style encoder parameters.
        (loss_adv + self.opt.lambda_swap * loss_swap).backward()
        if self.freeze_prototypes:
            self.prototypes.zero_grad(set_to_none=True)
        self.optimizers["D"].step()
        self.optimizers["D"].zero_grad(set_to_none=True)
        self.prototypes(command="normalize")

        # Compute the R1 gradient penalty.
        if self.do_lazy_r1:
            self.x_real.requires_grad_(True)
            logit = self.D(self.x_real, command="discriminate")
            r1 = compute_grad_gp(logit, self.x_real, gamma=self.opt.lambda_r1)
            self.x_real.requires_grad_(False)

            lazy_r1 = self.opt.lazy_r1_freq * r1
            lazy_r1.backward()
            self.loss["Loss/R1"] = lazy_r1.detach()

            self.optimizers["D"].step()
            self.optimizers["D"].zero_grad(set_to_none=True)
        return styles[0].detach()

    def _update_generator(self, x_fake, cnt_real, sty_org, sty_ref):
        self.D.requires_grad_(False)
        self.prototypes.requires_grad_(False)

        # Compute the reconstruction loss.
        x_rec = self.G(cnt_real, sty_org, command="decode", heatmap=self.hm)
        loss_rec = self.recon_loss(x_rec, self.x_real)
        self.loss["Loss/recon"] = loss_rec.detach()

        # Compute the adversarial loss.
        sty_fake, logit = self.D(x_fake)
        logit_rec = self.D(x_rec, command="discriminate")
        loss_adv = self.gan_loss("G", logit_f=torch.cat((logit, logit_rec)))
        self.loss["Loss/adv_G"] = loss_adv.detach()
        
        # Compute the style preserving loss.
        loss_sty = mse_loss(sty_fake, sty_ref)
        self.loss["Loss/style"] = loss_sty.detach()

        # Update the generator parameters.
        loss = loss_adv \
            + self.opt.lambda_rec * loss_rec \
            + self.opt.lambda_sty * loss_sty
        loss.backward()
        self.optimizers["G"].step()
        self.optimizers["G"].zero_grad(set_to_none=True)

        if self.do_content_preserving:
            mask = self.generate_mask(size=self.opt.cnt_res)
            styles = [sty_org, sty_ref]
            if torch.rand(1) < 0.5:
                styles.reverse()

            cnt_real = self.G(self.x_real, command="encode")
            x_fake = self.G(
                cnt_real, styles, command="decode", mask=mask, heatmap=self.hm,
            )
            cnt_fake = self.G(x_fake, command="encode")
            cnt_real = F.normalize(cnt_real, dim=1)
            cnt_fake = F.normalize(cnt_fake, dim=1)
            sim = 2.0 - 2.0 * (cnt_real * cnt_fake).sum(1)
            loss_cnt = sim.mean((1,2)).mean()  
            loss_cnt.backward()
            self.loss["Loss/content"] = loss_cnt.detach()
            self.optimizers["G"].step()
            self.optimizers["G"].zero_grad(set_to_none=True)

    def _update_average(self):
        if self.rank == 0:
            update_average(self.G, self.G_ema)
            update_average(self.D, self.D_ema)
            update_average(self.prototypes, self.prototypes_ema)
            self.prototypes_ema.normalize()

    @torch.no_grad()
    def synthesize(self, source, target, **kwargs):
        hm = self.fan.get_heatmap(source) if hasattr(self, "fan") else None
        if isinstance(target, (list, tuple)):
            style_code = self.D_ema(torch.cat(target), command="encode")
            style_code = torch.chunk(style_code, chunks=2)
        elif target.dim() == 4:  # If target is the image.
            style_code = self.D_ema(target, command="encode")
        elif target.dim() == 2:  # If target is the latent vector.
            style_code = target
        else:
            raise NotImplementedError
        return self.G_ema(source, style_code, heatmap=hm, **kwargs)

    @torch.no_grad()
    def snapshot(self):
        opt = self.opt
        num_debug = self.debug.size(0)
        snapshot_dir = os.path.join(
            opt.run_dir, f"snapshot/{self.step//1000:03d}k"
        )
        os.makedirs(snapshot_dir, exist_ok=True)

        # Reference-guided synthesis.
        fname = os.path.join(snapshot_dir, "ref.png")
        self.generate_grid(self.debug, self.debug, fname)

        # Prototype-guided synthesis.
        filename = os.path.join(snapshot_dir, "proto{}_{:03d}to{:03d}.png")
        for i, n_proto in enumerate(opt.nb_proto):
            for j in range(math.ceil(n_proto/10)):
                begin = j * 10
                end = min(n_proto, (j+1)*10)
                fname = filename.format(i, begin, end-1)
                prototypes = self.prototypes_ema[i].weight[begin:end]
                self.generate_grid(self.debug, prototypes, fname)

        # Prototype interpolation.
        for i, pairs in enumerate(self.lerp_pairs):
            for j, pair in enumerate(pairs):
                fname = os.path.join(snapshot_dir, f"lerp{i}_{j}.png")
                style_codes = self.prototypes_ema.interpolate(pair, id=i)
                self.generate_grid(self.debug, style_codes, fname)

        # Spatial-style mixing.
        grid = [
            torch.ones(3, *self.debug[0].size(), device=self.device),
            self.debug
        ]
        for i in range(num_debug):
            src1 = self.debug[i].unsqueeze(0)
            src2 = self.debug[num_debug-1-i].unsqueeze(0)
            mask = self.generate_mask()
            output = self.synthesize(
                self.debug,
                (src1.expand_as(self.debug), src2.expand_as(self.debug)),
                mask=mask,
            )

            mask = F.interpolate(mask, size=src1.size(-1), mode="nearest")
            grid.append(mask.expand_as(src1))
            grid.append(src1)
            grid.append(src2)
            grid.append(output)
        grid = unnormalize(torch.cat(grid))
        fname = os.path.join(snapshot_dir, "mixing.png")
        save_image(grid, fname, nrow=num_debug+3, padding=0)

    def generate_grid(self, sources, references, fname=None):
        grid = []
        if references.dim() == 4:  # if image
            grid.append(torch.ones(1, *sources[0].size(), device=self.device))
            grid.append(references)

        for i in range(sources.size(0)):
            src = sources[i].unsqueeze(0)
            expanded = src.expand(references.size(0), *sources[0].size())
            output = self.synthesize(expanded, references)
            grid.append(src)
            grid.append(output)
        grid = unnormalize(torch.cat(grid))

        if fname is not None:
            save_image(grid, fname, nrow=references.size(0)+1, padding=0)
        return grid

    def generate_mask(self, batch_size=1, size=16, halfmask=False):
        mask = torch.zeros((batch_size, 1, size, size), device=self.device)
        for i in range(batch_size):
            if halfmask:
                if torch.rand(1) < 0.5:
                    w, h = size//2, size
                    x = 0 if torch.rand(1) < 0.5 else 8
                    y = 0
                else:
                    w, h = size, size//2
                    x = 0
                    y = 0 if torch.rand(1) < 0.5 else 8
            else:
                low = size // 4
                high = size - low
                w, h = torch.randint(low=low, high=high, size=(2,))
                x = torch.randint(low=0, high=size-w, size=(1,))
                y = torch.randint(low=0, high=size-h, size=(1,))
            ones = torch.ones((w, h), device=self.device)
            mask[i, :, x:x+w, y:y+h] = ones
        return mask

    def prepare_snapshot(self, dataset):
        opt = self.opt
        indices = random.sample([i for i in range(len(dataset))], k=4)
        debug = []
        for i in indices:
            x = dataset[i]
            debug.append(x)
        self.debug = torch.stack(debug).to(self.device)
        self.lerp_pairs = []
        for n_proto in opt.nb_proto:
            pairs = []
            for _ in range(2):
                pair = random.sample([i for i in range(n_proto)], k=2)
                pairs.append(pair)
            self.lerp_pairs.append(pairs)
