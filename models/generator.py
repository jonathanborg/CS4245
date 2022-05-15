import torch as th

class GeneratorBlock(th.nn.Module):
    def __init__(self, in_channels, out_channels, first=False, last=False) -> None:
        assert(not (first and last)) # block can't be both first and last
        super().__init__()
        if first:
            self.main = th.nn.Sequential(
                th.nn.ConvTranspose2d(in_channels, out_channels, 3, 1, 0, bias=False),
                th.nn.BatchNorm2d(out_channels),
                th.nn.ReLU(True)
            )
        elif last:
            self.main = th.nn.Sequential(
                th.nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
                th.nn.Tanh()
            )
        else:
            self.main = th.nn.Sequential(
                th.nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
                th.nn.BatchNorm2d(out_channels),
                th.nn.ReLU(True)
            )

    def forward(self, x):
        return self.main(x)

class Generator(th.nn.Module):
    def __init__(self, noise_size, colour_channels, feature_map_depth) -> None:
        super().__init__()
        # first layer, no stride. Upsample from 1x1 to 4x4
        self.main = th.nn.Sequential(
            GeneratorBlock(noise_size, feature_map_depth * 8, first=True),
            GeneratorBlock(feature_map_depth * 8, feature_map_depth * 8),
            GeneratorBlock(feature_map_depth * 8, feature_map_depth * 4),
            GeneratorBlock(feature_map_depth * 4, feature_map_depth * 2),
            GeneratorBlock(feature_map_depth * 2, feature_map_depth * 1),
            GeneratorBlock(feature_map_depth * 1, colour_channels, last=True),
        )

    def forward(self, x):
        x = self.main(x)
        return x