import torch as th

class GeneratorBlock(th.nn.Module):
    def __init__(self, channels, first=False, last=False, noise_size=0, colour_channels=0) -> None:
        assert(not (first and last)) # block can't be both first and last
        assert((not first) or noise_size > 0) # first -> noise_size > 0 (noise_size needs to be set for the first layer)
        assert((not last) or colour_channels > 0) # last -> colour_channels > 0 (colour_channels needs to be set for the last layer)
        super().__init__()
        if first:
            self.main = th.nn.Sequential(
                th.nn.ConvTranspose2d(noise_size, channels, 3, 1, 0, bias=False),
                th.nn.BatchNorm2d(channels),
                th.nn.ReLU(True)
            )
        elif last:
            self.main = th.nn.Sequential(
                th.nn.ConvTranspose2d(channels, colour_channels, 4, 2, 1, bias=False),
                th.nn.Tanh()
            )
        else:
            self.main = th.nn.Sequential(
                th.nn.ConvTranspose2d(channels, channels // 2, 4, 2, 1, bias=False),
                th.nn.BatchNorm2d(channels // 2),
                th.nn.ReLU(True)
            )

    def forward(self, x):
        return self.main(x)

class Generator(th.nn.Module):
    def __init__(self, noise_size, colour_channels, feature_map_depth) -> None:
        super().__init__()
        # first layer, no stride. Upsample from 1x1 to 4x4
        self.main = th.nn.Sequential(
            GeneratorBlock(feature_map_depth * 16, first=True, noise_size=noise_size),
            GeneratorBlock(feature_map_depth * 16),
            GeneratorBlock(feature_map_depth * 8),
            GeneratorBlock(feature_map_depth * 4),
            GeneratorBlock(feature_map_depth * 2),
            GeneratorBlock(feature_map_depth * 1, last=True, colour_channels=colour_channels),
        )

    def forward(self, x):
        x = self.main(x)
        return x