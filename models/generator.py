import torch as th

class GeneratorBlock(th.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, first: bool = False, last: bool = False) -> None:
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
                th.nn.Sigmoid()
            )
        else:
            self.main = th.nn.Sequential(
                th.nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
                th.nn.BatchNorm2d(out_channels),
                th.nn.ReLU(True)
            )

    def forward(self, x: th.Tensor) -> th.Tensor:
        return self.main(x)

class Generator(th.nn.Module):
    def __init__(self, noise_size: int, feature_map_depth: int) -> None:
        super().__init__()
        # first layer, no stride. Upsample from 1x1 to 4x4
        self.main = th.nn.Sequential(
            GeneratorBlock(noise_size, feature_map_depth * 8, first=True),
            GeneratorBlock(feature_map_depth * 8, feature_map_depth * 8),
            GeneratorBlock(feature_map_depth * 8, feature_map_depth * 4),
            GeneratorBlock(feature_map_depth * 4, feature_map_depth * 2),
            GeneratorBlock(feature_map_depth * 2, feature_map_depth * 1),
            GeneratorBlock(feature_map_depth * 1, 3, last=True),
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        x = self.main(x)
        return x