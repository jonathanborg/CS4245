import torch as th

class DiscriminatorBlock(th.nn.Module):
    def __init__(self, in_channels, out_channels, first=False, last=False) -> None:
        assert(not (first and last)) # block can't be both first and last
        super().__init__()
        if first:
            self.main = th.nn.Sequential(
                th.nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False),
                th.nn.LeakyReLU(0.2, inplace=True),
            )
        elif last:
            self.main = th.nn.Sequential(
                th.nn.Conv2d(in_channels, out_channels, 3, 1, 0, bias=False),
                th.nn.Sigmoid()
            )
        else:
            self.main = th.nn.Sequential(
                th.nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False),
                th.nn.BatchNorm2d(out_channels),
                th.nn.LeakyReLU(0.2, inplace=True),
            )

    def forward(self, x):
        return self.main(x)

class Discriminator(th.nn.Module):
    def __init__(self, colour_channels, feature_map_depth) -> None:
        super().__init__()
        self.main = th.nn.Sequential(
            DiscriminatorBlock(colour_channels, feature_map_depth),
            DiscriminatorBlock(feature_map_depth, feature_map_depth * 2),
            DiscriminatorBlock(feature_map_depth * 2, feature_map_depth * 4),
            DiscriminatorBlock(feature_map_depth * 4, feature_map_depth * 8),
            DiscriminatorBlock(feature_map_depth * 8, feature_map_depth * 8),
            DiscriminatorBlock(feature_map_depth * 8, last=True)
        )

    def forward(self, x):
        x = self.main(x)
        return x