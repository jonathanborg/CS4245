import torch as th

class DiscriminatorBlock(th.nn.Module):
    def __init__(self, channels, first=False, last=False, colour_channels=0) -> None:
        assert(not (first and last)) # block can't be both first and last
        assert((not first) or colour_channels > 0) # last -> colour_channels > 0 (colour_channels needs to be set for the last layer)
        super().__init__()
        if first:
            self.main = th.nn.Sequential(
                th.nn.Conv2d(colour_channels, channels, 4, 2, 1, bias=False),
                th.nn.LeakyReLU(0.2, inplace=True),
            )
        elif last:
            self.main = th.nn.Sequential(
                th.nn.Conv2d(channels, 1, 3, 1, 0, bias=False),
                th.nn.Sigmoid()
            )
        else:
            self.main = th.nn.Sequential(
                th.nn.Conv2d(channels, channels * 2, 4, 2, 1, bias=False),
                th.nn.BatchNorm2d(channels * 2),
                th.nn.LeakyReLU(0.2, inplace=True),
            )

    def forward(self, x):
        return self.main(x)

class Discriminator(th.nn.Module):
    def __init__(self, colour_channels, feature_map_depth) -> None:
        super().__init__()
        self.main = th.nn.Sequential(
            DiscriminatorBlock(feature_map_depth, first=True, colour_channels=colour_channels),
            DiscriminatorBlock(feature_map_depth),
            DiscriminatorBlock(feature_map_depth * 2),
            DiscriminatorBlock(feature_map_depth * 4),
            DiscriminatorBlock(feature_map_depth * 8),
            DiscriminatorBlock(feature_map_depth * 16, last=True)
        )

    def forward(self, x):
        x = self.main(x)
        return x