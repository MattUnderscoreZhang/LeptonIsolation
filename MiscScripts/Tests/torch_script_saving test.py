import torch

"""
saving test
"""


class Model(torch.jit.ScriptModule):
    """dummy model """

    def __init__(self):
        super(Model, self).__init__()
        pass

    @torch.jit.script_method
    def forward(self):
        a = torch.rand([6, 1, 12])
        b = torch.rand([6, 1, 12])
        out = torch.cat([a, b], dim=2)
        return out

    def save_to_pytorch(self, output_path):
        torch.jit.save(self, output_path)


if __name__ == "__main__":
    model = Model()
    print(model())  # works
    script = torch.jit.script(model)  # throws error
