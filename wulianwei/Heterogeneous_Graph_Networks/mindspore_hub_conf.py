from src.bgcf import BGCF
from src.config import parser_args

def bgcf_net(*args, **kwargs):
    return BGCF(*args, **kwargs)

def create_network(name, *args, **kwargs):
    if name == "bgcf":
        config = parser_args()
        config.num_user = 7068
        config.num_item = 3570
        return bgcf_net([config.input_dim, config.num_user, config.num_item],
                        config.embedded_dimension,
                        config.activation,
                        [0.0, 0.0, 0.0],
                        config.num_user,
                        config.num_item,
                        config.input_dim,
                        *args, **kwargs)
    raise NotImplementedError(f"{name} is not implemented in the repo")
