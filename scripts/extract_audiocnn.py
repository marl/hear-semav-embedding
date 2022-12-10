import torch
from argparse import ArgumentParser


def extract_audiocnn(model_path, output_path, net_prefix="actor_critic.net.goal_encoder"):
    saved_model = torch.load(model_path, map_location=torch.device('cpu'))

    if not net_prefix.endswith('.'):
        net_prefix += '.'
    torch.save(
        {
            k.replace(net_prefix, ''): v
            for k, v in saved_model['state_dict'].items()
            if k.startswith(net_prefix)
        },
        output_path,
    )


if __name__ == "__main__":
    parser = ArgumentParser("Extracts AudioCNN from trained SAVi model")
    parser.add_argument(
        "model_path",
        type=str,
        help='Path to SAVi model checkpoint',
    )
    parser.add_argument(
        "output_path",
        type=str,
        help='Path to save AudioCNN checkpoint'
    )
    parser.add_argument(
        "--net-prefix",
        type=str,
        help="Prefix corresponding to AudioCNN model dot path in SAVi model",
    )
    args = vars(parser.parse_args())

    extract_audiocnn(**args)
