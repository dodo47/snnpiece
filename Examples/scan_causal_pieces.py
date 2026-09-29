import argparse
import os
import numpy as np
import torch
from tqdm import trange
import yaml
from snnpiece.analysis.datasets.image_data import scan_through_dataset as scan_image_dataset
from snnpiece.analysis.datasets.yinyang import scan_through_dataset as scan_yinyang_dataset
from snnpiece.datasets import get_EuroSAT_dataloaders
from snnpiece.datasets import get_FashionMNIST_dataloaders
from snnpiece.datasets import get_MNIST_dataloaders
from snnpiece.datasets import get_YinYang_dataloaders
from snnpiece.networks import MLPExpIF
from snnpiece.networks import MLPLIF
from snnpiece.training import test
from snnpiece.training import train as train_epoch

MAIN_PATH = "./results/"

def list_of_ints(arg):
    return list(map(int, arg.split(",")))

def bool_arg(value):
    if isinstance(value, bool):
        return value

    value = value.lower()
    if value in ("1", "true", "t", "yes", "y", "on"):
        return True
    if value in ("0", "false", "f", "no", "n", "off"):
        return False

    raise argparse.ArgumentTypeError("expected a boolean value")

def default_neurons(dataset):
    if dataset == "YinYang":
        return [4, 30, 3]
    if dataset in {"MNIST", "FashionMNIST"}:
        return [28 * 28, 200, 10]
    if dataset == "EuroSAT":
        return [3 * 16 * 16, 200, 10]

    raise ValueError(f"Unknown dataset: {dataset}")

def save_data(results_no_pieces, piece_sizes, set_sizes, ind, path):
    np.savetxt(os.path.join(path, "no_pieces.txt"), np.array(results_no_pieces))

    if piece_sizes is not None:
        np.save(os.path.join(path, f"piece_sizes_{ind}.npy"), piece_sizes)

    if set_sizes is not None:
        with open(os.path.join(path, f"set_sizes_{ind}.yaml"), "w") as outfile:
            yaml.dump(set_sizes, outfile)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train an LIF/nLIF model and evaluate causal pieces on training data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output-dir", default=MAIN_PATH, help="Directory for all results.")
    parser.add_argument("--data-path", default="data", help="Dataset download/cache directory for image datasets.")
    parser.add_argument("--num-seeds", type=int, default=50, help="Number of random seeds to run.")
    parser.add_argument("--seed-rng", type=int, default=5234, help="Seed used to generate run seeds.")
    parser.add_argument(
        "--seeds",
        type=list_of_ints,
        default=None,
        help="Comma-separated explicit run seeds. Overrides --num-seeds and --seed-rng.",
    )
    parser.add_argument("--batch-size", type=int, default=200, help="Training and test batch size.")
    parser.add_argument("--neuron-model", choices=("nLIF", "LIF"), default="nLIF", help="Neuron model.")
    parser.add_argument("--dataset", choices=("YinYang", "EuroSAT", "MNIST", "FashionMNIST"), default="YinYang", help="Dataset.")
    parser.add_argument("--max-epochs", type=int, default=200, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=1e-2, help="Learning rate.")
    parser.add_argument("--gamma", type=float, default=0, help="Weight bumping strength.")
    parser.add_argument("--loss", choices=("CE", "ttfs"), default="CE", help="Training loss.")
    parser.add_argument("--neurons", type=list_of_ints, default=None, help="Comma-separated layer sizes.")
    parser.add_argument("--taus", type=float, default=0.5, help="Neuron time constant.")
    parser.add_argument("--decoder", choices=("linear", "ttfs"), default="linear", help="Decoder type.")
    parser.add_argument("--init-type", default="normal", help="Weight initialisation distribution.")
    parser.add_argument("--init-offset", type=float, default=0.2, help="Offset subtracted from sampled init mean.")
    parser.add_argument("--trainable-decoder", type=bool_arg, default=True, help="Whether the decoder is trainable.")
    parser.add_argument("--positive-weights", type=bool_arg, default=False, help="Constrain weights to be positive.")
    parser.add_argument("--normed-grads", type=bool_arg, default=False, help="Normalise gradients during training.")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device used for training and testing.",
    )
    args = parser.parse_args()

    if args.neurons is None:
        args.neurons = default_neurons(args.dataset)

    return args

def get_dataloaders(args):
    if args.dataset == "YinYang":
        return get_YinYang_dataloaders(batch_size=args.batch_size)
    if args.dataset == "EuroSAT":
        return get_EuroSAT_dataloaders(args.data_path, batch_size=args.batch_size)
    if args.dataset == "MNIST":
        return get_MNIST_dataloaders(args.data_path, batch_size=args.batch_size)
    if args.dataset == "FashionMNIST":
        return get_FashionMNIST_dataloaders(args.data_path, batch_size=args.batch_size)

    raise ValueError(f"Unknown dataset: {args.dataset}")

def scan_dataset(model, args):
    if args.dataset == "YinYang":
        _, num_pieces, piece_sizes, set_sizes = scan_yinyang_dataset(model, batchsize=args.batch_size)
        return num_pieces, piece_sizes, set_sizes

    num_pieces = scan_image_dataset(model, args.dataset, batchsize=args.batch_size)
    return num_pieces, None, None

def build_model(seed, args, init_params):
    model_cls = MLPExpIF if args.neuron_model == "nLIF" else MLPLIF
    return model_cls(
        layer_dim=args.neurons,
        taus=args.taus,
        seed=seed + 666,
        decoder=args.decoder,
        positive_weights=args.positive_weights,
        return_detailed_output=True,
        trainable_decoder=args.trainable_decoder,
        init_type=args.init_type,
        init_params=init_params,
    )

def run_seed(seed, args):
    np.random.seed(seed)
    mu = np.random.random() - args.init_offset
    sigma = np.random.random()
    init_params = [mu, sigma]

    filename = f"{args.dataset}-{args.neuron_model}-{args.init_type}-{seed}"
    path = os.path.join(args.output_dir, filename)
    os.makedirs(path, exist_ok=True)

    train_loader, test_loader = get_dataloaders(args)
    model = build_model(seed, args, init_params)

    results_no_pieces = []
    num_pieces, piece_sizes, set_sizes = scan_dataset(model, args)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    results_no_pieces.append(num_pieces)
    save_data(results_no_pieces, piece_sizes, set_sizes, 0, path)

    torch.save(model.state_dict(), os.path.join(path, "model_0.pt"))
    accs = []
    model.to(args.device)
    model.return_detailed_output = False
    accuracy = test(model, args.device, test_loader, loss=args.loss)
    accs.append(accuracy)
    for epoch in trange(1, args.max_epochs + 1):
        model.to(args.device)
        model.return_detailed_output = False
        train_epoch(
            model,
            args.device,
            train_loader,
            optimizer,
            gamma=args.gamma,
            normed_grads=args.normed_grads,
            loss=args.loss,
        )
        accuracy = test(model, args.device, test_loader, loss=args.loss)
        accs.append(accuracy)
        torch.save(model.state_dict(), os.path.join(path, f"model_{epoch}.pt"))
        np.savetxt(os.path.join(path, "accuracy.txt"), np.array(accs))
        print(f"Epoch {epoch}: test accuracy = {accuracy}")

    best_id = np.argmax(accs)
    model.load_state_dict(torch.load(os.path.join(path, f"model_{best_id}.pt"), map_location=args.device))
    num_pieces, piece_sizes, set_sizes = scan_dataset(model, args)
    results_no_pieces.append(num_pieces)
    save_data(results_no_pieces, piece_sizes, set_sizes, "best", path)

if __name__ == "__main__":
    args = parse_args()
    if args.seeds is None:
        np.random.seed(args.seed_rng)
        seeds = np.random.randint(1e6, size=args.num_seeds)
    else:
        seeds = args.seeds

    for seed in seeds:
        run_seed(seed, args)
