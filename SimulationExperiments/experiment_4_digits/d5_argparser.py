import argparse


def parser_args():
    parser = argparse.ArgumentParser(description='Digits 5 classification')
    parser.add_argument('--run_all',
                        help='specify to run all experiments or select parameters for single experiment',
                        type=int,
                        default=1)

    parser.add_argument('--method',
                        help='cosine_similarity, MMD, projected',
                        type=str,
                        default='cosine_similarity')

    parser.add_argument('--TARGET_DOMAIN',
                        help='mnistm, mnist, syn, svhn, usps',
                        type=str,
                        default='mnistm')  # -1 so that we do not overwrite config if we do not pass anything

    parser.add_argument('--lambda_sparse',
                        default=1e-3,
                        type=float)

    parser.add_argument('--lambda_OLS',
                        type=float,
                        default=1e-3)

    parser.add_argument('--lambda_orth',
                        type=float,
                        default=0)

    parser.add_argument('--early_stopping',
                        type=bool,
                        default=True)

    parser.add_argument('--fine_tune',
                        type=bool,
                        default=True)

    parser.add_argument('--res_file_dir',
                        type=str,
                        default='/local/home/pokanovic/project2/results/srip')

    parser.add_argument('--running',
                        type=int,
                        default=0)

    args = parser.parse_args()
    if args.method == 'None':
        args.method = None
    args.fine_tune = bool(args.fine_tune)
    return args
