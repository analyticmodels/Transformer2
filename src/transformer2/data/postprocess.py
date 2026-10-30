"""
Corpus postprocessing, dispatched by dataset.

Each corpus needs different work -- Reddit strips discussion scaffolding and packs
comments, TinyStories normalises already-clean prose, Gutenberg strips the Project
Gutenberg boilerplate and chunks book-length text -- so their options differ and
cannot share one flat command line. Subcommands keep each tool's flags its own:

    t2-postprocess reddit --max_tokens 512
    t2-postprocess tinystories --max_tokens 320
    t2-postprocess gutenberg --max_tokens 512
"""

import argparse

from transformer2.data.gutenberg import postprocess_gutenberg
from transformer2.data.reddit import postprocess_reddit_data
from transformer2.data.tinystories import postprocess_tinystories


def main():
    parser = argparse.ArgumentParser(
        description='Clean and chunk a collected corpus for training')
    sub = parser.add_subparsers(dest='dataset', required=True,
                                help='Corpus to postprocess')

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument('--input', default=None, help='Source parquet')
    common.add_argument('--output', default=None, help='Destination parquet')
    common.add_argument('--max_tokens', type=int, required=True,
                        help='Maximum tokens per row; match your training --max_length')
    common.add_argument('--min_tokens', type=int, default=32,
                        help='Discard rows shorter than this (default: 32)')
    common.add_argument('--keep_duplicates', action='store_true',
                        help='Keep identical rows instead of deduplicating')

    reddit = sub.add_parser('reddit', parents=[common],
                            help='Strip discussion scaffolding and chunk by post')
    reddit.add_argument('--keep_titles', action='store_true',
                        help='Include post titles (excluded by default as metadata)')

    sub.add_parser('tinystories', parents=[common],
                   help='Normalise, deduplicate and length-bound the stories')

    sub.add_parser('gutenberg', parents=[common],
                   help='Chunk the collected books, one book at a time')

    args = parser.parse_args()
    shared = dict(input_path=args.input, output_path=args.output,
                  max_tokens=args.max_tokens, min_tokens=args.min_tokens,
                  drop_duplicates=not args.keep_duplicates)

    if args.dataset == 'reddit':
        postprocess_reddit_data(keep_titles=args.keep_titles, **shared)
    elif args.dataset == 'tinystories':
        postprocess_tinystories(**shared)
    elif args.dataset == 'gutenberg':
        postprocess_gutenberg(**shared)
    else:
        parser.error(f"No postprocessor for {args.dataset!r}")


if __name__ == "__main__":
    main()
