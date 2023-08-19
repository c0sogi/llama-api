from llama_api.server.app_settings import run, parser


if __name__ == "__main__":
    parser.add_argument(
        "-p",
        "--port",
        type=int,
        default=8000,
        help="Port to run the server on; default is 8000",
    )
    parser.add_argument(
        "-w",
        "--max-workers",
        type=int,
        default=1,
        help="Maximum number of process workers to run; default is 1",
    )
    parser.add_argument(
        "-k",
        "--api-key",
        type=str,
        default=None,
        help="API key to use for the server",
    )
    parser.add_argument(
        "-x",
        "--xformers",
        action="store_true",
        help="Apply xformers' memory-efficient optimizations",
    )
    parser.add_argument(
        "-ne",
        "--no-embed",
        action="store_true",
        help="Disable embeddings endpoint",
    )

    args = parser.parse_args()
    run(
        port=args.port,
        install_packages=args.install_pkgs,
        force_cuda=args.force_cuda,
        skip_pytorch_install=args.skip_torch_install,
        skip_tensorflow_install=args.skip_tf_install,
        skip_compile=args.skip_compile,
        no_cache=args.no_cache_dir,
        environs={
            "LLAMA_API_MAX_WORKERS": str(args.max_workers),
            "LLAMA_API_XFORMERS": "1" if args.xformers else "",
            "LLAMA_API_API_KEY": args.api_key or "",
            "FORCE_CUDA": "1" if args.force_cuda else "",
            "LLAMA_API_EMBEDDINGS": "1" if not args.no_embed else "",
        },
    )
