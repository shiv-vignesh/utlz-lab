from __future__ import annotations

from dataclasses import dataclass

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.utils.argparse_utils import FlexibleArgumentParser

@dataclass
class ServerConfig:
    engine_args: AsyncEngineArgs
    host: str = "127.0.0.1"
    port: int = 8000
    api_key: str | None = None
    served_model_name: str | None = None
    enable_lab_admin: bool = False

    @classmethod
    def from_cli(cls, argv: list[str] | None = None) -> ServerConfig:
        p = FlexibleArgumentParser(description="utlz-lab vllm server")
        p.add_argument("--host", default=ServerConfig.host)
        p.add_argument("--port", type=int, default=ServerConfig.port)
        p.add_argument("--api-key", default=None)
        p.add_argument("--enable-lab-admin", action="store_true")

        p = AsyncEngineArgs.add_cli_args(p)
        args = p.parse_args(argv)

        if args.host == "0.0.0.0" and not args.api_key:
            raise ValueError(
                "--api-key must be provided and non-empty when --host is 0.0.0.0"
            )        

        if args.served_model_name is None:
            raise ValueError(
                "--served-model-name must be provided and non-empty"
            )

        return cls(
            engine_args=AsyncEngineArgs.from_cli_args(args),
            host=args.host, port=args.port, api_key=args.api_key,
            enable_lab_admin=args.enable_lab_admin,
            served_model_name=(args.served_model_name[0] if isinstance(args.served_model_name, list) else args.served_model_name)
        )