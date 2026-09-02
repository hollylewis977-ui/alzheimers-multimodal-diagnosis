# -*- coding: utf-8 -*-
"""启动开发服务器：python run.py

生产部署请用 WSGI 服务器指向 adapp:create_app()，例如：
    waitress-serve --port=5000 --call adapp:create_app
"""
from __future__ import annotations

import argparse

from adapp import create_app


def main() -> None:
    parser = argparse.ArgumentParser(description="阿尔茨海默病风险预测 Web 应用")
    parser.add_argument("--host", default="127.0.0.1", help="监听地址（默认仅本机）")
    parser.add_argument("--port", type=int, default=5000, help="监听端口")
    parser.add_argument("--debug", action="store_true", help="开启 Flask 调试模式")
    args = parser.parse_args()

    app = create_app()
    print(f"\n打开 http://{args.host}:{args.port}/\n")
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
