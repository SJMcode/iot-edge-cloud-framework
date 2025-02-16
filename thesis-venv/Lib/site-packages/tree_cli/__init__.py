#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -*- author: sgt911 -*-

from argparse import ArgumentParser, Namespace
from getpass import getpass

from tree_cli.parsers import Parser
from tree_cli.parsers.users import UserParser
from tree_cli.parsers.roots import RootParser
from tree_cli.parsers.nodes import NodeParser

from tree_cli.handlers import Handler
from tree_cli.handlers.users import UserHandler
from tree_cli.handlers.roots import RootHandler


def parse_args() -> Namespace:
    parser = ArgumentParser(prog="tree-cli", description="Tree-DB command line client")

    parser.add_argument(
        "--url",
        default="tree://localhost:9110",
        help="url endpoint, example: 'tree://localhost:9110'",
    )

    parser.add_argument(
        "-U",
        "--user",
        type=str,
        help="auth username in case of authentication is active on server",
    )

    parser.add_argument(
        "-P",
        "--password",
        type=str,
        help="auth password in case of authentication is active on server",
    )

    parser.add_argument(
        "namespace",
        type=str,
        choices=["user", "root", "node"],
    )

    parser.add_argument(
        "command",
        type=str,
        choices=["list", "get", "create", "update", "delete"],
    )

    parser.add_argument(
        "args",
        nargs="*",
        help="positional arguments",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    auth = False
    if args.user is not None and args.password is None:
        auth = True
        args.password = getpass("Password: ")
        if args.password == "":
            print("error: password must be supplied")
            exit(2)

    parser_class = Parser
    handler_class = Handler
    if args.namespace == "user":
        parser_class = UserParser
        handler_class = UserHandler
    elif args.namespace == "root":
        parser_class = RootParser
        handler_class = RootHandler
    elif args.namespace == "node":
        parser_class = NodeParser
        handler_class = RootHandler

    parser = parser_class(args.command, args.args)
    handler = handler_class(args.command, parser.get(), args.url)

    if auth:
        handler.set_authentication(args.user, args.password)

    return handler()
