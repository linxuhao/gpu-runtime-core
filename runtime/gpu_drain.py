#!/usr/bin/env python3
"""Drain a GPU for maintenance, or put it back to work. An operator's tool.

    tools/gpu_drain.py gpu1            # finish what is queued there, take nothing new, then empty it
    tools/gpu_drain.py --resume gpu1   # take work again; the idle residency comes back

While a device is drained, new work that can go elsewhere (a tenant's fallback device) goes there
and the rest is refused; it stays drained across runtime restarts. `gpu-top` shows it.
"""
import argparse
import json

from gpu_borrow import call, default_socket


def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('device');parser.add_argument('--resume',action='store_true')
    parser.add_argument('--socket',default=default_socket(__file__))
    args=parser.parse_args()
    print(json.dumps(call(args.socket,'device_drain',device=args.device,drain=not args.resume)))


if __name__=='__main__':main()
