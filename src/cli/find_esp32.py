"""
ESP32-CAM Local Network Discovery Tool
1. Listens for live UDP broadcast beacons emitted by the ESP32-CAM (Port 8888).
2. Performs fast multi-subnet parallel HTTP scanning if UDP beaconing is restricted.
"""

import os
import sys
import socket
import urllib.request
import concurrent.futures
import time

UDP_PORT = 8888

def listen_udp_beacon(timeout=2.5):
    """Listen for UDP broadcast beacon from ESP32-CAM."""
    print("[*] Listening for live ESP32-CAM UDP beacons on port 8888...")
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        sock.settimeout(timeout)
        sock.bind(('', UDP_PORT))

        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                data, addr = sock.recvfrom(1024)
                text = data.decode('utf-8', errors='ignore')
                if "ESP32-CAM:" in text:
                    ip = addr[0]
                    url = text.replace("ESP32-CAM:", "").strip()
                    sock.close()
                    return ip, url
            except socket.timeout:
                break
        sock.close()
    except Exception as e:
        pass
    return None, None

def get_local_ip_info():
    """Get local IP and generate candidate subnets to scan."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
    except Exception:
        local_ip = "192.168.1.100"

    parts = [int(p) for p in local_ip.split(".")]
    
    # Generate multi-subnet list (e.g. 10.107.56.x to 10.107.60.x or 192.168.0.x to 192.168.3.x)
    subnets = []
    base_b = parts[2]
    for b in range(max(0, base_b - 2), min(255, base_b + 3)):
        subnets.append(f"{parts[0]}.{parts[1]}.{b}")

    # Common fallbacks
    for fallback in ["192.168.1", "192.168.0", "192.168.43", "10.0.0"]:
        if fallback not in subnets:
            subnets.append(fallback)

    return local_ip, subnets

def check_ip(ip):
    """Attempt HTTP GET request to check if IP is our ESP32-CAM."""
    url = f"http://{ip}/"
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'ESP32Finder'})
        with urllib.request.urlopen(req, timeout=0.6) as response:
            html = response.read().decode('utf-8', errors='ignore')
            if "ESP32-CAM" in html or "Camera" in html or "stream" in html:
                return ip
    except Exception:
        pass
    return None

def scan_network():
    # Step 1: Listen for UDP Beacon
    beacon_ip, beacon_url = listen_udp_beacon(timeout=2.0)
    if beacon_ip:
        print(f"\n[+] BEACON DETECTED! ESP32-CAM is online at: {beacon_url}")
        return

    # Step 2: Multi-Subnet Parallel HTTP Scan
    local_ip, subnets = get_local_ip_info()
    print(f"\n[*] Local PC IP: {local_ip}")
    print(f"[*] Scanning {len(subnets)} subnets in parallel...")

    ips_to_scan = []
    for sub in subnets:
        for host in range(1, 255):
            ips_to_scan.append(f"{sub}.{host}")

    found_devices = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=100) as executor:
        results = executor.map(check_ip, ips_to_scan)
        for ip in results:
            if ip:
                found_devices.append(ip)
                print(f"[+] FOUND ESP32-CAM at: http://{ip}")

    if not found_devices:
        print("\n[-] No ESP32-CAM devices responded.")
        print("[*] Alternative Ways to Access:")
        print(" 1. Mobile App: Use 'Fing' app (iOS/Android) to scan Wi-Fi devices.")
        print(" 2. Router Admin: Log in to your router (e.g. 192.168.1.1) to view DHCP Clients.")
    else:
        print(f"\n[+] Total ESP32-CAM devices found: {len(found_devices)}")

if __name__ == "__main__":
    scan_network()
