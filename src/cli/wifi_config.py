#!/usr/bin/env python3
"""
ESP32-CAM Dynamic Wi-Fi Provisioning CLI Tool

Use this CLI tool to configure the Wi-Fi SSID and password on an ESP32-CAM via USB Serial.
The credentials will be saved into the ESP32's NVS (Non-Volatile Storage) flash memory,
so it automatically connects on boot across different environments.
"""

import argparse
import getpass
import sys
import time

try:
    import serial
    import serial.tools.list_ports
except ImportError:
    print("Error: 'pyserial' package is required. Install it using: pip install pyserial")
    sys.exit(1)


def list_com_ports():
    """Returns a list of available serial ports."""
    ports = list(serial.tools.list_ports.comports())
    return ports


def select_com_port(explicit_port=None):
    """Detects or interactively prompts for serial COM port."""
    ports = list_com_ports()

    if not ports:
        print("❌ Error: No serial (COM) ports detected! Please connect your ESP32-CAM via USB.")
        sys.exit(1)

    if explicit_port:
        for p in ports:
            if p.device.upper() == explicit_port.upper():
                return p.device
        print(f"⚠️ Warning: Requested port '{explicit_port}' not found in active ports. Trying anyway...")
        return explicit_port

    if len(ports) == 1:
        selected = ports[0].device
        print(f"🔌 Auto-detected ESP32-CAM port: {selected} ({ports[0].description})")
        return selected

    print("\n🔌 Available Serial Ports:")
    for idx, p in enumerate(ports, 1):
        print(f"  [{idx}] {p.device} - {p.description}")

    while True:
        try:
            choice = input("\nSelect port number (e.g. 1): ").strip()
            num = int(choice)
            if 1 <= num <= len(ports):
                return ports[num - 1].device
        except ValueError:
            pass
        print("Invalid choice, please try again.")


def scan_wifi_networks(ser):
    """Requests ESP32 to scan surrounding Wi-Fi networks and returns a list."""
    print("\n🔍 Requesting ESP32-CAM to scan Wi-Fi networks...")
    ser.write(b"SCAN_WIFI\n")
    ser.flush()

    networks = []
    start_time = time.time()
    in_scan = False

    while time.time() - start_time < 10:
        if ser.in_waiting:
            line = ser.readline().decode("utf-8", errors="ignore").strip()
            if line == "SCAN_START":
                in_scan = True
            elif line.startswith("WIFI_NET:"):
                parts = line[len("WIFI_NET:") :].split("|")
                if len(parts) >= 2:
                    ssid = parts[0]
                    try:
                        rssi = int(parts[1])
                    except ValueError:
                        rssi = -100
                    if ssid and ssid not in [n[0] for n in networks]:
                        networks.append((ssid, rssi))
            elif line == "SCAN_END":
                break
        time.sleep(0.05)

    networks.sort(key=lambda x: x[1], reverse=True)
    return networks


def send_wifi_credentials(ser, ssid, password):
    """Sends new Wi-Fi credentials to ESP32 over serial and waits for connection status."""
    cmd = f"SET_WIFI:{ssid}:{password}\n"
    print(f"\n📡 Provisioning ESP32-CAM with Wi-Fi Network: '{ssid}'...")

    ser.write(cmd.encode("utf-8"))
    ser.flush()

    connected = False
    assigned_ip = None
    start_time = time.time()

    while time.time() - start_time < 20:
        if ser.in_waiting:
            line = ser.readline().decode("utf-8", errors="ignore").strip()
            if line:
                print(f"  [ESP32] {line}")

            if "OK:CONNECTED" in line or "[Wi-Fi Connected!]" in line:
                connected = True
            elif line.startswith("IP:") or "Ready at: http://" in line:
                assigned_ip = line.replace("IP:", "").replace("High-FPS Camera Stream Ready at: http://", "").strip()
                break
            elif "ERROR:CONNECT_FAILED" in line:
                break
        time.sleep(0.1)

    return connected, assigned_ip


def main():
    parser = argparse.ArgumentParser(description="ESP32-CAM Dynamic Wi-Fi Provisioning CLI Tool")
    parser.add_argument("--ssid", type=str, help="Target Wi-Fi SSID")
    parser.add_argument("--password", type=str, help="Target Wi-Fi Password")
    parser.add_argument("--port", type=str, help="Serial COM port (e.g. COM3 or /dev/ttyUSB0)")
    parser.add_argument("--baud", type=int, default=115200, help="Baud rate (default: 115200)")

    args = parser.parse_args()

    print("==================================================")
    print("⚡ ESP32-CAM Dynamic Wi-Fi Provisioning Tool")
    print("==================================================")

    port = select_com_port(args.port)

    try:
        ser = serial.Serial(port, args.baud, timeout=2)
        time.sleep(1.5)  # Wait for serial reset/initialization
    except Exception as e:
        print(f"❌ Failed to open serial port {port}: {e}")
        sys.exit(1)

    target_ssid = args.ssid
    target_pass = args.password

    # Interactive mode if arguments are missing
    if not target_ssid:
        networks = scan_wifi_networks(ser)

        if networks:
            print("\n📶 Discovered Wi-Fi Networks around ESP32-CAM:")
            for idx, (s, rssi) in enumerate(networks, 1):
                signal = "📶 Excellent" if rssi > -60 else ("📶 Good" if rssi > -75 else "📶 Weak")
                print(f"  [{idx}] {s:<25} ({signal}, {rssi} dBm)")
            print(f"  [{len(networks) + 1}] Custom / Hidden Network...")

            while True:
                choice = input("\nSelect Wi-Fi Network number: ").strip()
                try:
                    num = int(choice)
                    if 1 <= num <= len(networks):
                        target_ssid = networks[num - 1][0]
                        break
                    elif num == len(networks) + 1:
                        target_ssid = input("Enter Wi-Fi SSID: ").strip()
                        break
                except ValueError:
                    pass
                print("Invalid choice, please try again.")
        else:
            target_ssid = input("\nEnter Wi-Fi SSID: ").strip()

    if target_pass is None:
        target_pass = getpass.getpass(f"Enter Wi-Fi Password for '{target_ssid}': ")

    # Provision credentials
    connected, ip = send_wifi_credentials(ser, target_ssid, target_pass)

    if connected or ip:
        print("\n==================================================")
        print("🎉 SUCCESS: ESP32-CAM Connected & Provisioned!")
        print(f"  • Saved SSID: {target_ssid}")
        if ip:
            print(f"  • Stream Address: http://{ip}")
        print("  • Note: Credentials saved in NVS flash across reboots!")
        print("==================================================")
    else:
        print("\n==================================================")
        print("⚠️ WARNING: Could not verify Wi-Fi connection.")
        print("  • Credentials were sent and saved to NVS.")
        print("  • ESP32-CAM will attempt to connect on boot.")
        print("  • If it fails, fallback AP 'ESP32-CAM-Setup' will start.")
        print("==================================================")

    ser.close()


if __name__ == "__main__":
    main()
