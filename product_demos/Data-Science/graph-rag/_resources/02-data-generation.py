# Databricks notebook source
# MAGIC %md
# MAGIC # Data Generation for Graph RAG Demo
# MAGIC
# MAGIC This notebook generates synthetic technical documentation for networking products.
# MAGIC It creates:
# MAGIC - ~15-20 products (routers, modems, switches, access points)
# MAGIC - ~60-80 features per product category
# MAGIC - ~40-60 error codes with descriptions
# MAGIC - ~80-100 solutions linked to errors and products
# MAGIC - PDF documentation per product
# MAGIC
# MAGIC The generated data is saved as parquet files in the volume for downstream use.

# COMMAND ----------

# MAGIC %pip install -U -qqqq faker weasyprint
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ../config

# COMMAND ----------

# MAGIC %run ../../../../_resources/00-global-setup-v2

# COMMAND ----------

DBDemos.setup_schema(catalog, db, False, volume_name)
volume_folder = f"/Volumes/{catalog}/{db}/{volume_name}"

# COMMAND ----------

import random
import json
from faker import Faker

fake = Faker()
Faker.seed(42)
random.seed(42)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Define Products, Features, Errors, and Solutions

# COMMAND ----------

# Product definitions
products = [
    {"id": "PROD-001", "name": "Router X500",        "category": "Router",       "description": "Enterprise-grade wireless router with WiFi 6E support, dual-band connectivity, and advanced QoS management."},
    {"id": "PROD-002", "name": "Router X700",        "category": "Router",       "description": "High-performance tri-band router with mesh networking capability and built-in VPN server."},
    {"id": "PROD-003", "name": "Router X300",        "category": "Router",       "description": "Budget-friendly home router with WiFi 6 support and basic parental controls."},
    {"id": "PROD-004", "name": "Modem M200",         "category": "Modem",        "description": "DOCSIS 3.1 cable modem supporting speeds up to 2.5 Gbps with dual Ethernet ports."},
    {"id": "PROD-005", "name": "Modem M400",         "category": "Modem",        "description": "Fiber optic modem with integrated ONT and 10G SFP+ port for enterprise deployment."},
    {"id": "PROD-006", "name": "Modem M100",         "category": "Modem",        "description": "Entry-level DSL modem with integrated router functionality and basic firewall."},
    {"id": "PROD-007", "name": "Switch SW-3000",     "category": "Switch",       "description": "48-port managed Gigabit Ethernet switch with PoE+ and Layer 3 routing capability."},
    {"id": "PROD-008", "name": "Switch SW-1000",     "category": "Switch",       "description": "24-port unmanaged switch with auto-MDI/MDIX and energy-efficient Ethernet."},
    {"id": "PROD-009", "name": "Switch SW-5000",     "category": "Switch",       "description": "Enterprise 96-port switch with 10G uplinks, VLAN support, and SNMP management."},
    {"id": "PROD-010", "name": "AP Pro-600",         "category": "Access Point", "description": "WiFi 6E access point with tri-band support, MU-MIMO, and cloud management."},
    {"id": "PROD-011", "name": "AP Pro-400",         "category": "Access Point", "description": "Indoor/outdoor access point with IP67 rating and mesh networking support."},
    {"id": "PROD-012", "name": "AP Pro-800",         "category": "Access Point", "description": "High-density access point supporting 500+ concurrent clients with AI-driven RF optimization."},
    {"id": "PROD-013", "name": "Firewall FW-2000",   "category": "Firewall",     "description": "Next-generation firewall with deep packet inspection, IDS/IPS, and SSL decryption."},
    {"id": "PROD-014", "name": "Firewall FW-500",    "category": "Firewall",     "description": "SMB firewall with integrated VPN, content filtering, and threat intelligence feeds."},
    {"id": "PROD-015", "name": "Gateway GW-100",     "category": "Gateway",      "description": "IoT gateway with Zigbee, Z-Wave, and Bluetooth connectivity for smart building automation."},
    {"id": "PROD-016", "name": "Range Extender RE-50","category": "Extender",    "description": "Dual-band WiFi range extender with signal indicator and one-touch setup."},
    {"id": "PROD-017", "name": "NAS Drive ND-4000",  "category": "Storage",      "description": "4-bay network attached storage with RAID support and 2.5GbE connectivity."},
    {"id": "PROD-018", "name": "PoE Injector PI-30",  "category": "Accessory",   "description": "30W single-port PoE+ injector compatible with IEEE 802.3at devices."},
]

# Feature definitions
features = [
    {"id": "FEAT-001", "name": "WiFi 6E",             "description": "Support for 6 GHz band with up to 9.6 Gbps theoretical throughput."},
    {"id": "FEAT-002", "name": "Dual-Band",            "description": "Simultaneous 2.4 GHz and 5 GHz band operation for flexible connectivity."},
    {"id": "FEAT-003", "name": "Tri-Band",             "description": "Three separate radio bands including dedicated backhaul for mesh networking."},
    {"id": "FEAT-004", "name": "PoE (Power over Ethernet)", "description": "Delivers electrical power over Ethernet cables, eliminating need for separate power supplies."},
    {"id": "FEAT-005", "name": "VLAN Support",         "description": "Virtual LAN segmentation for network isolation and security."},
    {"id": "FEAT-006", "name": "QoS Management",       "description": "Quality of Service traffic prioritization for latency-sensitive applications."},
    {"id": "FEAT-007", "name": "VPN Server",           "description": "Built-in VPN server supporting OpenVPN and WireGuard protocols."},
    {"id": "FEAT-008", "name": "Mesh Networking",      "description": "Seamless roaming between multiple access points with single SSID."},
    {"id": "FEAT-009", "name": "MU-MIMO",              "description": "Multi-User Multiple Input Multiple Output for simultaneous multi-device communication."},
    {"id": "FEAT-010", "name": "SNMP Management",      "description": "Simple Network Management Protocol support for enterprise monitoring."},
    {"id": "FEAT-011", "name": "Deep Packet Inspection","description": "Layer 7 traffic analysis for application identification and threat detection."},
    {"id": "FEAT-012", "name": "Cloud Management",     "description": "Centralized cloud-based device management and monitoring dashboard."},
    {"id": "FEAT-013", "name": "Parental Controls",    "description": "Content filtering and screen time management for family networks."},
    {"id": "FEAT-014", "name": "Auto-MDI/MDIX",        "description": "Automatic cable type detection eliminating need for crossover cables."},
    {"id": "FEAT-015", "name": "10G SFP+ Uplink",      "description": "10 Gigabit SFP+ ports for high-speed backbone connections."},
    {"id": "FEAT-016", "name": "DOCSIS 3.1",           "description": "Cable modem standard supporting multi-gigabit downstream speeds."},
    {"id": "FEAT-017", "name": "IDS/IPS",              "description": "Intrusion Detection and Prevention System for real-time threat monitoring."},
    {"id": "FEAT-018", "name": "SSL Decryption",       "description": "Inspects encrypted traffic for malware and data exfiltration attempts."},
    {"id": "FEAT-019", "name": "Zigbee Connectivity",  "description": "Low-power mesh networking protocol for IoT device communication."},
    {"id": "FEAT-020", "name": "RAID Support",         "description": "Redundant Array of Independent Disks for data protection and performance."},
    {"id": "FEAT-021", "name": "Layer 3 Routing",      "description": "Inter-VLAN routing and static/dynamic routing protocol support."},
    {"id": "FEAT-022", "name": "802.3at PoE+",         "description": "Power over Ethernet Plus delivering up to 30W per port."},
    {"id": "FEAT-023", "name": "AI RF Optimization",   "description": "Machine learning-based radio frequency channel selection and power management."},
    {"id": "FEAT-024", "name": "WPA3 Security",        "description": "Latest WiFi security protocol with SAE authentication and 192-bit encryption."},
    {"id": "FEAT-025", "name": "Firmware Auto-Update",  "description": "Automatic firmware updates with rollback capability for security patches."},
]

# Product-to-feature mapping
product_features = {
    "PROD-001": ["FEAT-001", "FEAT-002", "FEAT-006", "FEAT-024", "FEAT-025", "FEAT-013"],
    "PROD-002": ["FEAT-001", "FEAT-003", "FEAT-007", "FEAT-008", "FEAT-006", "FEAT-024"],
    "PROD-003": ["FEAT-002", "FEAT-013", "FEAT-024", "FEAT-025"],
    "PROD-004": ["FEAT-016", "FEAT-025"],
    "PROD-005": ["FEAT-015", "FEAT-025"],
    "PROD-006": ["FEAT-002", "FEAT-025"],
    "PROD-007": ["FEAT-004", "FEAT-005", "FEAT-021", "FEAT-010", "FEAT-022"],
    "PROD-008": ["FEAT-014", "FEAT-004"],
    "PROD-009": ["FEAT-005", "FEAT-010", "FEAT-015", "FEAT-021", "FEAT-022"],
    "PROD-010": ["FEAT-001", "FEAT-003", "FEAT-009", "FEAT-012", "FEAT-024"],
    "PROD-011": ["FEAT-002", "FEAT-008", "FEAT-024"],
    "PROD-012": ["FEAT-001", "FEAT-009", "FEAT-023", "FEAT-012", "FEAT-024"],
    "PROD-013": ["FEAT-011", "FEAT-017", "FEAT-018", "FEAT-005"],
    "PROD-014": ["FEAT-007", "FEAT-011", "FEAT-017"],
    "PROD-015": ["FEAT-019"],
    "PROD-016": ["FEAT-002", "FEAT-024"],
    "PROD-017": ["FEAT-020", "FEAT-010"],
    "PROD-018": ["FEAT-022"],
}

# COMMAND ----------

# Error definitions
errors = [
    {"id": "ERR-001", "name": "Connection Timeout",           "description": "Device fails to establish connection within the configured timeout period. Often caused by signal interference or incorrect network configuration."},
    {"id": "ERR-002", "name": "DHCP Lease Failure",           "description": "Device cannot obtain an IP address from the DHCP server. May indicate DHCP pool exhaustion or misconfigured scope."},
    {"id": "ERR-003", "name": "Firmware Update Failed",       "description": "Firmware update process interrupted or corrupted. Device may revert to previous firmware version."},
    {"id": "ERR-004", "name": "PoE Power Budget Exceeded",    "description": "Total power consumption of PoE devices exceeds the switch's available power budget."},
    {"id": "ERR-005", "name": "VLAN Misconfiguration",        "description": "VLAN tagging mismatch between switch ports causing traffic isolation or connectivity issues."},
    {"id": "ERR-006", "name": "DNS Resolution Failure",       "description": "Device cannot resolve domain names. May be caused by incorrect DNS server configuration or upstream DNS outage."},
    {"id": "ERR-007", "name": "WiFi Authentication Error",    "description": "Client fails WPA2/WPA3 authentication. Common with outdated client drivers or incorrect credentials."},
    {"id": "ERR-008", "name": "SFP Module Not Recognized",    "description": "SFP+ transceiver module not detected by switch. May be incompatible vendor or dirty connector."},
    {"id": "ERR-009", "name": "High CPU Utilization",         "description": "Device CPU running above 90% for extended period. May cause packet drops and management interface unresponsiveness."},
    {"id": "ERR-010", "name": "Spanning Tree Loop Detected",  "description": "Network loop detected by STP/RSTP protocol. Affected ports will be blocked to prevent broadcast storms."},
    {"id": "ERR-011", "name": "VPN Tunnel Establishment Failed", "description": "IPSec or WireGuard tunnel cannot be established. Often caused by mismatched pre-shared keys or firewall rules."},
    {"id": "ERR-012", "name": "Memory Exhaustion",            "description": "Device RAM usage exceeds safe threshold. May cause service crashes and require device restart."},
    {"id": "ERR-013", "name": "Certificate Validation Error",  "description": "SSL/TLS certificate chain validation failed. May indicate expired certificate or untrusted CA."},
    {"id": "ERR-014", "name": "Mesh Node Disconnected",       "description": "Mesh satellite node lost connection to root node. Clients on this node will lose connectivity."},
    {"id": "ERR-015", "name": "Channel Congestion",           "description": "WiFi channel utilization exceeds 80%, causing throughput degradation and increased latency."},
    {"id": "ERR-016", "name": "RAID Array Degraded",          "description": "One or more disks in RAID array have failed. Data redundancy is reduced until replacement."},
    {"id": "ERR-017", "name": "IDS Alert: Port Scan Detected","description": "Intrusion detection system detected sequential port scanning activity from external source."},
    {"id": "ERR-018", "name": "NAT Table Overflow",           "description": "Network Address Translation table has reached maximum entries. New connections will be dropped."},
    {"id": "ERR-019", "name": "DFS Channel Switch",           "description": "Dynamic Frequency Selection detected radar, forcing channel change. Brief connectivity interruption expected."},
    {"id": "ERR-020", "name": "SNMP Community String Mismatch","description": "SNMP monitoring cannot authenticate. Community string does not match configured value on device."},
    {"id": "ERR-021", "name": "Zigbee Device Pairing Failed", "description": "IoT device fails to join Zigbee network. May need factory reset or closer proximity during pairing."},
    {"id": "ERR-022", "name": "QoS Policy Conflict",          "description": "Multiple QoS rules with conflicting priorities applied to same traffic class."},
    {"id": "ERR-023", "name": "Fiber Link Loss",              "description": "Optical power level dropped below receiver sensitivity threshold. Check fiber connectors and cable integrity."},
    {"id": "ERR-024", "name": "ARP Table Full",               "description": "ARP table has reached maximum capacity. Network may experience intermittent connectivity issues."},
    {"id": "ERR-025", "name": "WPS Authentication Timeout",   "description": "WiFi Protected Setup handshake timed out. Common when WPS is disabled on one end."},
    {"id": "ERR-026", "name": "Thermal Throttling Active",    "description": "Device temperature exceeded safe operating range. Performance reduced to prevent hardware damage."},
    {"id": "ERR-027", "name": "LACP Negotiation Failed",      "description": "Link Aggregation Control Protocol failed to form port-channel. Check partner switch configuration."},
    {"id": "ERR-028", "name": "Cloud Management Unreachable", "description": "Device cannot connect to cloud management platform. Verify internet connectivity and proxy settings."},
    {"id": "ERR-029", "name": "MAC Address Table Full",       "description": "Switch MAC address table capacity reached. Unknown unicast flooding may occur."},
    {"id": "ERR-030", "name": "Power Supply Failure",         "description": "One redundant power supply has failed. Device operating on single power supply."},
    {"id": "ERR-031", "name": "SSL Decryption Overload",      "description": "Firewall SSL inspection engine overloaded. Some encrypted traffic bypassing inspection."},
    {"id": "ERR-032", "name": "BGP Peer Down",                "description": "Border Gateway Protocol peer session lost. Routing table may be incomplete."},
    {"id": "ERR-033", "name": "DOCSIS Signal Out of Range",   "description": "Cable modem signal levels outside acceptable range. Upstream or downstream power needs adjustment."},
    {"id": "ERR-034", "name": "Multicast Storm Detected",     "description": "Excessive multicast traffic detected on network segment. Storm control activated."},
    {"id": "ERR-035", "name": "USB Storage Mount Failed",     "description": "External USB storage device not recognized or filesystem corruption detected."},
    {"id": "ERR-036", "name": "IGMP Snooping Error",          "description": "Multicast group management protocol error causing multicast traffic to flood all ports."},
    {"id": "ERR-037", "name": "Band Steering Failed",         "description": "Client device refused to connect to preferred 5GHz band, remaining on 2.4GHz."},
    {"id": "ERR-038", "name": "Captive Portal Redirect Failed","description": "Guest network captive portal not loading. May be DNS or certificate related."},
    {"id": "ERR-039", "name": "PoE Negotiation Failure",      "description": "PoE handshake with powered device failed. Device not receiving power."},
    {"id": "ERR-040", "name": "Configuration Backup Corrupt",  "description": "Saved configuration backup file is corrupted and cannot be restored."},
]

# Error-to-product mapping (which products are affected by which errors)
error_products = {
    "ERR-001": ["PROD-001", "PROD-002", "PROD-003", "PROD-010", "PROD-011", "PROD-016"],
    "ERR-002": ["PROD-001", "PROD-002", "PROD-003", "PROD-006"],
    "ERR-003": ["PROD-001", "PROD-002", "PROD-003", "PROD-004", "PROD-007", "PROD-013"],
    "ERR-004": ["PROD-007", "PROD-009", "PROD-018"],
    "ERR-005": ["PROD-007", "PROD-009", "PROD-013"],
    "ERR-006": ["PROD-001", "PROD-002", "PROD-003", "PROD-006", "PROD-013"],
    "ERR-007": ["PROD-001", "PROD-002", "PROD-003", "PROD-010", "PROD-011", "PROD-012"],
    "ERR-008": ["PROD-005", "PROD-009"],
    "ERR-009": ["PROD-001", "PROD-007", "PROD-009", "PROD-013"],
    "ERR-010": ["PROD-007", "PROD-008", "PROD-009"],
    "ERR-011": ["PROD-002", "PROD-013", "PROD-014"],
    "ERR-012": ["PROD-001", "PROD-007", "PROD-013", "PROD-017"],
    "ERR-013": ["PROD-013", "PROD-014"],
    "ERR-014": ["PROD-002", "PROD-010", "PROD-011"],
    "ERR-015": ["PROD-001", "PROD-002", "PROD-010", "PROD-012"],
    "ERR-016": ["PROD-017"],
    "ERR-017": ["PROD-013", "PROD-014"],
    "ERR-018": ["PROD-001", "PROD-002", "PROD-003", "PROD-006"],
    "ERR-019": ["PROD-001", "PROD-002", "PROD-010", "PROD-012"],
    "ERR-020": ["PROD-007", "PROD-009", "PROD-017"],
    "ERR-021": ["PROD-015"],
    "ERR-022": ["PROD-001", "PROD-002", "PROD-007"],
    "ERR-023": ["PROD-005", "PROD-009"],
    "ERR-024": ["PROD-007", "PROD-008", "PROD-009"],
    "ERR-025": ["PROD-001", "PROD-003", "PROD-016"],
    "ERR-026": ["PROD-007", "PROD-009", "PROD-012", "PROD-013"],
    "ERR-027": ["PROD-007", "PROD-009"],
    "ERR-028": ["PROD-010", "PROD-011", "PROD-012"],
    "ERR-029": ["PROD-007", "PROD-008", "PROD-009"],
    "ERR-030": ["PROD-007", "PROD-009", "PROD-013"],
    "ERR-031": ["PROD-013"],
    "ERR-032": ["PROD-013"],
    "ERR-033": ["PROD-004"],
    "ERR-034": ["PROD-007", "PROD-008", "PROD-009"],
    "ERR-035": ["PROD-001", "PROD-002", "PROD-017"],
    "ERR-036": ["PROD-007", "PROD-009"],
    "ERR-037": ["PROD-001", "PROD-002", "PROD-010", "PROD-012"],
    "ERR-038": ["PROD-001", "PROD-002", "PROD-010"],
    "ERR-039": ["PROD-007", "PROD-009", "PROD-018"],
    "ERR-040": ["PROD-001", "PROD-002", "PROD-007", "PROD-009", "PROD-013"],
}

# COMMAND ----------

# Generate solutions linked to errors
solutions = []
sol_counter = 1

solution_templates = {
    "ERR-001": ["Reset network adapter on client device and reconnect. If persists, check for WiFi interference using a spectrum analyzer.", "Reduce the connection timeout to 30s and enable fast roaming (802.11r) in router settings.", "Move the device closer to the router or install a range extender to improve signal strength."],
    "ERR-002": ["Expand DHCP pool range in router settings (Settings > Network > DHCP). Ensure no static IP conflicts.", "Restart the DHCP service: navigate to Admin > Services > DHCP and click Restart.", "Check for rogue DHCP servers on the network using network scan tools."],
    "ERR-003": ["Re-download firmware from the support portal and retry. Ensure stable power during update.", "Use the USB recovery method: download firmware to USB, insert into device, hold reset for 10 seconds.", "Enable automatic firmware updates in Settings > System > Firmware to prevent manual update issues."],
    "ERR-004": ["Identify high-power PoE devices and redistribute across switches. Consider upgrading to higher-wattage PoE switch.", "Lower PoE priority on non-critical devices in switch management interface.", "Install additional PoE injectors for devices that exceed the switch budget."],
    "ERR-005": ["Verify VLAN tags match on both ends of trunk links. Use 'show vlan' command to check configuration.", "Reset VLAN configuration to defaults and reconfigure systematically.", "Ensure native VLAN is consistent across all trunk ports."],
    "ERR-006": ["Set DNS servers to reliable providers (e.g., 8.8.8.8 and 1.1.1.1). Clear DNS cache on client.", "Check upstream internet connectivity. If ISP DNS is down, switch to alternate DNS.", "Flush the device DNS cache: Admin > Network > DNS > Flush Cache."],
    "ERR-007": ["Verify WiFi password is correct. For WPA3, ensure client device supports SAE authentication.", "Update client WiFi driver to the latest version. Disable and re-enable WiFi adapter.", "Try switching from WPA3 to WPA2/WPA3 mixed mode for broader client compatibility."],
    "ERR-008": ["Clean SFP+ connector with fiber optic cleaner. Try reseating the module.", "Verify SFP+ module is on the switch's compatibility list. Use vendor-approved transceivers.", "Update switch firmware to add support for newer SFP+ modules."],
    "ERR-009": ["Identify the process consuming CPU via Admin > Diagnostics > CPU Usage. Disable unnecessary services.", "Schedule a device reboot during maintenance window to clear memory leaks.", "Update to the latest firmware which may contain CPU optimization fixes."],
    "ERR-010": ["Identify and disconnect the cable causing the loop. Enable BPDU guard on access ports.", "Enable Rapid Spanning Tree Protocol (RSTP) for faster convergence after topology changes.", "Use loop detection LEDs on the switch to visually identify the looped port."],
    "ERR-011": ["Verify pre-shared keys match on both VPN endpoints. Check firewall allows UDP 500 and 4500.", "Ensure both ends use the same encryption and hashing algorithms (Phase 1 and Phase 2).", "Check if NAT traversal is enabled if either endpoint is behind NAT."],
    "ERR-012": ["Restart the device to clear memory. If recurring, check for memory leak in specific firmware version.", "Reduce the number of active services and monitoring processes.", "Upgrade to a device model with more RAM if workload consistently exceeds capacity."],
    "ERR-013": ["Renew the expired certificate via Admin > Security > Certificates. Import updated CA bundle.", "Verify system clock is accurate - incorrect time causes certificate validation failures.", "Import the root CA certificate manually if using a private certificate authority."],
    "ERR-014": ["Power cycle the mesh satellite node. If it doesn't reconnect, re-pair it with the root node.", "Move the satellite node closer to the root node or another connected satellite.", "Check for firmware version mismatch between root and satellite nodes."],
    "ERR-015": ["Enable AI RF Optimization to auto-select the least congested channel.", "Manually change to a less congested channel. Use a WiFi analyzer app to identify open channels.", "Reduce the channel width from 80MHz to 40MHz to reduce overlap with neighbors."],
    "ERR-016": ["Replace the failed disk immediately. Order the same model for guaranteed RAID compatibility.", "Initiate RAID rebuild from Admin > Storage > RAID > Rebuild after inserting replacement disk.", "Set up email alerts for disk health monitoring to catch failures early."],
    "ERR-017": ["Review the IDS alert details in Security > Logs. Block the source IP if confirmed malicious.", "Update IDS signatures to the latest version for improved detection accuracy.", "Configure rate limiting on the firewall to throttle port scanning attempts."],
    "ERR-018": ["Increase NAT table size in firewall settings. Default is often too low for large networks.", "Reduce NAT session timeout for idle connections to free up table entries.", "Consider deploying carrier-grade NAT (CGNAT) for very large networks."],
    "ERR-019": ["No action required - DFS channel switch is regulatory compliance. Consider using non-DFS channels (36-48).", "Enable DFS channel history logging to identify patterns and avoid problematic channels.", "If frequent, switch to 2.4GHz or non-DFS 5GHz channels for critical devices."],
    "ERR-020": ["Verify SNMP community string matches monitoring system. Reset via Admin > SNMP > Community.", "Upgrade to SNMPv3 with authentication and encryption for improved security.", "Check SNMP ACL to ensure monitoring server IP is permitted."],
    "ERR-021": ["Factory reset the Zigbee device and retry pairing within 1 meter of the gateway.", "Check if the Zigbee channel conflicts with WiFi 2.4GHz. Change Zigbee channel if needed.", "Ensure the gateway firmware supports the Zigbee profile version of the device."],
    "ERR-022": ["Review and prioritize QoS rules. Remove conflicting entries in Admin > QoS > Rules.", "Use auto-QoS feature to let the device intelligently manage traffic priorities.", "Create a QoS policy hierarchy with clear precedence rules."],
    "ERR-023": ["Inspect fiber connectors for damage or contamination. Clean with appropriate fiber cleaning tools.", "Test fiber cable with an optical power meter. Replace cable if loss exceeds specifications.", "Check SFP+ module optical power levels in switch diagnostics interface."],
    "ERR-024": ["Increase ARP table size or reduce ARP timeout to clear stale entries.", "Segment the network into smaller subnets to reduce ARP table pressure.", "Enable ARP inspection to prevent ARP spoofing attacks that fill the table."],
    "ERR-025": ["Ensure WPS is enabled on both router and client device. Press WPS button within 2 minutes.", "Try using PIN-based WPS instead of push-button method.", "Disable WPS and configure WiFi manually using SSID and password for better security."],
    "ERR-026": ["Ensure adequate ventilation around the device. Clean dust from vents and fans.", "Reduce workload or redistribute traffic to other devices to lower temperature.", "Install in a climate-controlled rack or server room with proper cooling."],
    "ERR-027": ["Verify LACP mode (active/passive) matches on both partner switches.", "Check that all member ports have identical speed, duplex, and VLAN configuration.", "Clear the port-channel configuration and recreate it step by step."],
    "ERR-028": ["Verify internet connectivity. Check proxy settings if behind a corporate firewall.", "Temporarily whitelist cloud management URLs in firewall rules.", "Fall back to local management via web browser: https://device-ip."],
    "ERR-029": ["Increase MAC table size in switch settings. Remove unused MAC entries.", "Segment network to reduce the number of MAC addresses per switch.", "Enable MAC address aging with shorter timeout to automatically remove stale entries."],
    "ERR-030": ["Order replacement power supply matching exact model number from vendor.", "Monitor remaining PSU closely. Schedule replacement during next maintenance window.", "Connect UPS to protect against total power loss with single PSU."],
    "ERR-031": ["Increase SSL inspection hardware resources or add a dedicated SSL appliance.", "Create bypass rules for known-safe high-bandwidth destinations to reduce load.", "Upgrade to a firewall model with dedicated SSL inspection ASIC."],
    "ERR-032": ["Verify BGP neighbor configuration including AS numbers and peer IP addresses.", "Check network connectivity to the BGP peer with ping and traceroute.", "Review BGP timer settings and increase hold-down timer for unstable links."],
    "ERR-033": ["Contact ISP to check signal levels at the demarcation point.", "Replace coaxial cable splitters with high-quality, low-loss versions.", "Install a signal amplifier if downstream power is consistently low."],
    "ERR-034": ["Enable storm control on switch ports to limit multicast traffic rate.", "Identify the source of excessive multicast with packet capture tools.", "Configure IGMP snooping to limit multicast to subscribed ports only."],
    "ERR-035": ["Try a different USB port. Reformat the drive to a supported filesystem (ext4 or NTFS).", "Check USB drive health with a disk utility before connecting to NAS.", "Update NAS firmware for expanded USB filesystem support."],
    "ERR-036": ["Disable and re-enable IGMP snooping on the affected VLAN.", "Ensure IGMP querier is configured on the network for proper group management.", "Update switch firmware to patch known IGMP snooping bugs."],
    "ERR-037": ["Enable aggressive band steering with higher RSSI threshold for 2.4GHz rejection.", "Check if client device supports 5GHz band and has updated drivers.", "Create a separate 5GHz-only SSID for devices that support it."],
    "ERR-038": ["Verify captive portal SSL certificate is valid and not self-signed.", "Check DNS redirect rules and ensure the portal hostname resolves correctly.", "Test with a different browser - some browsers cache DNS and bypass the portal."],
    "ERR-039": ["Verify powered device is PoE-compatible and matches the PoE standard (802.3af/at/bt).", "Try a different switch port. Reset port PoE settings in management interface.", "Test with a known-good PoE device to isolate if issue is with switch or device."],
    "ERR-040": ["Create a new configuration backup. Export via Admin > System > Backup > Export.", "If device is functional, reconfigure manually and create a fresh backup.", "Use the factory default configuration as a starting point and re-apply custom settings."],
}

for err_id, sol_list in solution_templates.items():
    for sol_text in sol_list:
        sol_id = f"SOL-{sol_counter:03d}"
        solutions.append({
            "id": sol_id,
            "name": f"Solution for {err_id}",
            "description": sol_text,
            "resolves_error": err_id
        })
        sol_counter += 1

print(f"Generated {len(products)} products, {len(features)} features, {len(errors)} errors, {len(solutions)} solutions")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Build Vertices and Edges

# COMMAND ----------

# Build vertices
vertices = []
for p in products:
    vertices.append({"id": p["id"], "entity_type": "Product", "name": p["name"], "description": p["description"], "properties": json.dumps({"category": p["category"]})})
for f in features:
    vertices.append({"id": f["id"], "entity_type": "Feature", "name": f["name"], "description": f["description"], "properties": json.dumps({})})
for e in errors:
    vertices.append({"id": e["id"], "entity_type": "Error", "name": e["name"], "description": e["description"], "properties": json.dumps({})})
for s in solutions:
    vertices.append({"id": s["id"], "entity_type": "Solution", "name": s["name"], "description": s["description"], "properties": json.dumps({"resolves_error": s["resolves_error"]})})

# Build edges
edges = []
# Product HAS_FEATURE edges
for prod_id, feat_ids in product_features.items():
    for feat_id in feat_ids:
        edges.append({"src": prod_id, "dst": feat_id, "relationship": "HAS_FEATURE", "weight": 1.0, "description": f"Product has this feature"})

# Error AFFECTS_PRODUCT edges
for err_id, prod_ids in error_products.items():
    for prod_id in prod_ids:
        edges.append({"src": err_id, "dst": prod_id, "relationship": "AFFECTS_PRODUCT", "weight": 1.0, "description": f"Error can occur on this product"})

# Solution RESOLVES_ERROR edges
for s in solutions:
    edges.append({"src": s["id"], "dst": s["resolves_error"], "relationship": "RESOLVES_ERROR", "weight": 1.0, "description": f"Solution resolves this error"})

# Solution APPLIES_TO_PRODUCT (derived: solution -> error -> product)
for s in solutions:
    err_id = s["resolves_error"]
    if err_id in error_products:
        for prod_id in error_products[err_id]:
            edges.append({"src": s["id"], "dst": prod_id, "relationship": "APPLIES_TO_PRODUCT", "weight": 0.5, "description": f"Solution can be applied to this product"})

# RELATED_PRODUCT edges (products in same category)
from itertools import combinations
category_products = {}
for p in products:
    cat = p.get("category", "Unknown")
    if cat not in category_products:
        category_products[cat] = []
    category_products[cat].append(p["id"])

for cat, prod_ids in category_products.items():
    for p1, p2 in combinations(prod_ids, 2):
        edges.append({"src": p1, "dst": p2, "relationship": "RELATED_PRODUCT", "weight": 0.3, "description": f"Products in the same {cat} category"})
        edges.append({"src": p2, "dst": p1, "relationship": "RELATED_PRODUCT", "weight": 0.3, "description": f"Products in the same {cat} category"})

print(f"Generated {len(vertices)} vertices and {len(edges)} edges")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save as Parquet and Generate PDFs

# COMMAND ----------

import pandas as pd

# Save vertices and edges as parquet
vertices_df = spark.createDataFrame(pd.DataFrame(vertices))
edges_df = spark.createDataFrame(pd.DataFrame(edges))

vertices_df.write.mode("overwrite").parquet(f"{volume_folder}/graph_data/graph_vertices")
edges_df.write.mode("overwrite").parquet(f"{volume_folder}/graph_data/graph_edges")

print("Saved graph_vertices and graph_edges parquet files")

# COMMAND ----------

# Generate PDF documentation per product
import os
from weasyprint import HTML

pdf_output_dir = f"{volume_folder}/pdf_documentation"
dbutils.fs.mkdirs(pdf_output_dir)

product_lookup = {p["id"]: p for p in products}
feature_lookup = {f["id"]: f for f in features}
error_lookup = {e["id"]: e for e in errors}
solution_lookup = {s["id"]: s for s in solutions}

for product in products:
    prod_id = product["id"]
    prod_name = product["name"]
    prod_desc = product["description"]

    # Gather features for this product
    prod_feats = product_features.get(prod_id, [])
    feat_html = ""
    for fid in prod_feats:
        f = feature_lookup[fid]
        feat_html += f"<li><strong>{f['name']}</strong>: {f['description']}</li>\n"

    # Gather errors affecting this product
    prod_errors = [eid for eid, pids in error_products.items() if prod_id in pids]
    err_html = ""
    for eid in prod_errors:
        e = error_lookup[eid]
        # Get solutions for this error
        err_solutions = [s for s in solutions if s["resolves_error"] == eid]
        sol_items = "".join([f"<li>{s['description']}</li>" for s in err_solutions])
        err_html += f"""
        <div class="error-section">
            <h3>{e['name']} ({eid})</h3>
            <p>{e['description']}</p>
            <h4>Solutions:</h4>
            <ul>{sol_items}</ul>
        </div>
        """

    html_content = f"""
    <html>
    <head><style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        h1 {{ color: #1B3A4B; border-bottom: 2px solid #1B3A4B; padding-bottom: 10px; }}
        h2 {{ color: #2C5F7C; margin-top: 30px; }}
        h3 {{ color: #3A7CA5; }}
        .error-section {{ background: #f8f9fa; padding: 15px; margin: 10px 0; border-left: 4px solid #FF6B35; }}
        ul {{ line-height: 1.8; }}
    </style></head>
    <body>
        <h1>{prod_name} - Technical Documentation</h1>
        <p><strong>Product ID:</strong> {prod_id}</p>
        <p>{prod_desc}</p>

        <h2>Features</h2>
        <ul>{feat_html}</ul>

        <h2>Known Issues and Troubleshooting</h2>
        {err_html if err_html else "<p>No known issues documented.</p>"}
    </body>
    </html>
    """

    # Write PDF via temp file
    local_tmp = f"/tmp/{prod_name.replace(' ', '_')}.pdf"
    HTML(string=html_content).write_pdf(local_tmp)
    dbutils.fs.cp(f"file:{local_tmp}", f"{pdf_output_dir}/{prod_name.replace(' ', '_')}.pdf")
    os.remove(local_tmp)

print(f"Generated {len(products)} PDF documents in {pdf_output_dir}")
