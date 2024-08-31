---
title: Networking Basics
draft: false
tags:
---
Here's a well-structured technical article based on the provided transcript. The article is tailored for engineering students and includes clear explanations, code snippets, and labeled ASCII illustrations to aid understanding.

---

# Networking Basics

Networking is a crucial concept in system design, especially for system design interviews and practical applications. Understanding how computers communicate over a network is essential for developing robust software solutions. In this article, we'll explore the basics of networking, covering IP addresses, data packets, communication protocols, and ports. We'll use simple analogies and illustrations to help demystify these concepts.

## 1. Understanding IP Addresses

### What is an IP Address?

An IP address (Internet Protocol address) uniquely identifies a machine on a network, much like how a postal address uniquely identifies a house. For computers to communicate over the internet, they need to know each other's IP addresses.

### IPv4 and IPv6

IP addresses can come in two versions: IPv4 and IPv6.

- **IPv4:** This version uses 32-bit addresses, allowing for approximately 4 billion unique addresses. An IPv4 address is typically written as four sets of numbers separated by dots, for example, `192.168.1.1`. Each number can range from 0 to 255.

- **IPv6:** To overcome the limitations of IPv4, IPv6 uses 128-bit addresses, providing a vastly larger pool of addresses. IPv6 addresses are written in hexadecimal format and separated by colons, such as `2001:0db8:85a3:0000:0000:8a2e:0370:7334`.

### ASCII Illustration of IPv4 Address:

```
+-------------------+
| 192 | 168 |  1 |  1 |
+-------------------+
   |      |      |      |
  0-255 0-255 0-255 0-255
```

## 2. Sending Data Over the Network: IP Packets

When data is sent from one computer to another over the internet, it's broken down into smaller units called **packets**. Each packet includes the data being sent and metadata that describes the data, such as the source and destination IP addresses.

### Internet Protocol (IP)

The Internet Protocol governs how data is sent from one machine to another over a network. Each IP packet has a header that contains metadata, such as:

- Source IP Address
- Destination IP Address
- Other control information

### Analogy: Mailing a Letter

Sending data over a network is similar to mailing a letter:

- **Envelope** = Packet
- **Address on Envelope** = IP Address
- **Contents of the Envelope** = Data being sent

```
+-------------------------------+
|          IP Packet            |
| +---------+-----------------+ |
| |  Header |      Data       | |
| +---------+-----------------+ |
+-------------------------------+
     | Source IP | Destination IP |
     | 192.168.1.1  |  93.184.216.34 |
```

## 3. Breaking Down Data: TCP and UDP

Sometimes, we need to send large amounts of data that can't fit into a single packet. For example, sending a book through the mail requires splitting it into multiple envelopes. Similarly, we use protocols like TCP (Transmission Control Protocol) and UDP (User Datagram Protocol) to manage data transmission.

### Transmission Control Protocol (TCP)

TCP ensures that data is sent reliably and in the correct order. It breaks down the data into packets, assigns sequence numbers, and reassembles them at the destination.

### ASCII Illustration of TCP Packet:

```
+------------------------------+
|          TCP Packet          |
| +---------+----------------+ |
| | TCP     |    Data        | |
| | Header  |                | |
| +---------+----------------+ |
+------------------------------+
   | Sequence Number |  Data  |
```

### User Datagram Protocol (UDP)

UDP, on the other hand, sends packets without establishing a connection or ensuring the packets arrive in order. It's faster but less reliable than TCP, making it suitable for applications like video streaming where speed is critical.

## 4. Communication Ports

Ports serve as communication endpoints in a networked machine. They are like doors that open specific channels of communication between machines. Common ports include:

- **Port 80:** Default port for HTTP traffic
- **Port 443:** Default port for HTTPS traffic
- **Port 22:** Used for SSH (Secure Shell)

Each machine can have up to 65,535 ports (using a 16-bit value), with specific applications using predefined ports to communicate.

### Example of Ports:

```
+-------------------+
|      Machine      |
| +---------------+ |
| |   Ports       | |
| +---+---+---+---+ |
| |80 |443|22 |...| |
| +---+---+---+---+ |
+-------------------+
```

## 5. Public vs. Private IP Addresses

- **Public IP Addresses:** Accessible over the internet, used by servers that need to be publicly reachable.
- **Private IP Addresses:** Used within local networks (LANs), not directly accessible over the public internet. These are typically assigned by routers to devices in a home or office network.

### Example of Public and Private IP Address:

```
+---------------------------------+
|         Internet                |
| +---------------------------+   |
| |        Public IP          |   |
| |    203.0.113.5            |   |
| +------------+--------------+   |
|              |                  |
+--------------+------------------+
               |
        +------+------+
        |  Router      |
        +------+------+
               |
    +----------+---------+
    | Private IPs        |
    | +-------+--------+ |
    | | 192.168.0.2    | |
    | +-------+--------+ |
    | | 192.168.0.3    | |
    | +-------+--------+ |
    +--------------------+
```

## 6. Localhost and Reserved IPs

- **Localhost (`127.0.0.1`):** A special IP address reserved for a machine to refer to itself. It’s used for testing and development purposes.
  
### Accessing Applications Locally:

```shell
# Accessing a locally running application
http://localhost:4200
# or
http://127.0.0.1:4200
```

## 7. Conclusion

Understanding networking basics like IP addresses, packets, protocols, and ports is essential for system design. These concepts form the backbone of how data is transmitted and received across the internet. For software engineers, it's crucial to grasp these fundamentals to design efficient and reliable systems.

---

This article should provide engineering students with a foundational understanding of networking concepts essential for system design. Let me know if you’d like to refine any part or add more specific details!