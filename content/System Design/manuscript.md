Computer Architecture


Computer Architecture
=====================

Before diving into the system design process, it is paramount to understand the building blocks of a computer, their importance, and the role they play in designing systems.

* * *

### Components

#### Disk

A disk is the primary storage device in a computer. It is **persistent**, meaning the data will be persisted regardless of the state of the machine (turned on or off). Most modern computers store information in disk on the order of **TBs** (terabytes).

Recall that a byte is made up of 8-bits, and bit is the smallest unit of measure in computers - a binary measure represented by 000 or 111. A terabyte is 101210^{12}1012 bytes, or a trillion bytes. A disk storage device like a USB drive might have storage on the order of **GBs** (gigabytes), which is 10910^9109, or one billion bytes.

You might have come across the terms: **HDD** and **SDD**. Hard disk drives (HDD) and Solid-state drive (SSD) are both persistent storage devices, with the latter one being more popular and faster. However, it does generally cost a bit more.

> HDDs are mechanical, and have a read/write head. The older they get, the more wear and tear they collect which slows them down overtime. SSDs are significantly faster because they do not have moving parts, and rely on reading and writing data electronically (similar to RAM).

![image](https://imagedelivery.net/CLfkmk9Wzy8_9HRyug4EVA/42dfcd00-bdf1-44ac-cfb9-6b3811e65000/sharpen=1)

#### RAM

Random Access Memory is also used for storing information but is typically a lot smaller in size compared to disk drives. RAM sizes generally vary from 1GB - 128GB because RAM is _much_ more expensive than disk space. The benefit is that reading and writing to RAM is _significantly_ faster than disk. For instance, writing 1 MB of data to RAM might take on the order of microseconds (millionths of a second, or 1/1061/10^61/106 seconds), while writing the same amount to a disk might take on the order of milliseconds (thousandths of a second, or 1/1031/10^31/103 seconds)."

> Note: these numbers are rough estimates, and can change as hardware technology improves.

RAM keeps the applications you have open in memory, include any `variables` your program has allocated. It is often described as volatile memory, meaning that the data gets erased once the computer is shut down. This is why it is important to save your work to disk before shutting down your computer.

It is important to note that the RAM and disk do not directly communicate with each other. They rely on the CPU to facilitate data transfer between them.

![image](https://imagedelivery.net/CLfkmk9Wzy8_9HRyug4EVA/881a7a37-ad2e-4bc1-5588-fa77e0a47800/sharpen=1)

#### CPU

The central processing unit (CPU) is the intermediary between the RAM and disk. Also referred to as the "brain" of the computer; it reads/writes from the RAM and disk(s).

For example, when you write code and run it, your code is translated into a set of binary instructions stored in RAM. This sentence could be made clearer: "The CPU reads and executes these instructions, which may involve manipulating data stored elsewhere in RAM or on disk. An example of reading from disk, would be opening a file in your file system and reading it line-by-line.

All computations are done within the CPU, such as addition/subtraction/multiplication etc. This occurs in a matter of milliseconds. It fetches instructions from the RAM, decodes those instructions and executes the decoded instructions. On the lowest level, all these instructions are represented as bytes.

The CPU also consists of a cache. A cache is extremely fast memory that lies on the same die as the CPU.

![image](https://imagedelivery.net/CLfkmk9Wzy8_9HRyug4EVA/cbd90cba-60fd-4a0e-47d0-46ddb8ae8400/sharpen=1)

#### Caches

Most CPUs have an L1, L2, and L3 cache, which are physical components that are much faster than RAM, but they only stores data on the order of KBs or tens of MBs.

Whenever a read operation is requested, the cache is checked before the RAM and the disk. If the data requested is in the cache, and is unchanged since the last time it was accessed, it will be fetched from the cache, and not the RAM. Reading and writing to the cache is a lot faster than RAM and disk. It is up to the operating system to decide what gets stored in the cache.

Caching is an important concept applied in _many_ areas beyond computer architecture. For example, web browsers use cache to keep track of frequently accessed web pages to load them faster. This stored data might include HTML, CSS, JavaScript, and images, among other things. If the data is still valid from the time the page was cached, it will load faster. But in this case, the browser is using the disk as cache, because making internet requests is a lot slower than reading from disk.

![image](https://imagedelivery.net/CLfkmk9Wzy8_9HRyug4EVA/159f5851-56ef-474e-ecd7-d07b3c552900/sharpen=1)

> The cache being part of the CPU is only part of the reason why it is faster than RAM. Cache is what's known as SRAM. If you are interested, you might view this resource from [MIT](https://ocw.mit.edu/courses/6-004-computation-structures-spring-2017/pages/c14/c14s1/#19), which gives a gentle introduction.

* * *

![image](https://imagedelivery.net/CLfkmk9Wzy8_9HRyug4EVA/e2c12423-9dec-47b7-f471-a4a057ad9000/sharpen=1)

> The visual above demonstrates how the CPU, RAM, Cache, and the disk interact with each other on an abstract level.

### Moore's Law

Moore's Law is an observation, which suggests that the number of transistors in a CPU double every two years. Looking at the visual below, it looks like a linear graph, but looking at the scale on the y-axis explains that it is exponential. So while the number of transistors doubles, the cost of computers tends to halve. In the recent years however, the number of transistors, and thus the speed of the computers, has begun to plateau.

![image](https://imagedelivery.net/CLfkmk9Wzy8_9HRyug4EVA/14f67968-0efc-4264-4af3-915ca12b5500/sharpen=1)

Closing Notes
=============

All three memory types have their own use. If you are writing an essay, and the computer were to shut down, you would want it stored on a disk drive. While multi-tasking, you would want the programs opened to be stored in the RAM.

Now that we have gone over computer architecture, we can start understanding application architecture, including what a distributed system is, which provides the basic anatomy of a production application from the perspective of a developer and an end-user.

Design Requirements


## Design Requirements

Designing large-scale systems is crucial in both real-world applications and system design interviews. The main goal is to build effective systems that solve big problems while navigating trade-offs and making optimal choices. Unlike typical technical topics that rely on memorization, system design emphasizes understanding trade-offs and making thoughtful decisions to create robust and scalable architectures. In this article, we'll explore the fundamental aspects of system design, including data movement, storage, transformation, and key metrics like availability and reliability.

## Core Concepts of System Design

### 1. Moving Data

At its core, system design revolves around moving data, whether within a single computer (e.g., between RAM and CPU) or across multiple machines, data centers, or even globally distributed networks. Moving data isn't always straightforward, especially when dealing with latency, bandwidth limitations, and reliability across networks. Ultimately, moving data efficiently is a critical component of designing scalable systems.

### 2. Storing Data

Storing data involves choosing the right storage mechanisms based on persistence, speed, and scalability needs. For instance:
- **RAM**: Fast but volatile; data is lost on power loss or reboot.
- **Disk Storage**: Persistent but slower; suitable for data that must survive crashes.
  
In larger systems, data might be stored in various ways, such as databases, blob stores, file systems, or distributed file systems, each with distinct trade-offs. For example, databases are often used for structured data, while blob stores might be more suitable for unstructured or large binary data.

### 3. Transforming Data

Transforming data involves processing it to derive meaningful insights or actions. This could include aggregating logs to calculate success rates, encoding videos for storage, or processing large data sets to generate reports. Most applications, whether it's YouTube or Twitter, fundamentally perform these three operations: moving, storing, and transforming data.

## Key Metrics in System Design

### 1. Availability

Availability measures how often a system is operational and accessible. It’s typically expressed as a percentage:
\[
\text{Availability} = \frac{\text{Uptime}}{\text{Uptime} + \text{Downtime}}
\]

For instance, if a system is available 23 hours out of 24, its availability is approximately 96%. Availability is commonly measured in "nines":
- **99% (two nines)**: Approximately 3.65 days of downtime per year.
- **99.9% (three nines)**: About 8.76 hours of downtime per year.
- **99.99% (four nines)**: Around 52.56 minutes of downtime per year.
- **99.999% (five nines)**: Roughly 5.26 minutes of downtime per year.

Achieving high availability is crucial, especially for critical services like online stores, where downtime can result in significant revenue loss.

### 2. Reliability, Fault Tolerance, and Redundancy

- **Reliability** refers to the probability that a system will function without failure over a given period. Unlike availability, reliability doesn't just mean the system is accessible but also that it consistently performs its intended functions correctly.
  
- **Fault Tolerance** is the ability of a system to continue operating correctly in the event of a failure of some of its components. A fault-tolerant system can handle hardware failures, software bugs, and other unexpected disruptions without complete system failure.
  
- **Redundancy** is a strategy used to enhance reliability and fault tolerance by adding extra components that can take over in case of failure. For example, using multiple servers for the same task ensures that if one server goes down, others can take over, reducing the risk of a single point of failure.

### 3. Throughput

Throughput measures the rate at which a system processes tasks, typically expressed as operations per second (e.g., requests per second for servers or queries per second for databases). High throughput is essential for systems that handle large volumes of data or requests, and can be increased through vertical scaling (adding more resources to a single server) or horizontal scaling (adding more servers).

#### Vertical vs. Horizontal Scaling

- **Vertical Scaling**: Increasing the capacity of a single server by adding more CPU, RAM, or disk space. It’s simpler but has physical limits.
- **Horizontal Scaling**: Adding more servers to distribute the load, which offers more flexibility but adds complexity in terms of load balancing and data consistency.

### 4. Load Balancing and Scaling Strategies

Load balancing is crucial for distributing incoming requests evenly across multiple servers, ensuring no single server becomes a bottleneck. This can be achieved through different strategies, such as round-robin, least connections, or IP hash. Effective load balancing enhances both availability and reliability by preventing overloads and distributing traffic efficiently.

## Conclusion

System design is a nuanced field focused on making informed decisions to balance various trade-offs. Whether it's choosing how to move, store, or transform data, or optimizing for availability, reliability, and throughput, each decision can significantly impact the system's overall performance and resilience. By understanding these core concepts and metrics, engineers can design robust systems that meet the demands of both users and business objectives. Remember, the key to successful system design lies not in finding perfect solutions, but in making thoughtful compromises and continuously iterating on the design based on real-world feedback.

Networking Basics


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

TCP and UDP


Here's a structured technical article on TCP and UDP based on the provided video transcript, tailored for engineering students:

---

## Introduction to TCP and UDP

When it comes to transmitting data over the internet, two fundamental protocols dominate the conversation: **TCP (Transmission Control Protocol)** and **UDP (User Datagram Protocol)**. Understanding these transport layer protocols is crucial for software developers, as they form the backbone of data communication on the internet. This article will provide an in-depth look at both protocols, highlighting their features, differences, and when to use each.

## TCP: Transmission Control Protocol

**TCP** is a connection-oriented protocol designed to ensure reliable communication between devices. It operates on top of the Internet Protocol (IP), making it part of the broader TCP/IP suite, often simply referred to as the internet protocol suite. Here's a breakdown of the key features and functionalities of TCP:

### 1. **Reliability and Data Integrity**

TCP is renowned for its reliability. It ensures that data packets are delivered in the correct order and without errors. Let's consider a scenario where you send several packets of data:

```
+---+  +---+  +---+  +---+
| 1 |  | 2 |  | 3 |  | 4 |
+---+  +---+  +---+  +---+
```

TCP handles the ordering of packets so that if they arrive out of order, like this:

```
+---+  +---+  +---+  +---+
| 1 |  | 3 |  | 2 |  | 4 |
+---+  +---+  +---+  +---+
```

It will reassemble them correctly:

```
+---+  +---+  +---+  +---+
| 1 |  | 2 |  | 3 |  | 4 |
+---+  +---+  +---+  +---+
```

### 2. **Error Detection and Retransmission**

Networks are inherently unreliable, with potential packet loss during transmission. TCP mitigates this by implementing error-checking and retransmission of lost packets:

```
Original Transmission: +---+  +---+  +---+  +---+
                         | 1 |  | 2 |  | 3 |  | 4 |
                         +---+  +---+  +---+  +---+

Packets Received:       +---+        +---+  +---+
                         | 1 |        | 3 |  | 4 |
                         +---+        +---+  +---+
```

In the above scenario, packet 2 is missing. TCP will detect this and request a retransmission of the missing packet:

```
Retransmission:         +---+
                         | 2 |
                         +---+
```

### 3. **Connection Establishment: The Three-Way Handshake**

TCP requires a connection to be established between the client and server before any data transfer occurs. This is achieved through a process known as the **three-way handshake**:

1. **SYN:** The client sends a synchronize (SYN) packet to the server.
   
   ```
   Client -> Server: [SYN]
   ```

2. **SYN-ACK:** The server acknowledges by sending a SYN-ACK packet back to the client.
   
   ```
   Server -> Client: [SYN, ACK]
   ```

3. **ACK:** The client responds with an acknowledgment (ACK), completing the connection setup.
   
   ```
   Client -> Server: [ACK]
   ```

### 4. **Overhead and Latency**

The features that make TCP reliable also introduce overhead. The connection setup, data ordering, and error-checking add latency and consume additional bandwidth. For applications where reliability is critical, such as web browsing (HTTP), email (SMTP), and file transfer, this trade-off is acceptable.

## UDP: User Datagram Protocol

**UDP** is a simpler, connectionless protocol that provides a fast but less reliable way to send data. It does not guarantee packet delivery, order, or error-checking, making it ideal for scenarios where speed is more important than reliability.

### 1. **No Connection Establishment**

Unlike TCP, UDP does not require a connection to be established between the client and server. Data packets, known as **datagrams**, are sent independently of each other, with no guarantee of order or delivery:

```
+---+  +---+  +---+  +---+
| 1 |  | 2 |  | 3 |  | 4 |
+---+  +---+  +---+  +---+

Sent: +---+  +---+  +---+  +---+
        | 1 |  | 2 |  | 3 |  | 4 |
        +---+  +---+  +---+  +---+

Received: +---+        +---+  +---+
           | 1 |        | 3 |  | 4 |
           +---+        +---+  +---+
```

In this case, packet 2 is lost and will not be retransmitted.

### 2. **Use Cases**

UDP is commonly used in real-time applications where speed is essential, and some data loss is tolerable:

- **Live Streaming:** For live video or audio streaming, missing a frame or two is acceptable because the priority is on maintaining real-time delivery.
  
- **Online Gaming:** In fast-paced multiplayer games, it is more important that data arrives quickly rather than in order.

### 3. **Minimal Overhead**

The lack of connection establishment and retransmission mechanisms means UDP has significantly lower overhead compared to TCP. This makes it much faster, but with the trade-off of potential data loss.

## Key Differences Between TCP and UDP

Here's a concise comparison of TCP and UDP:

| Feature              | TCP                                      | UDP                                  |
|----------------------|------------------------------------------|--------------------------------------|
| Connection Type      | Connection-oriented                      | Connectionless                       |
| Reliability          | Guaranteed (error-checking, retransmission) | Not guaranteed                      |
| Ordering             | Guaranteed                               | Not guaranteed                      |
| Speed                | Slower due to overhead                   | Faster, less overhead               |
| Use Cases            | Web browsing, email, file transfer       | Live streaming, online gaming, DNS  |

## Conclusion

Both TCP and UDP serve vital roles in data communication, each suited for different use cases. TCP is the go-to choice for applications that require reliable, ordered data transmission, while UDP is ideal for real-time applications where speed is paramount, and occasional data loss is acceptable. Understanding these protocols' strengths and trade-offs will enable you to make informed decisions when developing networked applications.

---

Feel free to adjust any specific aspects, such as adding code snippets, further illustrations, or exploring more in-depth explanations on certain points!

Domain Name System (DNS)


Here's a structured article based on the provided transcript about the Domain Name System (DNS). I'll break it down according to your preferred format, maintaining a casual yet formal tone suitable for engineering students.

---

## Understanding the Domain Name System (DNS)

### Introduction

The Domain Name System, commonly known as DNS, plays a crucial role in the functioning of the internet by translating human-friendly domain names (like `google.com`) into machine-friendly IP addresses. This system, which operates on a decentralized architecture, is like the internet’s phonebook, mapping domain names to their respective IP addresses. In this article, we’ll explore the fundamentals of DNS, diving into its key components and processes, with a focus on system design principles.

### Key Concepts of DNS

#### 1. **The Problem DNS Solves**

Imagine needing to remember IP addresses like `142.250.72.14` to visit your favorite websites. DNS solves this problem by allowing users to use simple, memorable domain names instead. It works similarly to your phone’s contact list, where you save names alongside phone numbers, allowing easy communication without memorizing the actual numbers.

#### 2. **How DNS Works**

At a high level, DNS translates domain names into IP addresses, enabling your computer to locate and interact with servers. When you type `google.com` into your browser:

1. **DNS Query:** Your computer sends a request to a DNS resolver, often operated by your Internet Service Provider (ISP).
   
2. **Recursive Lookup:** If the resolver doesn’t have the IP address cached, it queries multiple DNS servers in a hierarchical manner:
   - **Root Servers:** Directs the query to the appropriate top-level domain (TLD) server (e.g., `.com`).
   - **TLD Servers:** Provides information about where the domain’s authoritative server is.
   - **Authoritative DNS Servers:** Returns the specific IP address of the requested domain.

3. **Response:** The IP address is sent back to your computer, which uses it to connect to the website’s server.

Here’s a simplified ASCII illustration of the DNS lookup process:

```
Client
  |
  |---> DNS Resolver
             |
             |---> Root DNS Server
                       |
                       |---> TLD DNS Server (.com)
                                 |
                                 |---> Authoritative DNS Server (google.com)
                                            |
                                            |---> Returns IP Address
```

#### 3. **DNS Records and Types**

DNS uses various types of records to store different information. The most common is the **A Record** (Address Record), which maps domain names to IPv4 addresses. Other important records include:

- **AAAA Record:** Maps domain names to IPv6 addresses.
- **CNAME Record:** Canonical Name Record, which redirects one domain to another.
- **MX Record:** Mail Exchange Record, used for email routing.

### Code Snippet: Using `nslookup` to Find IP Addresses

You can use the command-line tool `nslookup` to manually query DNS servers. Here’s an example:

```bash
$ nslookup google.com
```

This command returns the IP address of `google.com`, demonstrating the DNS lookup process. By entering this IP address directly into your browser, you’ll reach the same website, showing that both domain names and IP addresses can lead to the same destination.

### The Role of ICANN and Domain Registrars

The Internet Corporation for Assigned Names and Numbers (ICANN) is the organization responsible for managing domain names globally, ensuring each domain is unique and not duplicated. However, ICANN itself doesn’t sell domains. Instead, it accredits domain registrars (like GoDaddy, Namecheap, and Google Domains) that act as middlemen for purchasing domains.

### Components of a Domain

A domain is composed of multiple parts:

1. **Protocol:** Typically `http` or `https`, specifying the communication protocol.
2. **Top-Level Domain (TLD):** The suffix, such as `.com`, `.org`, or country-specific codes like `.jp`.
3. **Domain Name:** The primary, registered part (e.g., `google` in `google.com`).
4. **Subdomain:** Optional prefix that extends the domain, such as `www` or `mail`.
5. **Path and Query Parameters:** The remainder of the URL that specifies resources or actions within the domain.

Here's an example URL breakdown:

```
https://www.google.com/search?q=dns
|      |    |       |       |    |
Protocol|   | Primary Domain|    Path and Query Parameters
       TLD  Subdomain
```

### Conclusion

DNS is an indispensable component of the internet, facilitating user-friendly access to websites by mapping domain names to IP addresses. Understanding its operation, from recursive lookups to the role of authoritative servers, provides valuable insights into system design. While often simplified in general use and interviews, the detailed knowledge of DNS can help you appreciate the complex, decentralized infrastructure that supports our daily internet activities.

### Additional Resources

- [DNS in Detail by Cloudflare](https://www.cloudflare.com/learning/dns/what-is-dns/)
- [ICANN’s Role in Internet Governance](https://www.icann.org/)

---

This article outlines the basics of DNS in a structured manner, with key concepts and practical examples that make it easier to grasp the system's workings. Let me know if you need further refinements or additional sections!

HTTP


### Introduction to System Design: HTTP Limitations and the Role of WebSockets

In the world of web development, the Hypertext Transfer Protocol (HTTP) has been the backbone of communication between clients and servers. However, as web applications evolved, the limitations of HTTP became apparent, especially in scenarios requiring real-time communication. In this article, we'll explore these limitations and introduce WebSockets, a protocol designed to address the gaps left by HTTP.

### Understanding HTTP and Its Limitations

HTTP is a stateless protocol primarily used for transmitting hypermedia documents, such as HTML. It operates in a request-response cycle, where the client sends a request to the server, and the server responds with the requested data. While this works well for most web applications, it falls short in scenarios that require continuous, real-time communication, like live chat applications.

**Key Limitations of HTTP:**
- **Statelessness:** Each HTTP request is independent, meaning the server does not retain information between requests. This makes maintaining continuous communication challenging.
- **Polling:** To simulate real-time communication, HTTP requires frequent polling, where the client repeatedly sends requests to check for new data. This is inefficient and can lead to unnecessary network traffic.
- **Latency:** The interval between polls can cause delays in receiving new data, which is unacceptable in real-time applications.

### Introducing WebSockets: A Solution for Real-Time Communication

WebSockets is a protocol that provides full-duplex communication channels over a single, long-lived connection. Unlike HTTP, WebSockets allows data to flow in both directions without the need for the client to repeatedly send requests.

**Key Features of WebSockets:**
- **Persistent Connection:** After the initial handshake, a WebSocket connection remains open, enabling continuous data exchange between the client and server.
- **Bi-Directional Communication:** Data can be sent and received simultaneously, making WebSockets ideal for real-time applications like chat systems and live feeds.
- **Reduced Latency:** Since the connection is persistent, there is minimal delay in data transmission, ensuring real-time updates.

### Use Case: Building a Chat Application with WebSockets

To understand the advantages of WebSockets, let's consider a chat application. In a typical HTTP-based setup, the client would need to poll the server at regular intervals to check for new messages. This approach not only increases latency but also consumes more resources due to the constant opening and closing of connections.

With WebSockets, the process is streamlined:

1. **WebSocket Handshake:** The client initiates an HTTP request to the server to establish a WebSocket connection. If successful, the server responds with a status code of `101 Switching Protocols`, upgrading the connection to WebSocket.
2. **Persistent Connection:** Once established, the WebSocket connection remains open, allowing the server to push new messages to the client in real-time.
3. **Real-Time Communication:** The client and server can now exchange messages continuously without the overhead of repeatedly establishing new connections.

```python
# Example of a WebSocket connection in Python
import asyncio
import websockets

async def chat():
    uri = "ws://localhost:8000"
    async with websockets.connect(uri) as websocket:
        # Send a message
        await websocket.send("Hello, Server!")
        # Receive a response
        response = await websocket.recv()
        print(f"Received: {response}")

asyncio.get_event_loop().run_until_complete(chat())
```

### ASCII Illustration: WebSocket Handshake and Communication Flow

```
+--------+       HTTP Request       +--------+
| Client | -----------------------> | Server |
+--------+                          +--------+
    |                                   |
    |       HTTP 101 Switching          |
    | <------------------------------- |
    |                                   |
    |  WebSocket Connection Established |
    | <------------------------------> |
    |      Bi-Directional Data Flow     |
    |                                   |
```

### Conclusion: WebSockets in Modern Web Applications

WebSockets have revolutionized the way we think about real-time communication on the web. By providing a persistent, bi-directional connection, they eliminate the need for inefficient polling and reduce latency, making them the protocol of choice for applications requiring instant data exchange.

Whether you're building a chat application, live feed, or any other system that demands real-time updates, WebSockets offer a robust solution that surpasses the limitations of HTTP. As web technologies continue to evolve, understanding and leveraging WebSockets will be crucial for creating responsive and efficient web applications.

### Key Topics Covered
- Limitations of HTTP: Statelessness, Polling, and Latency
- Introduction to WebSockets: Features and Benefits
- Use Case: Building a Real-Time Chat Application
- WebSocket Handshake and Bi-Directional Communication

### Additional Resources
- [MDN Web Docs on WebSockets](https://developer.mozilla.org/en-US/docs/Web/API/WebSockets_API)
- [WebSocket Protocol Specification](https://tools.ietf.org/html/rfc6455)

This article should give engineering students a clear understanding of the role of WebSockets in system design and how it addresses the challenges posed by HTTP.

WebSockets


### Understanding Application Layer Protocols in System Design

In this article, we'll delve into the intricacies of application layer protocols, focusing primarily on the **client-server model**, **Remote Procedure Calls (RPCs)**, and the most widely used protocols in system design interviews: **HTTP** and **WebSockets**. Understanding these concepts is crucial for any software engineer, particularly those preparing for system design interviews.

---

#### The Client-Server Model

The **client-server model** is the foundation of many networked applications. At its core, it involves a client (often an end-user device like a desktop or mobile phone) making requests to a server, which processes these requests and sends back a response. This interaction is straightforward but underpins most of the modern web.

```ascii
   +---------+        Request          +---------+
   |  Client |  ------------------->  |  Server |
   | (Browser)|                        |(google.com)|
   +---------+        Response         +---------+
```

In this model, the **client** is the entity that initiates the request, and the **server** is the one that processes and responds to it. However, it's crucial to note that the client doesn't always have to be a user-facing application like a browser. It could also be another server, making what is known as a **server-to-server request**.

For instance, when you visit `google.com`, your browser (the client) sends a request to Google's servers. But within Google's infrastructure, those servers may communicate with other servers to fulfill your request, where one server acts as the client and another as the server.

---

#### Remote Procedure Calls (RPCs)

An essential concept in distributed computing is the **Remote Procedure Call (RPC)**. This allows code on one machine to execute a procedure (or function) on a remote machine as if it were local. The term may seem outdated, but it's fundamental to understanding many modern application protocols.

Consider a scenario where you're on `YouTube.com`, and your browser needs to fetch a list of recommended videos. The code to generate this list resides on YouTube's servers, not on your local machine. Through an RPC, your browser sends a request to execute this code on the server, which then responds with the data.

```ascii
   +---------+        RPC Request          +---------+
   |  Client |  ------------------->  |  Server |
   | (Browser)|                        |(YouTube Server)|
   +---------+        RPC Response         +---------+
```

While the term "RPC" is broad, most modern application protocols, including **HTTP** and **WebSockets**, can be thought of as specialized forms of RPCs.

---

#### HTTP: The Protocol of the Web

**Hypertext Transfer Protocol (HTTP)** is the backbone of the World Wide Web. It's an **application layer protocol** built on top of the **Internet Protocol (IP)** and **Transmission Control Protocol (TCP)**. HTTP is a **request-response** protocol, meaning a client sends a request, and the server responds.

```ascii
   +---------+        HTTP Request         +---------+
   |  Client |  ------------------->  |  Server |
   | (Browser)|                        |(Web Server)|
   +---------+        HTTP Response        +---------+
```

HTTP operates on a **stateless** model, meaning each request is independent. The server doesn't retain any information about previous requests from the same client, making the protocol simple but also requiring additional mechanisms for session management.

##### HTTP Request Structure
An HTTP request consists of two main parts:

1. **Headers**: Metadata about the request, such as the method (`GET`, `POST`, etc.), the target URL, and other details like the user agent and accepted content types.
2. **Body**: The actual data being sent with the request (not always present).

##### Common HTTP Methods
- **GET**: Retrieve data from the server.
- **POST**: Send data to the server, often used for creating resources.
- **PUT**: Update existing resources on the server.
- **DELETE**: Remove resources from the server.

##### HTTP Status Codes
Status codes are crucial for understanding the outcome of a request:
- **200 OK**: The request was successful.
- **201 Created**: A new resource was successfully created (often in response to a `POST` request).
- **400 Bad Request**: The client sent an invalid request.
- **404 Not Found**: The requested resource couldn't be found.
- **500 Internal Server Error**: The server encountered an error while processing the request.

These codes provide essential feedback to developers, indicating whether the client made a mistake or the server encountered a problem.

---

#### WebSockets: Real-Time Communication

While HTTP is well-suited for traditional request-response interactions, it isn't ideal for real-time communication. This is where **WebSockets** come in. WebSockets provide a full-duplex communication channel over a single, long-lived connection, enabling real-time data exchange between the client and server.

```ascii
   +---------+       WebSocket Connection       +---------+
   |  Client | <----------------------------> |  Server |
   +---------+                                 +---------+
```

WebSockets are particularly useful for applications like live chat, real-time notifications, or online gaming, where the client and server need to exchange data continuously without the overhead of establishing new HTTP connections.

---

### Conclusion

Understanding the client-server model, RPCs, HTTP, and WebSockets is essential for modern software engineers, particularly when designing scalable systems. These protocols form the backbone of many web applications, and mastering them will undoubtedly make you a more effective developer.

If you're preparing for system design interviews, focus on the key concepts discussed here, and don't get bogged down in memorizing every detail. Instead, aim to understand how these protocols work together to build robust, scalable systems.

API Paradigms


# Understanding APIs: REST, GraphQL, and gRPC

APIs, or Application Programming Interfaces, are essential tools in modern web development, enabling communication between software applications. In this article, we'll explore three prominent API paradigms: REST, GraphQL, and gRPC. We'll focus on their definitions, key concepts, and the pros and cons of each, providing a comprehensive overview suitable for engineering students.

## Introduction to APIs

### What is an API?
An API, or Application Programming Interface, is a set of rules and tools that allow different software applications to communicate with each other. APIs are often used to request data or services from other applications, especially over the web. They can be as simple as accessing local storage in a browser or as complex as interacting with cloud services.

## REST APIs: The Backbone of Modern Web Development

### What is REST?
REST, which stands for Representational State Transfer, is an architectural style for designing networked applications. Unlike a protocol, REST is a set of guidelines for creating scalable web services. REST APIs typically operate over HTTP and are stateless, meaning each request from a client to a server must contain all the information needed to understand and process the request.

#### Key Characteristics of REST:
- **Statelessness:** No client context is stored on the server between requests.
- **Resources:** Everything in REST is considered a resource, and each resource is identified by a URL.
- **HTTP Methods:** REST uses standard HTTP methods such as GET, POST, PUT, DELETE, etc., to perform operations on resources.
- **Scalability:** The stateless nature of REST allows horizontal scaling, where multiple servers can handle requests without needing to share state information.

### Example: Pagination in REST
One of the challenges with statelessness is managing large datasets. REST handles this through pagination, which allows clients to request specific subsets of data.

```plaintext
GET /videos?limit=10&offset=20
```
In this example, the client requests 10 videos, starting from the 21st video. This approach ensures that the server doesn't need to remember the client's previous requests, enabling efficient scaling.

## GraphQL: A Flexible Alternative to REST

### What is GraphQL?
GraphQL, developed by Facebook in 2015, is a query language for APIs and a runtime for executing those queries. Unlike REST, where each resource has its own endpoint, GraphQL allows clients to request exactly the data they need in a single query. This flexibility reduces over-fetching and under-fetching of data.

#### Key Characteristics of GraphQL:
- **Single Endpoint:** GraphQL operates through a single endpoint, regardless of the type of request.
- **Query Flexibility:** Clients can specify exactly which fields they need, reducing data transfer and improving performance.
- **Post Requests:** All GraphQL queries are sent using HTTP POST requests, with the query included in the request body.

### Example: Querying Data in GraphQL
With GraphQL, clients can retrieve specific data fields without fetching unnecessary information. For instance, if a client only needs a user's profile picture and username, they can request just those fields:

```graphql
query {
  user(id: "1") {
    profilePicture
    username
  }
}
```
This query returns only the specified fields, optimizing data transfer and reducing the load on both the server and the client.

## gRPC: High-Performance APIs for Microservices

### What is gRPC?
gRPC, short for Google Remote Procedure Call, is a high-performance, open-source framework that uses HTTP/2 for transport, Protocol Buffers as the interface description language, and provides features such as authentication, load balancing, and more. gRPC is particularly well-suited for connecting microservices in distributed systems.

#### Key Characteristics of gRPC:
- **HTTP/2:** Supports multiplexing, which allows multiple requests to be sent on a single connection.
- **Protocol Buffers:** gRPC uses Protocol Buffers (Protobuf) for serialization, which is more efficient than JSON.
- **Bidirectional Streaming:** gRPC supports bidirectional streaming, allowing both client and server to send a sequence of messages.

### Example: A Simple gRPC Service
In gRPC, services are defined using Protobuf, and the server implements these services. Here's a simple example:

```protobuf
service Greeter {
  rpc SayHello (HelloRequest) returns (HelloReply);
}

message HelloRequest {
  string name = 1;
}

message HelloReply {
  string message = 1;
}
```
This Protobuf definition specifies a `Greeter` service with a `SayHello` method. The client sends a `HelloRequest`, and the server responds with a `HelloReply`.

## Conclusion: Choosing the Right API Paradigm

REST, GraphQL, and gRPC each have their own strengths and weaknesses:

- **REST** is ideal for standard web services that require scalability and simplicity.
- **GraphQL** offers flexibility and efficiency in data fetching, making it suitable for complex front-end applications.
- **gRPC** excels in high-performance, microservices environments, particularly where low latency and efficient communication are crucial.

Understanding these paradigms and their appropriate use cases is essential for modern software development.

Application Architecture


Application Architecture
========================

In this high-level overview, we'll explore the architecture of a production-grade application. This will serve as a foundation for the rest of the course, allowing us to delve into each component in more detail later on.

Within a production application architecture, various components work together to create a robust system. While we'll provide a brief introduction to these components here, we'll cover each one extensively in separate chapters throughout the course.

* * *

### A developer's perspective

We can start viewing this application architecture from the perspective of a developer, which will be familiar to most of you. Developers write code that is **deployed** to a **server**. For now, let's define a **server** as a computer that handles requests from another computer. This server also requires **persistent storage** to store the application's data. A server may have built-in storage, but that has its limitations in terms of size. As such, a server may talk to an external storage system (database, cloud etc). This storage may not be part of the same server, and is instead connected through a **network**.

### A user's perspective

A user is someone who makes a request from the server, usually through a web browser. In this case, the web browser is the **client** to whom the server responds to.

If a user wanted to use a front-end feature, the server will respond with the necessary JavaScript/HTML/CSS code, compiled to display what the user requested. But, what if we have a lot of users and the single server cannot handle all of the requests on its own? There is bound to be a bottleneck, either through our RAM or our CPU. To maintain performance while dealing with multiple users, we will need to scale our server.

### Scaling our server

To handle multiple requests, it might be a good idea to add more RAM or upgrade to a CPU with more cores and higher clocking speed. However, every computer has a limitation in terms of upgrades. Upgrading components _within_ the same computer is referred to as **vertical scaling**.

We can also have multiple servers running our code, and we can distribute the user requests among these servers. This way, not all users are talking to one server, which ensures that the speed of each server remains intact. This also ensures that if one server were to go down, we can direct our traffic to one of our other servers. This is known as **horizontal scaling**.

Generally, in large systems, we prefer horizontal scaling, as it is much more powerful, and can be achieved with commodity hardware (i.e., relatively inexpensive, standard hardware). However, it also requires much more engineering effort, as we need to ensure that the servers are communicating with each other, and that the user requests are being distributed evenly.

For simple applications however, vertical scaling may be sufficient and the easier solution to implement. Even some services within [Amazon Prime Video](https://www.primevideotech.com/video-streaming/scaling-up-the-prime-video-audio-video-monitoring-service-and-reducing-costs-by-90) were recently migrated from a microservice architecture to a monolithic architecture.

> But with multiple servers, what determines which requests go to which server? This is achieved through a **load balancer**. A load balancer will evenly distribute the incoming requests across a group of servers.

It's also important to remember that servers don't exist in isolation. It is highly likely that servers are interacting with external servers, through APIs. For example, the neetcode.io website interacts with other services like Stripe, through an API.

### Logging and Metrics

Servers also have **logging** services, which gives the developer a log of all the activity that happened. Logs can be written to the same server, but for better reliability they are commonly written to _another_ external server.

This gives developers insight into how the requests went, if any errors occured, or what happened before a server crashed. However, logs don't provide the complete picture. If our RAM has become the bottleneck of our server, or our CPU resources are restricting the requests being handled efficiently, we require a **metrics** service. A metric service will collect data from different sources within our server environment, such as CPU usage, network traffic etc. This allows developers to gain insights into server's behavior and identify potential bottlenecks.

### Alerts

As developers, we wouldn't want to keep checking metrics to see if any unexpected behavior exhibits itself. This would be like checking your phone every 555 minutes for a notification. It is more ideal to receive a push notification. We can program alerts so that whenever a certain metric fails to meet the target, the developers receive a push notification. For example, if 100%100\\%100% of the user requests receive successful responses, we could set an alert to be notified if this metric dips under 95%95\\%95%.

![image](https://imagedelivery.net/CLfkmk9Wzy8_9HRyug4EVA/f54abb89-ab9b-4713-874b-7568651b2800/sharpen=1)

> The visual above demonstrates (on a very high level) how the components interact with each other, and what components the users interacts with and what components the developer interacts with.

Closing Notes
=============

What we discussed above is a gentle introduction, and there is a lot more that goes into application architecture than what we just talked about. For example, how do all of these components communicate with each other. What protocols do they need to abide by? Are there ways to optimize these protocols? These components could very well be scattered across different computers, so networking is required. We will discuss all this in detail in the upcoming chapters.