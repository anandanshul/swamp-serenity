---
title: HTTP
draft: false
tags:
---
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