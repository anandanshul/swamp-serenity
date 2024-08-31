---
title: WebSockets
draft: false
tags:
---
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