# neo4j-server-remote
neo4j-server-remote is an MCP server that uses Server-Sent Events (SSE) as the transport protocol.

## Overview

A Model Context Protocol (MCP) server implementation that provides database interaction and allows for graph exploration capabilities through Neo4j. This server enables the execution of Cypher graph queries, analysis of complex domain data, and supports the selection of remotely accessible databases. Inspired by [neo4j-contrib/mcp-neo4j](https://github.com/neo4j-contrib/mcp-neo4j/tree/main/servers/mcp-neo4j-cypher).

### Prompts

The server provides a demonstration prompt:

- `mcp-demo`: Interactive prompt that guides users through database operations
  - Generates appropriate database schemas and sample data

### Tools

The server offers six core tools:

#### Query Tools

- `read-neo4j-cypher`
  - Execute Cypher read queries to read data from the database
  - Input: 
    - `query` (string): The Cypher query to execute
  - Returns: Query results as array of objects

- `write-neo4j-cypher`
  - Execute updating Cypher queries
  - Input:
    - `query` (string): The Cypher update query
  - Returns: a result summary counter with `{ nodes_updated: number, relationships_created: number, ... }`

#### Schema Tools

- `get-neo4j-schema`
  - Get a list of all nodes types in the graph database, their attributes with name, type and relationships to other node types
  - No input required
  - Returns: List of node label with two dictionaries one for attributes and one for relationships

## Usage with Cline client

1.Clone the repository

```cmd
git clone https://github.com/dsimile/mcp-neo4j-server-sse.git
```

2.Install required

- Python 3.12+

```cmd
cd mcp-neo4j-server-sse

pip install -r requirements.txt
```

3.Run server

Run the MCP server using the UX command, and you can select the database of your choice. The default address ip is 0.0.0.0. The default address port is 8543.

```cmd
uv run .\src\mcp-neo4j-cypher\neo4j_server_remote.py --url bolt://localhost:7687 --username neo4j --password neo4j123 --database neo4j
```

> Note: Please ensure that Neo4j is running and accessible for remote connections.

### Released Package

Add the server to your `cline_mcp_settings.json` with configuration of 

```json
{
  "mcpServers": {
    "neo4j-remote": {
      "url": "http://0.0.0.0:8543/sse",
      "disabled": false,
      "autoApprove": []
    }
  }
}
```

## License

This MCP server is licensed under the MIT License. This means you are free to use, modify, and distribute the software, subject to the terms and conditions of the MIT License. For more details, please see the LICENSE file in the project repository.
