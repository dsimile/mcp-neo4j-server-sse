import neo4j
import logging
from logging.handlers import RotatingFileHandler
import traceback
from contextlib import closing
from pathlib import Path
from mcp.server.models import InitializationOptions
import mcp.types as types
from mcp.server import NotificationOptions, Server
from mcp.server.sse import SseServerTransport
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.middleware.cors import CORSMiddleware
import uvicorn
from pydantic import AnyUrl
from typing import Any
from neo4j import GraphDatabase
import re

logger = logging.getLogger('mcp_neo4j_cypher_remote')
logger.info("Starting MCP neo4j Server is remote")


def is_write_query(query: str) -> bool:
    """Checks if a Cypher query contains common write clauses."""
    return (
        re.search(r"\b(MERGE|CREATE|SET|DELETE|REMOVE|ADD)\b", query, re.IGNORECASE)
        is not None
    )


class neo4jDatabase:
    def __init__(
        self,
        neo4j_uri: str,
        neo4j_username: str,
        neo4j_password: str,
        neo4j_database: str,
    ):
        """Initialize connection to the neo4j database"""
        logger.debug(f"Initializing database connection to {neo4j_uri}")
        d = GraphDatabase.driver(neo4j_uri, auth=(neo4j_username, neo4j_password))
        d.verify_connectivity()
        self.database = neo4j_database
        self.driver = d

    def _execute_query(
        self, query: str, params: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Execute a Cypher query and return results as a list of dictionaries"""
        logger.debug(f"Executing query: {query}")
        try:
            result = self.driver.execute_query(query, params, database_=self.database)
            counters = vars(result.summary.counters)
            if is_write_query(query):
                logger.debug(f"Write query affected {counters}")
                return [counters]
            else:
                results = [dict(r) for r in result.records]
                logger.debug(f"Read query returned {len(results)} rows")
                return results
        except Exception as e:
            logger.error(f"Database error executing query: {e}\n{query}")
            raise

    def close(self) -> None:
        "Close the Neo4j Driver"
        self.driver.close()


async def main(
    neo4j_url: str,
    neo4j_username: str,
    neo4j_password: str,
    neo4j_database: str = "neo4j",
    host: str = "0.0.0.0",
    port: int = 8543,
):
    logger.info(
        f"Connecting to neo4j MCP Server with DB URL: {neo4j_url}, DB name: {neo4j_database}"
    )

    db = neo4jDatabase(neo4j_url, neo4j_username, neo4j_password, neo4j_database)
    server = Server("neo4j-manager")

    # Register handlers
    logger.debug("Registering handlers")

    @server.list_tools()
    async def handle_list_tools() -> list[types.Tool]:
        """List available tools"""
        return [
            types.Tool(
                name="read-neo4j-cypher",
                description="Execute a Cypher query on the neo4j database",
                annotations={
                    "destructiveHint": False,
                    "idempotentHint": True,
                    "readOnlyHint": True,
                    "title": "Read from Neo4j Database",
                },
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Cypher read query to execute",
                        },
                    },
                    "required": ["query"],
                },
            ),
            types.Tool(
                name="write-neo4j-cypher",
                description="Execute a write Cypher query on the neo4j database",
                annotations={
                    "destructiveHint": True,
                    "idempotentHint": False,
                    "readOnlyHint": False,
                    "title": "Update Neo4j Database",
                },
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Cypher write query to execute",
                        },
                    },
                    "required": ["query"],
                },
            ),
            types.Tool(
                name="get-neo4j-schema",
                description="List all node types, their attributes and their relationships TO other node-types in the neo4j database",
                annotations={
                    "destructiveHint": False,
                    "idempotentHint": True,
                    "readOnlyHint": True,
                    "title": "Get Neo4j Database Schema",
                },
                inputSchema={
                    "type": "object",
                    "properties": {},
                },
            ),
        ]

    @server.call_tool()
    async def handle_call_tool(
        name: str, arguments: dict[str, Any] | None
    ) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
        """Handle tool execution requests"""
        try:
            if name == "get-neo4j-schema":
                results = db._execute_query(
                    """
call apoc.meta.data() yield label, property, type, other, unique, index, elementType
where elementType = 'node' and not label starts with '_'
with label, 
    collect(case when type <> 'RELATIONSHIP' then [property, type + case when unique then " unique" else "" end + case when index then " indexed" else "" end] end) as attributes,
    collect(case when type = 'RELATIONSHIP' then [property, head(other)] end) as relationships
RETURN label, apoc.map.fromPairs(attributes) as attributes, apoc.map.fromPairs(relationships) as relationships
                    """
                )
                return [types.TextContent(type="text", text=str(results))]

            elif name == "read-neo4j-cypher":
                if is_write_query(arguments["query"]):
                    raise ValueError("Only MATCH queries are allowed for read-query")
                results = db._execute_query(arguments["query"])
                return [types.TextContent(type="text", text=str(results))]

            elif name == "write-neo4j-cypher":
                if not is_write_query(arguments["query"]):
                    raise ValueError("Only write queries are allowed for write-query")
                results = db._execute_query(arguments["query"])
                return [types.TextContent(type="text", text=str(results))]

            else:
                raise ValueError(f"Unknown tool: {name}")

        except Exception as e:
            return [types.TextContent(type="text", text=f"Error: {str(e)}")]

    # --- SSE Transport Setup ---
    # Initialize SseServerTransport with the path *without* the trailing slash
    sse_transport = SseServerTransport("/message")

    async def message_sse(request: Request):
        async with sse_transport.connect_sse(
            request.scope, request.receive, request._send
        ) as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="neo4j-remote",
                    server_version="0.1.1",
                    capabilities=server.get_capabilities(
                        notification_options=NotificationOptions(),
                        experimental_capabilities={},
                    ),
                ),
            )
        return JSONResponse({"success": True})

    async def message_post(request: Request):
        try:
            return await sse_transport.handle_post_message(
                request.scope, request.receive, request._send
            )
        except Exception as e:
            logger.error(
                f"POST /message handler crashed: {e}\n{traceback.format_exc()}"
            )
            return JSONResponse(
                {"error": str(e), "traceback": traceback.format_exc()}, status_code=500
            )

    routes = [
        Route("/message", message_sse, methods=["GET"]),
        Route("/message", message_post, methods=["POST"]),
    ]

    app = Starlette(routes=routes)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # --- Add Uvicorn Server Startup ---
    # Create a Uvicorn config
    config = uvicorn.Config(app=app, host=host, port=port, log_level="info")
    # Create a Uvicorn server instance
    uvicorn_server = uvicorn.Server(config)
    # Run the server. This call is blocking until the server stops.
    # Since main is async, we need to await serve().
    logger.info(f"Starting Uvicorn server on http://{host}:{port}")
    await uvicorn_server.serve()


if __name__ == "__main__":
    import argparse
    import asyncio
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    )
    logger.setLevel(logging.DEBUG)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.error").setLevel(logging.WARNING)

    parser = argparse.ArgumentParser(
        description="Run Neo4j MCP Server using SSE transport."
    )
    parser.add_argument(
        "--url",
        required=True,
        help="Neo4j connection URL (e.g., neo4j://localhost:7687)",
    )
    parser.add_argument("--username", required=True, help="Neo4j database username")
    parser.add_argument("--password", required=True, help="Neo4j database password")
    parser.add_argument("--database", default="neo4j", help="Neo4j database name")
    parser.add_argument("--host", default="0.0.0.0", help="Host address to bind to")
    parser.add_argument("--port", type=int, default=8543, help="Port to listen on")
    args = parser.parse_args()

    try:
        asyncio.run(
            main(
                args.url,
                args.username,
                args.password,
                args.database,
                host=args.host,
                port=args.port,
            )
        )
    except Exception as e:
        logger.critical(f"Application failed: {e}", exc_info=True)
        sys.exit(1)
