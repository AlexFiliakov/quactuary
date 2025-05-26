powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
set PATH=%PATH%;C:\Users\alexf\.local\bin
@REM Mainly need this on subsequent server updates:
mcp install quactuary/quactuary/mcp/server.py --name "quActuary" --with-editable .
@REM Then restart Claude Desktop
