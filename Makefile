.PHONY: test eval chat serve mcp
test:   ; python -m unittest discover -s tests -t .
eval:   ; python -m agenthub eval --min-pass 0.9
chat:   ; python -m agenthub chat
serve:  ; pip install -e ".[server]" && python -m agenthub serve
mcp:    ; python -m agenthub mcp
