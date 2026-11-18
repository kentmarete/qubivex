# Local demo: the qubivex server plus the qubivex-ui dev server.
#
#   make up       build if needed, start both in the background, wait until healthy
#   make down     stop both
#   make dev      both in the foreground; Ctrl-C stops both
#   make status   what is running
#   make logs     follow both logs
#
# Override anything on the command line, for example
#   make up MODEL=qwen3-0.6b BACKEND=metal PORT=8081

MODEL       ?= qwen2.5-0.5b-instruct
MODELS_DIR  ?= $(HOME)/.qubivex/models
BACKEND     ?= cpu
HOST        ?= 127.0.0.1
PORT        ?= 8080
ADMIN_TOKEN ?= dev-admin
SERVE_ARGS  ?=
UI_DIR      ?= ../qubivex-ui
UI_PORT     ?= 5173

BIN     := target/release/qubivex
RUN_DIR := .run
API     := http://$(HOST):$(PORT)
# Vite binds localhost, which can be IPv6 only (::1), so probe and print localhost.
UI_URL  := http://localhost:$(UI_PORT)/ui/

SERVE_CMD = $(BIN) serve --dir $(MODELS_DIR) --model $(MODEL) --backend $(BACKEND) \
	--listen $(HOST):$(PORT) --admin-token $(ADMIN_TOKEN) $(SERVE_ARGS)
VITE_CMD  = ./node_modules/.bin/vite --port $(UI_PORT) --strictPort

.DEFAULT_GOAL := help
.PHONY: help build ui-deps up server-up ui-up down status logs dev serve ui

help:
	@echo "make up | down | dev | status | logs | serve | ui | build | ui-deps"
	@echo "model $(MODEL) on $(BACKEND), API $(API), UI $(UI_URL)"

build:
	cargo build --release -p qubivex-cli

ui-deps: $(UI_DIR)/node_modules/.package-lock.json

$(UI_DIR)/node_modules/.package-lock.json: $(UI_DIR)/package-lock.json
	cd $(UI_DIR) && npm ci

$(RUN_DIR):
	@mkdir -p $(RUN_DIR)

up: build ui-deps server-up ui-up
	@echo ""
	@echo "Qubivex UI   $(UI_URL)"
	@echo "API          $(API)  (admin token: $(ADMIN_TOKEN))"
	@echo "Logs         make logs      Stop  make down"

server-up: | $(RUN_DIR)
	@if curl -sf $(API)/health >/dev/null; then \
		echo "server: already answering on $(API)"; exit 0; fi; \
	echo "server: starting $(MODEL) on $(API)"; \
	nohup $(SERVE_CMD) > $(RUN_DIR)/server.log 2>&1 & echo $$! > $(RUN_DIR)/server.pid; \
	for i in $$(seq 1 120); do \
		curl -sf $(API)/health >/dev/null && { echo "server: healthy"; exit 0; }; \
		kill -0 $$(cat $(RUN_DIR)/server.pid) 2>/dev/null || { echo "server: exited, see $(RUN_DIR)/server.log"; tail -20 $(RUN_DIR)/server.log; exit 1; }; \
		sleep 1; \
	done; echo "server: not healthy after 120 s, see $(RUN_DIR)/server.log"; exit 1

ui-up: | $(RUN_DIR)
	@if curl -sf $(UI_URL) >/dev/null; then \
		echo "ui: already answering on $(UI_URL)"; exit 0; fi; \
	echo "ui: starting on $(UI_URL)"; \
	(cd $(UI_DIR) && QUBIVEX_API=$(API) exec nohup $(VITE_CMD)) > $(RUN_DIR)/ui.log 2>&1 & echo $$! > $(RUN_DIR)/ui.pid; \
	for i in $$(seq 1 30); do \
		curl -sf $(UI_URL) >/dev/null && { echo "ui: ready"; exit 0; }; \
		kill -0 $$(cat $(RUN_DIR)/ui.pid) 2>/dev/null || { echo "ui: exited, see $(RUN_DIR)/ui.log"; tail -20 $(RUN_DIR)/ui.log; exit 1; }; \
		sleep 1; \
	done; echo "ui: not ready after 30 s, see $(RUN_DIR)/ui.log"; exit 1

down:
	@for name in ui server; do \
		f=$(RUN_DIR)/$$name.pid; \
		if [ -f $$f ] && kill -0 $$(cat $$f) 2>/dev/null; then \
			kill $$(cat $$f) && echo "$$name: stopped"; \
		else echo "$$name: not running (from make)"; fi; \
		rm -f $$f; \
	done

status:
	@curl -sf $(API)/health >/dev/null && echo "server: up on $(API) ($$(curl -s $(API)/v1/models | grep -o '"id":"[^"]*"' | cut -d'"' -f4 | tr '\n' ' '))" || echo "server: down"
	@curl -sf $(UI_URL) >/dev/null && echo "ui:     up on $(UI_URL)" || echo "ui:     down"

logs:
	@tail -n 20 -f $(RUN_DIR)/server.log $(RUN_DIR)/ui.log

# Foreground: both share this terminal; Ctrl-C stops both.
dev: build ui-deps
	@trap 'kill 0' INT TERM EXIT; \
	$(SERVE_CMD) & \
	(cd $(UI_DIR) && QUBIVEX_API=$(API) exec $(VITE_CMD)) & \
	wait

serve: build
	$(SERVE_CMD)

ui: ui-deps
	cd $(UI_DIR) && QUBIVEX_API=$(API) $(VITE_CMD)
