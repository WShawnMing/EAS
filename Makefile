SHELL := /bin/zsh
.DEFAULT_GOAL := help

.PHONY: help dev clean deepclean

help: ## 显示命令及其安全范围
	@echo "EAS 开发命令"
	@echo ""
	@echo "  make dev        使用 uv 创建或同步 .venv 与开发依赖；不改源码"
	@echo "  make clean      仅删除项目内可再生的 Python/测试/覆盖率/lint/构建缓存"
	@echo "  make deepclean  先执行 clean，再删除本项目 .venv 与可再生工具环境"
	@echo "  make help       显示本帮助"
	@echo ""
	@echo "安全边界：clean/deepclean 不删除源码、README、docs、.git 或用户数据。"

dev: ## 使用 uv 创建或同步开发环境
	@command -v uv >/dev/null 2>&1 || { echo "错误：未找到 uv，请先安装 uv。" >&2; exit 127; }
	@UV_NO_PROGRESS=1 uv sync --dev

clean: ## 删除可再生缓存与构建产物，保留环境和项目内容
	@find "$(CURDIR)" \
		\( -path "$(CURDIR)/.git" -o -path "$(CURDIR)/.venv" \) -prune -o \
		-type d -name '__pycache__' -exec rm -rf -- {} +
	@find "$(CURDIR)" \
		\( -path "$(CURDIR)/.git" -o -path "$(CURDIR)/.venv" \) -prune -o \
		-type f \( -name '*.pyc' -o -name '*.pyo' \) -delete
	@rm -rf -- \
		"$(CURDIR)/.pytest_cache" \
		"$(CURDIR)/.mypy_cache" \
		"$(CURDIR)/.ruff_cache" \
		"$(CURDIR)/.hypothesis" \
		"$(CURDIR)/htmlcov" \
		"$(CURDIR)/build" \
		"$(CURDIR)/dist"
	@find "$(CURDIR)" -maxdepth 1 -type f \
		\( -name '.coverage' -o -name '.coverage.*' -o -name '.dmypy.json' \) \
		-delete
	@find "$(CURDIR)" -maxdepth 1 -type d -name '*.egg-info' -exec rm -rf -- {} +
	@echo "已清理可再生缓存与构建产物。"

deepclean: clean ## 额外删除本项目虚拟环境和工具环境
	@rm -rf -- \
		"$(CURDIR)/.venv" \
		"$(CURDIR)/.tox" \
		"$(CURDIR)/.nox" \
		"$(CURDIR)/.uv-cache"
	@echo "已深度清理本项目的虚拟环境与可再生工具状态。"
