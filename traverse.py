#!/usr/bin/env python3
import os
import argparse
from datetime import datetime

try:
	import pyperclip
	CLIPBOARD_AVAILABLE = True
except ImportError:
	CLIPBOARD_AVAILABLE = False


FILE_SEPARATOR = "\n" + "=" * 80 + "\n"
DEFAULT_IGNORES = {
	".git", "__pycache__", "venv", ".venv",
	"node_modules", ".idea", ".vscode"
}


def write_root_header(out, root_dir: str, extensions: set[str]):
	out.write("# CONTEXT DUMP\n")
	out.write(f"# ROOT_DIR: {os.path.abspath(root_dir)}\n")
	out.write(f"# EXTENSIONS: {', '.join(sorted(extensions))}\n")
	out.write(f"# GENERATED_AT: {datetime.utcnow().isoformat()}Z\n")
	out.write(FILE_SEPARATOR)


def traverse_and_collect(
	root_dir: str,
	extensions: set[str],
	output_path: str,
	ignore_dirs: set[str],
	max_size_kb: int,
	with_line_numbers: bool,
	copy_to_clipboard: bool
):
	root_dir = os.path.abspath(root_dir)
	buffer: list[str] = []

	def emit(text: str):
		buffer.append(text)

	with open(output_path, "w", encoding="utf-8") as out:
		write_root_header(out, root_dir, extensions)
		emit(out.getvalue() if hasattr(out, "getvalue") else "")

		for current_root, dirs, files in os.walk(root_dir):
			dirs[:] = sorted(d for d in dirs if d not in ignore_dirs)
			files.sort()

			for file in files:
				if not any(file.endswith(ext) for ext in extensions):
					continue

				full_path = os.path.join(current_root, file)
				rel_path = os.path.relpath(full_path, root_dir)

				if os.path.getsize(full_path) > max_size_kb * 1024:
					text = f"# SKIPPED (too large): {rel_path}\n{FILE_SEPARATOR}"
					out.write(text)
					buffer.append(text)
					continue

				try:
					with open(full_path, "r", encoding="utf-8") as f:
						lines = f.readlines()
				except Exception as e:
					text = (
						f"# FAILED TO READ: {rel_path}\n"
						f"# ERROR: {e}\n"
						f"{FILE_SEPARATOR}"
					)
					out.write(text)
					buffer.append(text)
					continue

				header = f"# FILE: {rel_path}\n{FILE_SEPARATOR}"
				out.write(header)
				buffer.append(header)

				if with_line_numbers:
					for i, line in enumerate(lines, 1):
						row = f"{i:4d}: {line}"
						out.write(row)
						buffer.append(row)
				else:
					out.writelines(lines)
					buffer.extend(lines)

				out.write(FILE_SEPARATOR)
				buffer.append(FILE_SEPARATOR)

	if copy_to_clipboard:
		if not CLIPBOARD_AVAILABLE:
			print("⚠ Clipboard requested but pyperclip is not installed.")
			return

		joined = "".join(buffer)
		pyperclip.copy(joined)
		print("Context copied to clipboard.")


def main():
	parser = argparse.ArgumentParser(
		description="Traverse directory and dump file contents for LLM context."
	)
	parser.add_argument(
		"root",
		help="Root directory (context anchor)"
	)
	parser.add_argument(
		"-e", "--ext",
		default=".py",
		help="Comma-separated extensions (default: .py)"
	)
	parser.add_argument(
		"-o", "--output",
		default="context_dump.txt",
		help="Output file"
	)
	parser.add_argument(
		"--max-size",
		type=int,
		default=512,
		help="Max file size in KB (default: 512)"
	)
	parser.add_argument(
		"--line-numbers",
		action="store_true",
		help="Include line numbers"
	)
	parser.add_argument(
		"--clipboard",
		action="store_true",
		help="Automatically copy result to clipboard"
	)

	args = parser.parse_args()
	extensions = {e.strip() for e in args.ext.split(",")}

	traverse_and_collect(
		root_dir=args.root,
		extensions=extensions,
		output_path=args.output,
		ignore_dirs=DEFAULT_IGNORES,
		max_size_kb=args.max_size,
		with_line_numbers=args.line_numbers,
		copy_to_clipboard=args.clipboard
	)


if __name__ == "__main__":
	main()
